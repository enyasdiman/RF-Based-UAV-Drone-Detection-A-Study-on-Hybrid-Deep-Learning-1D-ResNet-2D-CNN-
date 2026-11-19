"""
RF Sinyalleri (Ham IQ & Spektrogram) ile Drone Var/Yok İkili Sınıflandırma (Binary Classification)
Veri Seti: Synthetic RF Spectrograms and IQ Signals for UAV Detection (2.4 GHz band)
Dizin: C:\\Users\\enyas\\Desktop\\Dataset_RF_IQ
  - iq_signals/   : 29.019 adet .npy (complex128, değişken uzunluklu [5..375] ham IQ sinyali)
  - signal_crops/ : 29.019 adet .png (RGB spektrogram kırpmaları)
  Toplam          : 58.038 dosya (Tamamı Hibrit 1D-ResNet + 2D-CNN Mimarisi ile İşlenir)
"""

import os
import re
import sys
import glob
import copy
import time
import random
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


# ==============================================================================
# 0. KONFİGÜRASYON VE HİPERPARAMETRELER
# ==============================================================================
class Config:
    # Masaüstündeki veri seti klasörünüz doğrudan burada tanımlıdır:
    DATASET_DIR = r"C:\Users\enyas\Desktop\Dataset_RF_IQ"
    IQ_DIR = os.path.join(DATASET_DIR, "iq_signals")
    CROP_DIR = os.path.join(DATASET_DIR, "signal_crops")
    LABELS_DIR = os.path.join(DATASET_DIR, "labels")  # Opsiyonel YOLO .txt klasörü

    # Veri Modu Seçimi:
    #   "hybrid"      -> 58.038 dosyanın TAMAMI (29.019 IQ + 29.019 Spektrogram çift dallı 1D+2D füzyon ağı)
    #   "iq"          -> Sadece ham IQ sinyalleri (.npy) + 1D ResNet-CNN (29.019 örnek)
    #   "spectrogram" -> Sadece spektrogram görselleri (.png) + 2D CNN (29.019 örnek)
    DATA_MODE = "hybrid"

    # Etiketleme Modu Seçimi:
    #   "bandwidth"   -> Fiziksel RF STFT bant genişliğine göre (Geniş bant DJI video iletimi >= 90 bin -> 1, Diğer -> 0)
    #   "auto"        -> labels/ klasörü varsa YOLO class_id okur; yoksa "bandwidth" modunu kullanır
    #   "readme"      -> Dosya adındaki _signal_0 -> 1, _signal_[1..N] -> 0
    LABEL_MODE = "bandwidth"

    # Sabit Boyutlar (Değişken uzunluklu verileri batch içinde birleştirmek için)
    IQ_FIXED_LEN = 256        # Ham IQ sinyalleri (5..375 örnek) -> (2, 256) [I ve Q kanalları]
    IQ_GLOBAL_SCALE = 0.05    # Güç spektral yoğunluğunu (PSD) koruyan global normalizasyon katsayısı
    IMG_SIZE = (64, 64)       # Spektrogram kırpmaları -> (3, 64, 64)

    # Eğitim Hiperparametreleri
    SEED = 42
    BATCH_SIZE = 128
    NUM_WORKERS = 0           # Windows'ta güvenli çalışma için 0
    IO_THREADS = 16           # 58.038 dosyayı çoklu iş parçacığıyla hızlı okuma
    EPOCHS = 50               # Tam 50 Epoch boyunca kesintisiz eğitim
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Early Stopping Ayarı:
    # False -> Eğitimi yarıda kesmez, belirlenen tüm EPOCHS (50) sayısını sonuna kadar tamamlar
    #          ve tüm epoch'lar arasından en düşük Val Loss değerine sahip en iyi modeli seçer.
    USE_EARLY_STOPPING = False
    EARLY_STOPPING_PATIENCE = 50

    MODEL_SAVE_PATH = os.path.join(DATASET_DIR, "best_drone_rf_model.pt")
    CM_PLOT_PATH = os.path.join(DATASET_DIR, "confusion_matrix.png")


def set_seed(seed: int = 42):
    """Tekrarlanabilirlik (reproducibility) için rastgelelik tohumlarını sabitler."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# 1. VERİ İNDEKSLEME, HIZLI ÇOKLU-THREAD YÜKLEME VE TRAIN / VAL / TEST AYIRMA
# ==============================================================================
FILENAME_PATTERN = re.compile(r"SpectrogramBTFinal_SVG_(\d+)_signal_(\d+)(?:_iq)?\.(?:npy|png)$")


def build_dataset_index(cfg: Config) -> List[Tuple[str, str, int]]:
    """
    Dizindeki tüm IQ (.npy) ve Spektrogram (.png) dosyalarını eşleştirir
    ve ikili sınıf etiketlerini (1: Drone Var, 0: Drone Yok [WiFi/Bluetooth]) oluşturur.
    """
    iq_files = sorted(glob.glob(os.path.join(cfg.IQ_DIR, "*.npy")))
    if not iq_files:
        raise FileNotFoundError(f"IQ dosyaları bulunamadı: {cfg.IQ_DIR}")

    has_yolo_labels = os.path.isdir(cfg.LABELS_DIR) and len(os.listdir(cfg.LABELS_DIR)) > 0
    active_label_mode = cfg.LABEL_MODE
    if active_label_mode == "auto":
        active_label_mode = "yolo" if has_yolo_labels else "bandwidth"

    print(f"[BİLGİ] Etiketleme modu: '{active_label_mode}' | Veri modu: '{cfg.DATA_MODE}'")

    samples: List[Tuple[str, str, int]] = []
    yolo_cache = {}

    for iq_path in iq_files:
        fname = os.path.basename(iq_path)
        match = FILENAME_PATTERN.search(fname)
        if not match:
            continue

        spec_id, sig_idx = int(match.group(1)), int(match.group(2))
        crop_fname = f"SpectrogramBTFinal_SVG_{spec_id}_signal_{sig_idx}.png"
        crop_path = os.path.join(cfg.CROP_DIR, crop_fname)

        # --- İkili Etiket (1: Drone Var, 0: Drone Yok) Hesaplama ---
        if active_label_mode == "yolo":
            label_txt = os.path.join(cfg.LABELS_DIR, f"SpectrogramBTFinal_SVG_{spec_id}.txt")
            if label_txt not in yolo_cache:
                if os.path.exists(label_txt):
                    with open(label_txt, "r", encoding="utf-8") as f:
                        yolo_cache[label_txt] = [
                            int(line.strip().split()[0])
                            for line in f
                            if line.strip()
                        ]
                else:
                    yolo_cache[label_txt] = []
            classes_in_img = yolo_cache[label_txt]
            raw_class = classes_in_img[sig_idx] if sig_idx < len(classes_in_img) else sig_idx
            binary_label = 1 if raw_class == 0 else 0

        elif active_label_mode == "bandwidth":
            # complex128 .npy dosyasında 128 byte header + 16 byte * N eleman bulunur
            n_samples = (os.path.getsize(iq_path) - 128) // 16
            binary_label = 1 if n_samples >= 90 else 0

        else:  # "readme"
            binary_label = 1 if sig_idx == 0 else 0

        samples.append((iq_path, crop_path, binary_label))

    pos_count = sum(s[2] for s in samples)
    neg_count = len(samples) - pos_count
    print(
        f"[BİLGİ] Toplam Eşleşen Sinyal Çifti: {len(samples):,} "
        f"(Toplam İşlenecek Dosya: {len(samples)*2:,} adet -> {len(samples):,} IQ + {len(samples):,} Spektrogram)"
    )
    print(f"[BİLGİ] Sınıf Dağılımı -> Drone YOK (0): {neg_count:,} | Drone VAR (1): {pos_count:,}")
    return samples


class RFDroneDataset(Dataset):
    """
    Bellek dostu (sıkıştırılmış uint8 spektrogram + float32 IQ tensörleri, toplam ~415 MB)
    ve 16-Thread paralel ön-yükleme destekli PyTorch Dataset sınıfı.
    58.038 dosyanın tamamını ~15 saniyede paralel okuyarak 50 Epoch boyunca GPU'nun
    hiç disk darboğazına takılmadan tam hızda çalışmasını sağlar.
    """

    def __init__(
        self,
        samples: List[Tuple[str, str, int]],
        cfg: Config,
        is_train: bool = False,
        split_name: str = "Train",
    ):
        self.samples = samples
        self.data_mode = cfg.DATA_MODE
        self.iq_fixed_len = cfg.IQ_FIXED_LEN
        self.iq_global_scale = cfg.IQ_GLOBAL_SCALE
        self.img_size = cfg.IMG_SIZE
        self.is_train = is_train
        self.split_name = split_name

        self.iq_tensors: List[Optional[torch.Tensor]] = [None] * len(samples)
        self.spec_uint8: List[Optional[torch.Tensor]] = [None] * len(samples)
        self._preload_parallel(cfg.IO_THREADS)

    def _process_single_item(self, idx_and_sample: Tuple[int, Tuple[str, str, int]]):
        idx, (iq_path, crop_path, _) = idx_and_sample

        if self.data_mode in ("iq", "hybrid"):
            iq_complex = np.load(iq_path)
            i_chan = np.real(iq_complex).astype(np.float32) / self.iq_global_scale
            q_chan = np.imag(iq_complex).astype(np.float32) / self.iq_global_scale
            iq_2d = np.stack([i_chan, q_chan], axis=0)
            t_iq = torch.from_numpy(iq_2d).unsqueeze(0)
            t_iq = F.interpolate(t_iq, size=self.iq_fixed_len, mode="linear", align_corners=False).squeeze(0)
            self.iq_tensors[idx] = t_iq

        if self.data_mode in ("spectrogram", "hybrid"):
            with Image.open(crop_path) as img:
                img_rgb = img.convert("RGB").resize(self.img_size, Image.BILINEAR)
                arr_u8 = np.asarray(img_rgb, dtype=np.uint8)  # Bellek tasarrufu için uint8 (1 byte/pixel)
            self.spec_uint8[idx] = torch.from_numpy(arr_u8).permute(2, 0, 1).contiguous()

        return idx

    def _preload_parallel(self, max_workers: int):
        total = len(self.samples)
        files_per_sample = 2 if self.data_mode == "hybrid" else 1
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for done_count, _ in enumerate(
                executor.map(self._process_single_item, enumerate(self.samples)), start=1
            ):
                if done_count % 500 == 0 or done_count == total:
                    pct = (done_count / total) * 100.0
                    elapsed = time.time() - t0
                    sys.stdout.write(
                        f"\r  [VERİ YÜKLEME - {self.split_name:5s}] "
                        f"{done_count * files_per_sample:,}/{total * files_per_sample:,} dosya "
                        f"(%{pct:5.1f}) | Süre: {elapsed:4.1f}s"
                    )
                    sys.stdout.flush()
        print()

    def __len__(self) -> int:
        return len(self.samples)

    def _get_iq(self, idx: int) -> torch.Tensor:
        tensor = self.iq_tensors[idx]
        if self.is_train and random.random() < 0.3:
            return tensor + 0.01 * torch.randn_like(tensor)
        return tensor

    def _get_spectrogram(self, idx: int) -> torch.Tensor:
        # uint8 [0..255] -> float32 [-1.0..1.0] dönüşümü batch anında yapılır (RAM şişmesini önler)
        t_float = self.spec_uint8[idx].to(dtype=torch.float32) / 255.0
        return (t_float - 0.5) / 0.5

    def __getitem__(self, idx: int):
        _, _, label = self.samples[idx]
        target = torch.tensor(label, dtype=torch.long)

        if self.data_mode == "iq":
            return self._get_iq(idx), target
        elif self.data_mode == "spectrogram":
            return self._get_spectrogram(idx), target
        elif self.data_mode == "hybrid":
            return (self._get_iq(idx), self._get_spectrogram(idx)), target
        else:
            raise ValueError(f"Geçersiz DATA_MODE: {self.data_mode}")


def create_dataloaders(cfg: Config):
    """
    Veriyi Stratified (sınıf oranlarını koruyarak) %70 Train, %15 Validation, %15 Test
    olarak ayırır ve tüm dosyaları hızlı paralel okuyucuyla PyTorch DataLoader'lara hazırlar.
    """
    samples = build_dataset_index(cfg)
    labels = [s[2] for s in samples]

    train_samples, temp_samples, _, temp_labels = train_test_split(
        samples, labels, test_size=0.30, random_state=cfg.SEED, stratify=labels
    )
    val_samples, test_samples = train_test_split(
        temp_samples, test_size=0.50, random_state=cfg.SEED, stratify=temp_labels
    )

    print(
        f"[BİLGİ] Veri Ayrımı -> Train: {len(train_samples):,} çift ({len(train_samples)*2:,} dosya) | "
        f"Val: {len(val_samples):,} çift ({len(val_samples)*2:,} dosya) | "
        f"Test: {len(test_samples):,} çift ({len(test_samples)*2:,} dosya)"
    )

    train_ds = RFDroneDataset(train_samples, cfg, is_train=True, split_name="Train")
    val_ds = RFDroneDataset(val_samples, cfg, is_train=False, split_name="Val")
    test_ds = RFDroneDataset(test_samples, cfg, is_train=False, split_name="Test")

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=pin_mem
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=pin_mem
    )

    train_labels = [s[2] for s in train_samples]
    count_0 = max(train_labels.count(0), 1)
    count_1 = max(train_labels.count(1), 1)
    total = count_0 + count_1
    class_weights = torch.tensor([total / (2.0 * count_0), total / (2.0 * count_1)], dtype=torch.float32)

    return train_loader, val_loader, test_loader, class_weights


# ==============================================================================
# 2. MODEL MİMARİLERİ (1D CNN / 2D CNN / HYBRID 1D+2D FUSION)
# ==============================================================================
class ResidualBlock1D(nn.Module):
    """Ham IQ sinyalleri için 1-Boyutlu Artık (Residual) Evrişim Bloğu."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.gelu(out)


class IQ1DResNet(nn.Module):
    """
    2 Kanallı (In-phase & Quadrature) Ham RF IQ Sinyalleri için Hafif 1D-ResNet Mimarisi.
    Girdi Boyutu: (Batch, 2, 256) -> Çıktı Boyutu: (Batch, 2)
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.layer1 = ResidualBlock1D(32, 64, stride=2)
        self.layer2 = ResidualBlock1D(64, 128, stride=2)
        self.layer3 = ResidualBlock1D(128, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.pool(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(x)
        return self.classifier(feats)


class Spectrogram2DCNN(nn.Module):
    """
    Spektrogram Görselleri (.png) için 2D CNN Mimarisi.
    Girdi Boyutu: (Batch, 3, 64, 64) -> Çıktı Boyutu: (Batch, 2)
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(x)
        return self.classifier(feats)


class HybridRFClassifier(nn.Module):
    """
    Hem Ham IQ Sinyalini (1D ResNet) hem de Spektrogram Görselini (2D CNN) birleştiren
    Çift Dallı (Multimodal) Derin Öğrenme Modeli (58.038 dosyanın tamamını birlikte işler).
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.iq_branch = IQ1DResNet(num_classes=num_classes)
        self.spec_branch = Spectrogram2DCNN(num_classes=num_classes)
        self.fusion_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256 + 256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        iq_x, spec_x = inputs
        iq_feat = self.iq_branch.extract_features(iq_x)
        spec_feat = self.spec_branch.extract_features(spec_x)
        combined = torch.cat([iq_feat, spec_feat], dim=1)
        return self.fusion_head(combined)


def build_model(cfg: Config) -> nn.Module:
    if cfg.DATA_MODE == "iq":
        return IQ1DResNet(num_classes=2)
    elif cfg.DATA_MODE == "spectrogram":
        return Spectrogram2DCNN(num_classes=2)
    elif cfg.DATA_MODE == "hybrid":
        return HybridRFClassifier(num_classes=2)
    else:
        raise ValueError(f"Bilinmeyen DATA_MODE: {cfg.DATA_MODE}")


# ==============================================================================
# 3. EĞİTİM DÖNGÜSÜ (CUDA, ADAM, SCHEDULER, MODEL CHECKPOINT / EARLY STOPPING)
# ==============================================================================
class EarlyStopping:
    """
    Her epoch sonunda Validation Loss değerini kontrol eder ve en iyi model ağırlıklarını kaydeder.
    USE_EARLY_STOPPING=False olduğunda eğitimi asla yarıda kesmez; tüm epoch'lar bittiğinde
    en yüksek doğrulama başarısına ulaşmış ağırlıkları yükler.
    """

    def __init__(
        self,
        patience: int = 50,
        min_delta: float = 1e-5,
        save_path: str = "best_model.pt",
        enabled: bool = False,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.enabled = enabled
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.counter = 0
        self.best_weights = None

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> Tuple[bool, bool]:
        improved = False
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            self.best_weights = copy.deepcopy(model.state_dict())
            torch.save(self.best_weights, self.save_path)
            improved = True
        else:
            self.counter += 1

        should_stop = self.enabled and (self.counter >= self.patience)
        return should_stop, improved


def move_to_device(inputs, device: torch.device):
    if isinstance(inputs, (tuple, list)):
        return tuple(x.to(device, non_blocking=True) for x in inputs)
    return inputs.to(device, non_blocking=True)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch_num: int = 1,
    total_epochs: int = 50,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    stage = "Train" if is_train else "Val  "

    running_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(loader)

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_idx, (inputs, targets) in enumerate(loader, start=1):
            inputs = move_to_device(inputs, device)
            targets = targets.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad()

            logits = model(inputs)
            loss = criterion(logits, targets)

            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += batch_size

            if batch_idx % 20 == 0 or batch_idx == num_batches:
                pct = (batch_idx / num_batches) * 100.0
                sys.stdout.write(
                    f"\r  -> [{stage}] Epoch [{epoch_num:02d}/{total_epochs:02d}] "
                    f"Batch [{batch_idx:03d}/{num_batches:03d}] (%{pct:5.1f}) | "
                    f"Anlık Loss: {running_loss/total:.4f} | Acc: {(correct/total)*100:5.2f}%"
                )
                sys.stdout.flush()

    sys.stdout.write("\r" + " " * 105 + "\r")
    sys.stdout.flush()
    return running_loss / total, correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    cfg: Config,
    device: torch.device,
) -> nn.Module:
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS, eta_min=1e-6
    )
    checkpoint_tracker = EarlyStopping(
        patience=cfg.EARLY_STOPPING_PATIENCE,
        save_path=cfg.MODEL_SAVE_PATH,
        enabled=cfg.USE_EARLY_STOPPING,
    )

    print("\n" + "=" * 92)
    print(
        f"EĞİTİM BAŞLIYOR | Cihaz: {device} | Toplam Epoch: {cfg.EPOCHS} (Kesintisiz Tam Eğitim)"
    )
    print("=" * 92)

    for epoch in range(1, cfg.EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, epoch, cfg.EPOCHS
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device, None, epoch, cfg.EPOCHS
        )
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        should_stop, improved = checkpoint_tracker.step(val_loss, model, epoch)
        star = " ★ [EN İYİ]" if improved else ""

        print(
            f"Epoch [{epoch:02d}/{cfg.EPOCHS:02d}] ({elapsed:4.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:6.2f}% || "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:6.2f}% | LR: {current_lr:.2e}{star}"
        )

        if should_stop:
            print(f"\n[EARLY STOPPING] Eğitim {epoch}. epoch'ta durduruldu.")
            break

    if checkpoint_tracker.best_weights is not None:
        model.load_state_dict(checkpoint_tracker.best_weights)
        print(
            f"\n[BİLGİ] {cfg.EPOCHS} Epoch tamamlandı! En iyi model ağırlıkları "
            f"(Epoch {checkpoint_tracker.best_epoch}, Val Loss: {checkpoint_tracker.best_loss:.4f}) yüklendi."
        )
        print(f"[BİLGİ] Kaydedilen model dosyası: {cfg.MODEL_SAVE_PATH}")

    return model


# ==============================================================================
# 4. DEĞERLENDİRME, METRİKLER VE CONFUSION MATRIX GÖRSELLEŞTİRME
# ==============================================================================
def evaluate_model(model: nn.Module, test_loader: DataLoader, cfg: Config, device: torch.device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = move_to_device(inputs, device)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 75)
    print("NİHAİ TEST SETİ DEĞERLENDİRME SONUÇLARI (58.038 DOSYA - HİBRİT MODEL)")
    print("=" * 75)
    print(f"Doğruluk  (Accuracy)  : {acc  * 100:.2f}% ({acc:.4f})")
    print(f"Kesinlik  (Precision) : {prec * 100:.2f}% ({prec:.4f})")
    print(f"Duyarlılık (Recall)   : {rec  * 100:.2f}% ({rec:.4f})")
    print(f"F1-Skoru  (F1-Score)  : {f1   * 100:.2f}% ({f1:.4f})")
    print("-" * 75)
    print("Detaylı Sınıflandırma Raporu:\n")
    print(
        classification_report(
            y_true, y_pred, target_names=["Drone Yok (0)", "Drone Var (1)"], digits=4
        )
    )
    print("Confusion Matrix:\n", cm)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Drone Yok (0)", "Drone Var (1)"]
    )
    disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=False)
    ax.set_title(f"RF Drone Tespiti - Confusion Matrix ({cfg.DATA_MODE.upper()})")
    plt.tight_layout()
    plt.savefig(cfg.CM_PLOT_PATH)
    print(f"[BİLGİ] Confusion Matrix görseli kaydedildi: {cfg.CM_PLOT_PATH}")
    plt.show()


# ==============================================================================
# ANA ÇALIŞTIRMA BLOĞU
# ==============================================================================
def main():
    cfg = Config()
    set_seed(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SİSTEM] Kullanılan Cihaz: {device}")
    if device.type == "cuda":
        print(f"[SİSTEM] GPU Modeli: {torch.cuda.get_device_name(0)}")

    # 1. Veri Yükleyiciler (58.038 dosyanın tamamını 16-thread ile hızlı yükler)
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(cfg)

    # 2. Model Kurulumu (HybridRFClassifier: 1D-ResNet + 2D-CNN)
    model = build_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Mimari: {model.__class__.__name__} | Eğitilebilir Parametre Sayısı: {total_params:,}")

    # 3. Model Eğitimi (50 Epoch kesintisiz)
    model = train_model(model, train_loader, val_loader, class_weights, cfg, device)

    # 4. Test Seti Değerlendirmesi ve Görselleştirme
    evaluate_model(model, test_loader, cfg, device)


if __name__ == "__main__":
    main()
