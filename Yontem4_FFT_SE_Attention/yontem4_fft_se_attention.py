"""
YÖNTEM 4: Frekans Uzayı (FFT / PSD) + Squeeze-and-Excitation (SE) Kanal Dikkat Mekanizmalı Hibrit Ağ
Veri Seti: Synthetic RF Spectrograms and IQ Signals for UAV Detection (2.4 GHz band)
Dizin: C:\\Users\\enyas\\Desktop\\Dataset_RF_IQ (58.038 Dosya: 29.019 IQ + 29.019 Spektrogram)

Yenilikler:
  1) Zaman + Frekans Uzayı 4-Kanallı 1D Giriş (Batch x 4 x 256):
       - Kanal 0: I(t)                                (Zaman Uzayı In-phase)
       - Kanal 1: Q(t)                                (Zaman Uzayı Quadrature)
       - Kanal 2: |z(t)|                              (Zaman Uzayı Anlık Genlik Zarfı)
       - Kanal 3: log10(|FFT(z(t))|^2 + eps)          (Frekans Uzayı Hızlı Fourier Dönüşümü - Güç Spektral Yoğunluğu / PSD)
  2) Squeeze-and-Excitation (SE) Kanal Dikkat Blokları (SEBlock1D & SEBlock2D):
       - Hem 1D-ResNet hem de 2D-CNN katmanlarında her kanalın önem ağırlığını dinamik olarak öğrenir.
  3) Kapılı Dikkat Füzyonu (Gated Attention Fusion):
       - 1D Sinyal öznitelikleri ile 2D Spektrogram özniteliklerini dikkat kapısı (Sigmoid Gate) ile birleştirir.
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
    DATASET_DIR = r"C:\Users\enyas\Desktop\Dataset_RF_IQ"
    IQ_DIR = os.path.join(DATASET_DIR, "iq_signals")
    CROP_DIR = os.path.join(DATASET_DIR, "signal_crops")

    METHOD_DIR = r"C:\Users\enyas\PycharmProjects\PythonProject1\Yontem4_FFT_SE_Attention"

    NUM_CLASSES = 2
    CLASS_NAMES = ["Drone Yok (0)", "Drone Var (1)"]

    IQ_CHANNELS = 4           # [I(t), Q(t), |z(t)|, FFT_PSD(f)]
    IQ_FIXED_LEN = 256
    IQ_GLOBAL_SCALE = 0.05
    IMG_SIZE = (64, 64)

    SEED = 42
    BATCH_SIZE = 128
    IO_THREADS = 16
    EPOCHS = 40               # 40 Epoch kesintisiz tam eğitim
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    MODEL_SAVE_PATH = os.path.join(DATASET_DIR, "best_drone_rf_model_yontem4_se_fft.pt")
    CM_PLOT_PATH = os.path.join(METHOD_DIR, "cm_yontem4_se_fft.png")
    CURVES_PLOT_PATH = os.path.join(METHOD_DIR, "curves_yontem4_se_fft.png")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# 1. VERİ İNDEKSLEME VE ZAMAN + FREKANS (FFT PSD) PARALEL YÜKLEME
# ==============================================================================
FILENAME_PATTERN = re.compile(r"SpectrogramBTFinal_SVG_(\d+)_signal_(\d+)(?:_iq)?\.(?:npy|png)$")


def build_dataset_index(cfg: Config) -> List[Tuple[str, str, int]]:
    os.makedirs(cfg.METHOD_DIR, exist_ok=True)
    iq_files = sorted(glob.glob(os.path.join(cfg.IQ_DIR, "*.npy")))
    if not iq_files:
        raise FileNotFoundError(f"IQ dosyaları bulunamadı: {cfg.IQ_DIR}")

    print("[BİLGİ] YÖNTEM 4: Zaman+Frekans (FFT PSD) + Squeeze-and-Excitation (SE) Attention Hibrit Modeli")

    samples: List[Tuple[str, str, int]] = []
    for iq_path in iq_files:
        fname = os.path.basename(iq_path)
        match = FILENAME_PATTERN.search(fname)
        if not match:
            continue
        spec_id, sig_idx = int(match.group(1)), int(match.group(2))
        crop_path = os.path.join(cfg.CROP_DIR, f"SpectrogramBTFinal_SVG_{spec_id}_signal_{sig_idx}.png")
        n_samples = (os.path.getsize(iq_path) - 128) // 16
        binary_label = 1 if n_samples >= 90 else 0
        samples.append((iq_path, crop_path, binary_label))

    pos_count = sum(s[2] for s in samples)
    neg_count = len(samples) - pos_count
    print(f"[BİLGİ] Toplam Eşleşen Sinyal Çifti: {len(samples):,} ({len(samples)*2:,} dosya)")
    print(f"[BİLGİ] Sınıf Dağılımı -> Drone YOK (0): {neg_count:,} | Drone VAR (1): {pos_count:,}")
    return samples


class RFFFTAttentionDataset(Dataset):
    """
    Her ham complex128 IQ sinyali için Zaman + Frekans (FFT) 4 kanallı tensör üretir:
      0: I(t) / scale
      1: Q(t) / scale
      2: |z(t)| / scale
      3: FFT Log-PSD = log10(|fftshift(FFT(z))|^2 + 1e-6) normalize kanalı
    """

    def __init__(
        self,
        samples: List[Tuple[str, str, int]],
        cfg: Config,
        is_train: bool = False,
        split_name: str = "Train",
    ):
        self.samples = samples
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

        iq_complex = np.load(iq_path)
        i_chan = np.real(iq_complex).astype(np.float32) / self.iq_global_scale
        q_chan = np.imag(iq_complex).astype(np.float32) / self.iq_global_scale
        amp_chan = np.abs(iq_complex).astype(np.float32) / self.iq_global_scale

        # Frekans Uzayı: Merkezlenmiş FFT ve Logaritmik Güç Spektral Yoğunluğu (Log-PSD)
        fft_vals = np.fft.fftshift(np.fft.fft(iq_complex))
        psd = np.log10((np.abs(fft_vals) ** 2) + 1e-6).astype(np.float32)
        psd_norm = (psd + 3.0) / 3.0  # Yaklaşık [-1, 1] aralığına ölçekleme

        iq_4d = np.stack([i_chan, q_chan, amp_chan, psd_norm], axis=0)  # shape: (4, L)
        t_iq = torch.from_numpy(iq_4d).unsqueeze(0)
        t_iq = F.interpolate(t_iq, size=self.iq_fixed_len, mode="linear", align_corners=False).squeeze(0)
        self.iq_tensors[idx] = t_iq  # shape: (4, 256)

        with Image.open(crop_path) as img:
            img_rgb = img.convert("RGB").resize(self.img_size, Image.BILINEAR)
            arr_u8 = np.array(img_rgb, dtype=np.uint8, copy=True)
        self.spec_uint8[idx] = torch.from_numpy(arr_u8).permute(2, 0, 1).contiguous()

        return idx

    def _preload_parallel(self, max_workers: int):
        total = len(self.samples)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for done_count, _ in enumerate(
                executor.map(self._process_single_item, enumerate(self.samples)), start=1
            ):
                if done_count % 500 == 0 or done_count == total:
                    pct = (done_count / total) * 100.0
                    elapsed = time.time() - t0
                    sys.stdout.write(
                        f"\r  [VERİ YÜKLEME (FFT+IQ) - {self.split_name:5s}] "
                        f"{done_count * 2:,}/{total * 2:,} dosya (%{pct:5.1f}) | Süre: {elapsed:4.1f}s"
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
        t_float = self.spec_uint8[idx].to(dtype=torch.float32) / 255.0
        return (t_float - 0.5) / 0.5

    def __getitem__(self, idx: int):
        _, _, label = self.samples[idx]
        return (self._get_iq(idx), self._get_spectrogram(idx)), torch.tensor(label, dtype=torch.long)


def create_dataloaders(cfg: Config):
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

    train_ds = RFFFTAttentionDataset(train_samples, cfg, is_train=True, split_name="Train")
    val_ds = RFFFTAttentionDataset(val_samples, cfg, is_train=False, split_name="Val")
    test_ds = RFFFTAttentionDataset(test_samples, cfg, is_train=False, split_name="Test")

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, pin_memory=pin_mem)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, pin_memory=pin_mem)

    train_labels = [s[2] for s in train_samples]
    count_0 = max(train_labels.count(0), 1)
    count_1 = max(train_labels.count(1), 1)
    total = count_0 + count_1
    class_weights = torch.tensor([total / (2.0 * count_0), total / (2.0 * count_1)], dtype=torch.float32)

    return train_loader, val_loader, test_loader, class_weights


# ==============================================================================
# 2. MİMARİ: SQUEEZE-AND-EXCITATION (SE) ATTENTION 1D-RESNET + 2D-CNN
# ==============================================================================
class SEBlock1D(nn.Module):
    """1-Boyutlu Sinyaller İçin Squeeze-and-Excitation (Kanal Dikkat) Bloğu."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.GELU(),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).unsqueeze(-1)
        return x * w


class SEBlock2D(nn.Module):
    """2-Boyutlu Spektrogramlar İçin Squeeze-and-Excitation (Kanal Dikkat) Bloğu."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.GELU(),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class SEResidualBlock1D(nn.Module):
    """SE Dikkat Mekanizması Entegre Edilmiş 1D Residual Blok."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock1D(out_channels, reduction=8)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)  # Kanal Dikkat Ağırlıklandırması
        out += self.shortcut(x)
        return F.gelu(out)


class SEIQ1DResNet(nn.Module):
    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.layer1 = SEResidualBlock1D(32, 64, stride=2)
        self.layer2 = SEResidualBlock1D(64, 128, stride=2)
        self.layer3 = SEResidualBlock1D(128, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.layer3(self.layer2(self.layer1(self.stem(x))))).flatten(1)


class SESpectrogram2DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            SEBlock2D(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            SEBlock2D(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            SEBlock2D(128),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            SEBlock2D(256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)


class SEHybridRFClassifier(nn.Module):
    """
    Zaman+Frekans (FFT) 1D-SE-ResNet ve 2D-SE-CNN dallarını
    Kapılı Dikkat Füzyonu (Gated Attention Fusion) ile birleştiren mimari.
    """

    def __init__(self, iq_channels: int = 4, num_classes: int = 2):
        super().__init__()
        self.iq_branch = SEIQ1DResNet(in_channels=iq_channels)
        self.spec_branch = SESpectrogram2DCNN()

        # Kapılı Dikkat Füzyonu (Gated Multimodal Attention)
        self.attention_gate = nn.Sequential(
            nn.Linear(512, 128, bias=False),
            nn.GELU(),
            nn.Linear(128, 512, bias=False),
            nn.Sigmoid(),
        )
        self.fusion_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        iq_x, spec_x = inputs
        iq_feat = self.iq_branch.extract_features(iq_x)
        spec_feat = self.spec_branch.extract_features(spec_x)
        combined = torch.cat([iq_feat, spec_feat], dim=1)
        gated = combined * self.attention_gate(combined)
        return self.fusion_head(gated)


# ==============================================================================
# 3. EĞİTİM DÖNGÜSÜ VE ÖĞRENME EĞRİSİ KAYDI
# ==============================================================================
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
    total_epochs: int = 40,
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
                    f"Loss: {running_loss/total:.4f} | Acc: {(correct/total)*100:5.2f}%"
                )
                sys.stdout.flush()

    sys.stdout.write("\r" + " " * 105 + "\r")
    sys.stdout.flush()
    return running_loss / total, correct / total


def plot_learning_curves(history: dict, save_path: str):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=130)

    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2)
    ax1.set_title("Yöntem 4 (FFT + SE-Attention) - Kayıp (Loss) Eğrisi")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CrossEntropy Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Acc (%)", color="#2ca02c", linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], label="Val Acc (%)", color="#d62728", linewidth=2)
    ax2.set_title("Yöntem 4 (FFT + SE-Attention) - Doğruluk (Accuracy) Eğrisi")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"[BİLGİ] Öğrenme eğrileri kaydedildi: {save_path}")


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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)

    best_loss = float("inf")
    best_epoch = 0
    best_weights = None
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print("\n" + "=" * 92)
    print(f"YÖNTEM 4 (FFT FREKANS UZAYI + SE-ATTENTION HİBRİT MODEL) EĞİTİMİ BAŞLIYOR | Epoch: {cfg.EPOCHS}")
    print("=" * 92)

    for epoch in range(1, cfg.EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer, epoch, cfg.EPOCHS)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, None, epoch, cfg.EPOCHS)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        star = ""
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, cfg.MODEL_SAVE_PATH)
            star = " ★ [EN İYİ]"

        print(
            f"Epoch [{epoch:02d}/{cfg.EPOCHS:02d}] ({elapsed:4.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:6.2f}% || "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:6.2f}% | LR: {current_lr:.2e}{star}"
        )

    plot_learning_curves(history, cfg.CURVES_PLOT_PATH)

    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(
            f"\n[BİLGİ] {cfg.EPOCHS} Epoch tamamlandı! En iyi model ağırlıkları "
            f"(Epoch {best_epoch}, Val Loss: {best_loss:.4f}) yüklendi."
        )

    return model


# ==============================================================================
# 4. TEST SETİ DEĞERLENDİRMESİ VE CONFUSION MATRIX
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

    print("\n" + "=" * 80)
    print("YÖNTEM 4 TEST SETİ SONUÇLARI (FFT FREKANS UZAYI + SE-ATTENTION HİBRİT MODEL)")
    print("=" * 80)
    print(f"Doğruluk  (Accuracy)  : {acc  * 100:.2f}% ({acc:.4f})")
    print(f"Kesinlik  (Precision) : {prec * 100:.2f}% ({prec:.4f})")
    print(f"Duyarlılık (Recall)   : {rec  * 100:.2f}% ({rec:.4f})")
    print(f"F1-Skoru  (F1-Score)  : {f1   * 100:.2f}% ({f1:.4f})")
    print("-" * 80)
    print("Detaylı Sınıflandırma Raporu:\n")
    print(classification_report(y_true, y_pred, target_names=cfg.CLASS_NAMES, digits=4))
    print("Confusion Matrix:\n", cm)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=130)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cfg.CLASS_NAMES)
    disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=False)
    ax.set_title("Yöntem 4: FFT (Log-PSD) + SE-Attention Hibrit RF Drone Tespiti")
    plt.tight_layout()
    plt.savefig(cfg.CM_PLOT_PATH)
    print(f"[BİLGİ] Confusion Matrix kaydedildi: {cfg.CM_PLOT_PATH}")
    plt.show()


def main():
    cfg = Config()
    set_seed(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SİSTEM] Kullanılan Cihaz: {device}")
    if device.type == "cuda":
        print(f"[SİSTEM] GPU Modeli: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, test_loader, class_weights = create_dataloaders(cfg)

    model = SEHybridRFClassifier(iq_channels=cfg.IQ_CHANNELS, num_classes=cfg.NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Mimari: {model.__class__.__name__} | Eğitilebilir Parametre Sayısı: {total_params:,}")

    model = train_model(model, train_loader, val_loader, class_weights, cfg, device)
    evaluate_model(model, test_loader, cfg, device)


if __name__ == "__main__":
    main()
