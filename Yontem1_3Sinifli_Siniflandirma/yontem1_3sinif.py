"""
YÖNTEM 1 (v2): 3 Sınıflı Çoklu RF Sinyal Sınıflandırması (Drone vs WiFi vs Bluetooth)
Veri Seti: Synthetic RF Spectrograms and IQ Signals for UAV Detection (2.4 GHz band)
Dizin: C:\\Users\\enyas\\Desktop\\Dataset_RF_IQ (58.038 Dosya: 29.019 IQ + 29.019 Spektrogram)

Sınıflar (README.txt & STFT Fiziksel Bant Genişliği Kriteri - 56 MS/s, NFFT=512):
  - Class 0: Drone_Signal (DJI Phantom 4 Pro - Geniş Bant Aktif Video İletimi, L >= 90 bin)
  - Class 1: WiFi         (2.4 GHz WLAN - Orta Bant Paket İletimi, 35 <= L < 90 bin)
  - Class 2: Bluetooth    (FHSS - Dar Bant Frekans Atlamalı Sinyal, L < 35 bin)
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
    LABELS_DIR = os.path.join(DATASET_DIR, "labels")

    PROJECT_DIR = r"C:\Users\enyas\PycharmProjects\PythonProject1"
    ASSETS_DIR = os.path.join(PROJECT_DIR, "assets")

    DATA_MODE = "hybrid"      # 58.038 dosyanın tamamı (1D-ResNet + 2D-CNN)
    NUM_CLASSES = 3           # 0: Drone, 1: WiFi, 2: Bluetooth
    CLASS_NAMES = ["Drone (0)", "WiFi (1)", "Bluetooth (2)"]

    IQ_FIXED_LEN = 256
    IQ_GLOBAL_SCALE = 0.05
    IMG_SIZE = (64, 64)

    SEED = 42
    BATCH_SIZE = 128
    NUM_WORKERS = 0
    IO_THREADS = 16
    EPOCHS = 40               # 40 Epoch kesintisiz tam eğitim
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    USE_EARLY_STOPPING = False

    MODEL_SAVE_PATH = os.path.join(DATASET_DIR, "best_drone_rf_model_v2_3class.pt")
    CM_PLOT_PATH = os.path.join(ASSETS_DIR, "cm_v2_method1_3class.png")
    CURVES_PLOT_PATH = os.path.join(ASSETS_DIR, "curves_v2_method1_3class.png")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# 1. VERİ İNDEKSLEME (3 SINIF: DRONE=0, WIFI=1, BLUETOOTH=2) VE PARALEL YÜKLEME
# ==============================================================================
FILENAME_PATTERN = re.compile(r"SpectrogramBTFinal_SVG_(\d+)_signal_(\d+)(?:_iq)?\.(?:npy|png)$")


def build_dataset_index(cfg: Config) -> List[Tuple[str, str, int]]:
    os.makedirs(cfg.ASSETS_DIR, exist_ok=True)
    iq_files = sorted(glob.glob(os.path.join(cfg.IQ_DIR, "*.npy")))
    if not iq_files:
        raise FileNotFoundError(f"IQ dosyaları bulunamadı: {cfg.IQ_DIR}")

    has_yolo_labels = os.path.isdir(cfg.LABELS_DIR) and len(os.listdir(cfg.LABELS_DIR)) > 0
    print(
        f"[BİLGİ] YÖNTEM 1: 3 Sınıflı Sınıflandırma (Drone vs WiFi vs Bluetooth) | "
        f"Veri Modu: '{cfg.DATA_MODE}'"
    )

    samples: List[Tuple[str, str, int]] = []
    yolo_cache = {}
    class_counts = {0: 0, 1: 0, 2: 0}

    for iq_path in iq_files:
        fname = os.path.basename(iq_path)
        match = FILENAME_PATTERN.search(fname)
        if not match:
            continue

        spec_id, sig_idx = int(match.group(1)), int(match.group(2))
        crop_fname = f"SpectrogramBTFinal_SVG_{spec_id}_signal_{sig_idx}.png"
        crop_path = os.path.join(cfg.CROP_DIR, crop_fname)

        if has_yolo_labels:
            label_txt = os.path.join(cfg.LABELS_DIR, f"SpectrogramBTFinal_SVG_{spec_id}.txt")
            if label_txt not in yolo_cache:
                if os.path.exists(label_txt):
                    with open(label_txt, "r", encoding="utf-8") as f:
                        yolo_cache[label_txt] = [int(line.strip().split()[0]) for line in f if line.strip()]
                else:
                    yolo_cache[label_txt] = []
            classes_in_img = yolo_cache[label_txt]
            label = classes_in_img[sig_idx] if sig_idx < len(classes_in_img) else 2
        else:
            # STFT Fiziksel Bant Genişliği (56 MS/s, NFFT=512 -> ~109.4 kHz/bin):
            # L >= 90 -> Class 0: Drone_Signal (DJI Phantom 4 Pro aktif video iletimi)
            # 35 <= L < 90 -> Class 1: WiFi (2.4 GHz)
            # L < 35 -> Class 2: Bluetooth (FHSS dar bant sıçrama)
            n_samples = (os.path.getsize(iq_path) - 128) // 16
            if n_samples >= 90:
                label = 0  # Drone_Signal
            elif n_samples >= 35:
                label = 1  # WiFi
            else:
                label = 2  # Bluetooth

        class_counts[label] += 1
        samples.append((iq_path, crop_path, label))

    print(
        f"[BİLGİ] Toplam Eşleşen Sinyal Çifti: {len(samples):,} "
        f"(Toplam İşlenecek Dosya: {len(samples)*2:,} adet)"
    )
    print(
        f"[BİLGİ] 3 Sınıf Dağılımı -> "
        f"Drone (0): {class_counts[0]:,} | WiFi (1): {class_counts[1]:,} | Bluetooth (2): {class_counts[2]:,}"
    )
    return samples


class RFDroneDataset(Dataset):
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
                arr_u8 = np.array(img_rgb, dtype=np.uint8, copy=True)
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
        t_float = self.spec_uint8[idx].to(dtype=torch.float32) / 255.0
        return (t_float - 0.5) / 0.5

    def __getitem__(self, idx: int):
        _, _, label = self.samples[idx]
        target = torch.tensor(label, dtype=torch.long)
        return (self._get_iq(idx), self._get_spectrogram(idx)), target


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

    train_ds = RFDroneDataset(train_samples, cfg, is_train=True, split_name="Train")
    val_ds = RFDroneDataset(val_samples, cfg, is_train=False, split_name="Val")
    test_ds = RFDroneDataset(test_samples, cfg, is_train=False, split_name="Test")

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, pin_memory=pin_mem)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, pin_memory=pin_mem)

    train_labels = [s[2] for s in train_samples]
    total = len(train_labels)
    weights = [total / (cfg.NUM_CLASSES * max(train_labels.count(c), 1)) for c in range(cfg.NUM_CLASSES)]
    class_weights = torch.tensor(weights, dtype=torch.float32)

    return train_loader, val_loader, test_loader, class_weights


# ==============================================================================
# 2. MODEL MİMARİSİ (3 SINIFLI HİBRİT 1D-RESNET + 2D-CNN)
# ==============================================================================
class ResidualBlock1D(nn.Module):
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
    def __init__(self, num_classes: int = 3):
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

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.pool(x).flatten(1)


class Spectrogram2DCNN(nn.Module):
    def __init__(self, num_classes: int = 3):
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

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)


class HybridRFClassifier3Class(nn.Module):
    def __init__(self, num_classes: int = 3):
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
    ax1.set_title("Yöntem 1 (3 Sınıf) - Kayıp (Loss) Eğrisi")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CrossEntropy Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Acc (%)", color="#2ca02c", linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], label="Val Acc (%)", color="#d62728", linewidth=2)
    ax2.set_title("Yöntem 1 (3 Sınıf) - Doğruluk (Accuracy) Eğrisi")
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
    print(f"YÖNTEM 1 (3 SINIF: DRONE vs WIFI vs BLUETOOTH) EĞİTİMİ BAŞLIYOR | Epoch: {cfg.EPOCHS}")
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
# 4. TEST SETİ DEĞERLENDİRMESİ VE 3x3 CONFUSION MATRIX
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
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 78)
    print("YÖNTEM 1 TEST SETİ SONUÇLARI (3 SINIF: DRONE vs WIFI vs BLUETOOTH)")
    print("=" * 78)
    print(f"Genel Doğruluk (Accuracy)        : {acc        * 100:.2f}% ({acc:.4f})")
    print(f"Makro Kesinlik (Macro Precision) : {prec_macro * 100:.2f}% ({prec_macro:.4f})")
    print(f"Makro Duyarlılık (Macro Recall)  : {rec_macro  * 100:.2f}% ({rec_macro:.4f})")
    print(f"Makro F1-Skoru (Macro F1-Score)  : {f1_macro   * 100:.2f}% ({f1_macro:.4f})")
    print("-" * 78)
    print("Detaylı 3-Sınıflı Sınıflandırma Raporu:\n")
    print(classification_report(y_true, y_pred, target_names=cfg.CLASS_NAMES, digits=4))
    print("3x3 Confusion Matrix:\n", cm)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=130)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cfg.CLASS_NAMES)
    disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=False)
    ax.set_title("Yöntem 1: 3-Sınıflı RF Sinyal Tespiti (Drone / WiFi / Bluetooth)")
    plt.tight_layout()
    plt.savefig(cfg.CM_PLOT_PATH)
    print(f"[BİLGİ] 3x3 Confusion Matrix kaydedildi: {cfg.CM_PLOT_PATH}")
    plt.show()


def main():
    cfg = Config()
    set_seed(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SİSTEM] Kullanılan Cihaz: {device}")
    if device.type == "cuda":
        print(f"[SİSTEM] GPU Modeli: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, test_loader, class_weights = create_dataloaders(cfg)

    model = HybridRFClassifier3Class(num_classes=cfg.NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Mimari: {model.__class__.__name__} | Eğitilebilir Parametre Sayısı: {total_params:,}")

    model = train_model(model, train_loader, val_loader, class_weights, cfg, device)
    evaluate_model(model, test_loader, cfg, device)


if __name__ == "__main__":
    main()
