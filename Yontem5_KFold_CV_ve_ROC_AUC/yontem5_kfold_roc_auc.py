"""
YÖNTEM 5: 5-Katlı Çapraz Doğrulama (5-Fold Stratified Cross-Validation) ve ROC-AUC Eğrisi Analizi
Veri Seti: Synthetic RF Spectrograms and IQ Signals for UAV Detection (2.4 GHz band)
Dizin: C:\\Users\\enyas\\Desktop\\Dataset_RF_IQ (58.038 Dosya: 29.019 IQ + 29.019 Spektrogram)

Amacı:
  - Tek bir %70/%15/%15 veri ayrımının (split) şansa bağlı olabileceği şüphesini tamamen ortadan kaldırmak için
    58.038 dosyanın TAMAMI 5 eşit parçaya (5-Fold Stratified K-Fold) bölünür.
  - Her Fold'da %80 (23.215 çift = 46.430 dosya) ile model sıfırdan eğitilir ve kalan %20 (5.804 çift = 11.608 dosya)
    üzerinde test edilir.
  - 5 Fold tamamlandığında veri setindeki 29.019 sinyal çiftinin (58.038 dosyanın) %100'ü eğitim dışı (Out-of-Fold)
    test edilmiş olur; Ortalama ± Standart Sapma (Mean ± Std) metrikleri, 5-Katlı ROC-AUC Eğrisi ve
    29.019 çiftlik Tam Veri Seti Confusion Matrix'i üretilir.
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
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
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

    METHOD_DIR = r"C:\Users\enyas\PycharmProjects\PythonProject1\Yontem5_KFold_CV_ve_ROC_AUC"

    NUM_CLASSES = 2
    CLASS_NAMES = ["Drone Yok (0)", "Drone Var (1)"]

    IQ_CHANNELS = 4           # [I, Q, |IQ|, ∠IQ]
    IQ_FIXED_LEN = 256
    IQ_GLOBAL_SCALE = 0.05
    IMG_SIZE = (64, 64)

    SEED = 42
    N_SPLITS = 5              # 5-Fold Stratified Cross-Validation
    EPOCHS_PER_FOLD = 15      # Her Fold için 15 Epoch (5 Fold x 15 = Toplam 75 Epoch)
    BATCH_SIZE = 128
    IO_THREADS = 16
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    ROC_PLOT_PATH = os.path.join(METHOD_DIR, "roc_auc_5fold_yontem5.png")
    CM_PLOT_PATH = os.path.join(METHOD_DIR, "cm_5fold_toplam_yontem5.png")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# 1. VERİ İNDEKSLEME VE TEK SEFERLİK HIZLI PARALEL ÖN-YÜKLEME (58.038 DOSYA)
# ==============================================================================
FILENAME_PATTERN = re.compile(r"SpectrogramBTFinal_SVG_(\d+)_signal_(\d+)(?:_iq)?\.(?:npy|png)$")


def build_dataset_index(cfg: Config) -> List[Tuple[str, str, int]]:
    os.makedirs(cfg.METHOD_DIR, exist_ok=True)
    iq_files = sorted(glob.glob(os.path.join(cfg.IQ_DIR, "*.npy")))
    if not iq_files:
        raise FileNotFoundError(f"IQ dosyaları bulunamadı: {cfg.IQ_DIR}")

    print(
        f"[BİLGİ] YÖNTEM 5: {cfg.N_SPLITS}-Katlı Çapraz Doğrulama (5-Fold Stratified CV) & ROC-AUC Analizi"
    )

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


class MasterRFDataset(Dataset):
    """
    58.038 dosyanın tamamını başlangıçta tek seferde (~18 sn) belleğe yükler;
    böylece 5 Fold boyunca diskten tekrar okuma yapılmaz.
    """

    def __init__(self, samples: List[Tuple[str, str, int]], cfg: Config):
        self.samples = samples
        self.iq_fixed_len = cfg.IQ_FIXED_LEN
        self.iq_global_scale = cfg.IQ_GLOBAL_SCALE
        self.img_size = cfg.IMG_SIZE

        self.iq_tensors: List[Optional[torch.Tensor]] = [None] * len(samples)
        self.spec_uint8: List[Optional[torch.Tensor]] = [None] * len(samples)
        self.labels = np.array([s[2] for s in samples], dtype=np.int64)
        self._preload_parallel(cfg.IO_THREADS)

    def _process_single_item(self, idx_and_sample: Tuple[int, Tuple[str, str, int]]):
        idx, (iq_path, crop_path, _) = idx_and_sample

        iq_complex = np.load(iq_path)
        i_chan = np.real(iq_complex).astype(np.float32) / self.iq_global_scale
        q_chan = np.imag(iq_complex).astype(np.float32) / self.iq_global_scale
        amp_chan = np.abs(iq_complex).astype(np.float32) / self.iq_global_scale
        phase_chan = np.angle(iq_complex).astype(np.float32) / np.pi

        iq_4d = np.stack([i_chan, q_chan, amp_chan, phase_chan], axis=0)
        t_iq = torch.from_numpy(iq_4d).unsqueeze(0)
        t_iq = F.interpolate(t_iq, size=self.iq_fixed_len, mode="linear", align_corners=False).squeeze(0)
        self.iq_tensors[idx] = t_iq

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
                if done_count % 1000 == 0 or done_count == total:
                    pct = (done_count / total) * 100.0
                    elapsed = time.time() - t0
                    sys.stdout.write(
                        f"\r  [TÜM VERİ SETİ ÖN-YÜKLEME] "
                        f"{done_count * 2:,}/{total * 2:,} dosya (%{pct:5.1f}) | Süre: {elapsed:4.1f}s"
                    )
                    sys.stdout.flush()
        print()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        t_iq = self.iq_tensors[idx]
        t_spec = (self.spec_uint8[idx].to(dtype=torch.float32) / 255.0 - 0.5) / 0.5
        target = torch.tensor(self.labels[idx], dtype=torch.long)
        return (t_iq, t_spec), target


# ==============================================================================
# 2. MODEL MİMARİSİ (4-KANALLI 1D-RESNET + 2D-CNN HİBRİT MODEL)
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


class IQ4Channel1DResNet(nn.Module):
    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.layer1 = ResidualBlock1D(32, 64, stride=2)
        self.layer2 = ResidualBlock1D(64, 128, stride=2)
        self.layer3 = ResidualBlock1D(128, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.layer3(self.layer2(self.layer1(self.stem(x))))).flatten(1)


class Spectrogram2DCNN(nn.Module):
    def __init__(self):
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


class Hybrid4ChannelRFClassifier(nn.Module):
    def __init__(self, iq_channels: int = 4, num_classes: int = 2):
        super().__init__()
        self.iq_branch = IQ4Channel1DResNet(in_channels=iq_channels)
        self.spec_branch = Spectrogram2DCNN()
        self.fusion_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        iq_x, spec_x = inputs
        combined = torch.cat([self.iq_branch.extract_features(iq_x), self.spec_branch.extract_features(spec_x)], dim=1)
        return self.fusion_head(combined)


# ==============================================================================
# 3. 5-FOLD STRATIFIED CROSS-VALIDATION EĞİTİM VE DEĞERLENDİRME DÖNGÜSÜ
# ==============================================================================
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for (iq_x, spec_x), targets in loader:
            iq_x = iq_x.to(device, non_blocking=True)
            spec_x = spec_x.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad()

            logits = model((iq_x, spec_x))
            loss = criterion(logits, targets)

            if is_train:
                loss.backward()
                optimizer.step()

            bs = targets.size(0)
            running_loss += loss.item() * bs
            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += bs

    return running_loss / total, correct / total


def predict_fold(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_targets, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for (iq_x, spec_x), targets in loader:
            iq_x = iq_x.to(device, non_blocking=True)
            spec_x = spec_x.to(device, non_blocking=True)
            logits = model((iq_x, spec_x))
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
    return np.array(all_targets), np.array(all_preds), np.array(all_probs)


def main():
    cfg = Config()
    set_seed(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SİSTEM] Kullanılan Cihaz: {device}")
    if device.type == "cuda":
        print(f"[SİSTEM] GPU Modeli: {torch.cuda.get_device_name(0)}")

    samples = build_dataset_index(cfg)
    master_ds = MasterRFDataset(samples, cfg)
    labels = master_ds.labels

    skf = StratifiedKFold(n_splits=cfg.N_SPLITS, shuffle=True, random_state=cfg.SEED)
    pin_mem = torch.cuda.is_available()

    fold_metrics = []
    oof_targets_all = np.zeros(len(master_ds), dtype=np.int64)
    oof_preds_all = np.zeros(len(master_ds), dtype=np.int64)
    oof_probs_all = np.zeros(len(master_ds), dtype=np.float64)

    mean_fpr = np.linspace(0, 1, 500)
    tprs = []
    roc_curves_data = []

    print("\n" + "=" * 92)
    print(
        f"5-FOLD STRATIFIED CROSS-VALIDATION BAŞLIYOR | "
        f"Her Fold: {cfg.EPOCHS_PER_FOLD} Epoch (Toplam: {cfg.N_SPLITS * cfg.EPOCHS_PER_FOLD} Epoch)"
    )
    print("=" * 92)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        print(
            f"\n--- FOLD [{fold_idx}/{cfg.N_SPLITS}] | "
            f"Train: {len(train_idx):,} çift ({len(train_idx)*2:,} dosya) | "
            f"Test (Out-of-Fold): {len(val_idx):,} çift ({len(val_idx)*2:,} dosya) ---"
        )

        train_sub = Subset(master_ds, train_idx)
        val_sub = Subset(master_ds, val_idx)

        train_loader = DataLoader(train_sub, batch_size=cfg.BATCH_SIZE, shuffle=True, pin_memory=pin_mem)
        val_loader = DataLoader(val_sub, batch_size=cfg.BATCH_SIZE, shuffle=False, pin_memory=pin_mem)

        fold_train_labels = labels[train_idx]
        c0 = max(int((fold_train_labels == 0).sum()), 1)
        c1 = max(int((fold_train_labels == 1).sum()), 1)
        tot = c0 + c1
        class_weights = torch.tensor([tot / (2.0 * c0), tot / (2.0 * c1)], dtype=torch.float32).to(device)

        model = Hybrid4ChannelRFClassifier(iq_channels=cfg.IQ_CHANNELS, num_classes=cfg.NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS_PER_FOLD, eta_min=1e-6)

        best_val_loss = float("inf")
        best_weights = None

        for epoch in range(1, cfg.EPOCHS_PER_FOLD + 1):
            t0 = time.time()
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
            va_loss, va_acc = run_epoch(model, val_loader, criterion, device, None)
            dt = time.time() - t0
            scheduler.step()

            star = ""
            if va_loss < best_val_loss - 1e-5:
                best_val_loss = va_loss
                best_weights = copy.deepcopy(model.state_dict())
                star = " ★"

            print(
                f"  [Fold {fold_idx}] Epoch [{epoch:02d}/{cfg.EPOCHS_PER_FOLD:02d}] ({dt:4.1f}s) | "
                f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:6.2f}% || "
                f"Val Loss: {va_loss:.4f} | Val Acc: {va_acc*100:6.2f}%{star}"
            )

        if best_weights is not None:
            model.load_state_dict(best_weights)

        y_true_f, y_pred_f, y_prob_f = predict_fold(model, val_loader, device)
        oof_targets_all[val_idx] = y_true_f
        oof_preds_all[val_idx] = y_pred_f
        oof_probs_all[val_idx] = y_prob_f

        acc_f = accuracy_score(y_true_f, y_pred_f)
        prec_f = precision_score(y_true_f, y_pred_f, pos_label=1, zero_division=0)
        rec_f = recall_score(y_true_f, y_pred_f, pos_label=1, zero_division=0)
        f1_f = f1_score(y_true_f, y_pred_f, pos_label=1, zero_division=0)
        auc_f = roc_auc_score(y_true_f, y_prob_f)

        fold_metrics.append((fold_idx, acc_f, prec_f, rec_f, f1_f, auc_f))
        print(
            f"  >>> FOLD {fold_idx} SONUCU -> "
            f"Acc: %{acc_f*100:.2f} | Prec: %{prec_f*100:.2f} | "
            f"Recall: %{rec_f*100:.2f} | F1: %{f1_f*100:.2f} | ROC-AUC: %{auc_f*100:.4f}"
        )

        fpr, tpr, _ = roc_curve(y_true_f, y_prob_f)
        roc_curves_data.append((fold_idx, fpr, tpr, auc_f))
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    # ==========================================================================
    # 4. 5-FOLD GENEL ÖZET TABLOSU VE GÖRSELLEŞTİRME
    # ==========================================================================
    arr_m = np.array([m[1:] for m in fold_metrics])  # shape: (5, 5)
    means = arr_m.mean(axis=0)
    stds = arr_m.std(axis=0)

    print("\n" + "=" * 92)
    print("YÖNTEM 5: 5-FOLD STRATIFIED CROSS-VALIDATION NİHAİ SONUÇLARI (58.038 DOSYA TAMAMI)")
    print("=" * 92)
    print(f"{'Fold':<10} | {'Accuracy (%)':<14} | {'Precision (%)':<14} | {'Recall (%)':<14} | {'F1-Score (%)':<14} | {'ROC-AUC (%)'}")
    print("-" * 92)
    for f_id, acc_f, prec_f, rec_f, f1_f, auc_f in fold_metrics:
        print(
            f"Fold {f_id:<5} | {acc_f*100:13.2f}% | {prec_f*100:13.2f}% | "
            f"{rec_f*100:13.2f}% | {f1_f*100:13.2f}% | {auc_f*100:11.4f}%"
        )
    print("-" * 92)
    print(
        f"{'ORT ± STD':<10} | "
        f"{means[0]*100:6.2f}±{stds[0]*100:4.2f}% | "
        f"{means[1]*100:6.2f}±{stds[1]*100:4.2f}% | "
        f"{means[2]*100:6.2f}±{stds[2]*100:4.2f}% | "
        f"{means[3]*100:6.2f}±{stds[3]*100:4.2f}% | "
        f"{means[4]*100:7.4f}±{stds[4]*100:5.4f}%"
    )
    print("=" * 92)

    cm_total = confusion_matrix(oof_targets_all, oof_preds_all)
    print("\nTam Veri Seti (29.019 Çift / 58.038 Dosya) Out-of-Fold Sınıflandırma Raporu:\n")
    print(classification_report(oof_targets_all, oof_preds_all, target_names=cfg.CLASS_NAMES, digits=4))
    print("Tam Veri Seti (29.019 Çift) Out-of-Fold Confusion Matrix:\n", cm_total)

    # --- GRAFİK 1: 5-Fold ROC-AUC Eğrisi (Yakınlaştırılmış Üst Sol Köşe + Tam Görünüm) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=130)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for (f_id, fpr, tpr, auc_f), c in zip(roc_curves_data, colors):
        ax1.plot(fpr, tpr, color=c, alpha=0.65, linewidth=1.5, label=f"Fold {f_id} (AUC = {auc_f:.5f})")
        ax2.plot(fpr * 100, tpr * 100, color=c, alpha=0.65, linewidth=1.5, label=f"Fold {f_id} (%{auc_f*100:.3f})")

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)

    ax1.plot(
        mean_fpr, mean_tpr, color="black", linewidth=2.5,
        label=f"Ortalama ROC (AUC = {means[4]:.5f} ± {stds[4]:.5f})"
    )
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.6)
    ax1.set_xlabel("False Positive Rate (Yanlış Alarm Oranı)")
    ax1.set_ylabel("True Positive Rate (Duyarlılık / Recall)")
    ax1.set_title("5-Fold Cross-Validation ROC Eğrisi (Tam Ölçek)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=8.5)

    ax2.plot(
        mean_fpr * 100, mean_tpr * 100, color="black", linewidth=2.5,
        label=f"Ortalama ROC (%{means[4]*100:.3f} ± %{stds[4]*100:.3f})"
    )
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1.0)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0.0)
    ax2.fill_between(mean_fpr * 100, tprs_lower * 100, tprs_upper * 100, color="gray", alpha=0.25, label="±1 Std. Sapma Bandı")
    ax2.set_xlim(0.0, 2.0)
    ax2.set_ylim(98.0, 100.05)
    ax2.set_xlabel("False Positive Rate / Yanlış Alarm (%) [0% - 2% Yakınlaştırma]")
    ax2.set_ylabel("True Positive Rate / Recall (%) [98% - 100%]")
    ax2.set_title("5-Fold ROC Eğrisi (Yüksek Hassasiyet Bölgesi Yakınlaştırması)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(cfg.ROC_PLOT_PATH)
    plt.close(fig)
    print(f"[BİLGİ] 5-Fold ROC-AUC grafiği kaydedildi: {cfg.ROC_PLOT_PATH}")

    # --- GRAFİK 2: 29.019 Çiftlik (58.038 Dosya) Tam Out-of-Fold Confusion Matrix ---
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=130)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_total, display_labels=cfg.CLASS_NAMES)
    disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=False)
    ax.set_title(
        f"Yöntem 5: 5-Fold CV Toplam Out-of-Fold Confusion Matrix\n"
        f"(29.019 Çift / 58.038 Dosya | Ort. Acc: %{means[0]*100:.2f} ± %{stds[0]*100:.2f})"
    )
    plt.tight_layout()
    plt.savefig(cfg.CM_PLOT_PATH)
    print(f"[BİLGİ] 5-Fold Toplam Confusion Matrix kaydedildi: {cfg.CM_PLOT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
