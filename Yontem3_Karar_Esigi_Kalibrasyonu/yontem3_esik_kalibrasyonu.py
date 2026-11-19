"""
YÖNTEM 3: Karar Eşiği (Decision Threshold) Kalibrasyonu ve Precision-Recall Optimizasyonu
Veri Seti: Synthetic RF Spectrograms and IQ Signals for UAV Detection (2.4 GHz band)
Dizin: C:\\Users\\enyas\\Desktop\\Dataset_RF_IQ (58.038 Dosya: 29.019 IQ + 29.019 Spektrogram)

Amacı:
  - Standart sınıflandırmada kullanılan sabit P(Drone) >= 0.50 karar eşiği yerine,
    Validation (Doğrulama) seti üzerinde 1.000 farklı eşik değeri (tau in [0.001, 0.999]) taranarak:
      1) En Yüksek F1-Skorunu veren Optimal Eşik (tau_f1*)
      2) Anti-Drone / Kritik Güvenlik Modu Eşiği (tau_recall* -> Kaçırılan Drone FN = 0 veya minimum)
    hesaplanır ve Test seti üzerinde standart eşik (0.50) ile karşılaştırılır.
"""

import os
import re
import sys
import glob
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
    precision_recall_curve,
    average_precision_score,
    ConfusionMatrixDisplay,
)


# ==============================================================================
# 0. KONFİGÜRASYON
# ==============================================================================
class Config:
    DATASET_DIR = r"C:\Users\enyas\Desktop\Dataset_RF_IQ"
    IQ_DIR = os.path.join(DATASET_DIR, "iq_signals")
    CROP_DIR = os.path.join(DATASET_DIR, "signal_crops")

    METHOD_DIR = r"C:\Users\enyas\PycharmProjects\PythonProject1\Yontem3_Karar_Esigi_Kalibrasyonu"

    # Eğitilmiş ağırlık dosyaları (Yöntem 2 4-Kanallı model varsa onu, yoksa v1 Baseline modeli yükler)
    WEIGHTS_4CH = os.path.join(DATASET_DIR, "best_drone_rf_model_yontem2_4kanal.pt")
    WEIGHTS_2CH = os.path.join(DATASET_DIR, "best_drone_rf_model.pt")

    IQ_FIXED_LEN = 256
    IQ_GLOBAL_SCALE = 0.05
    IMG_SIZE = (64, 64)

    SEED = 42
    BATCH_SIZE = 128
    IO_THREADS = 16

    CURVE_PLOT_PATH = os.path.join(METHOD_DIR, "threshold_analizi_yontem3.png")
    CM_PLOT_PATH = os.path.join(METHOD_DIR, "cm_yontem3_kalibre.png")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# 1. VERİ YÜKLEME VE MİMARİ TANIMLARI
# ==============================================================================
FILENAME_PATTERN = re.compile(r"SpectrogramBTFinal_SVG_(\d+)_signal_(\d+)(?:_iq)?\.(?:npy|png)$")


def build_dataset_index(cfg: Config) -> List[Tuple[str, str, int]]:
    os.makedirs(cfg.METHOD_DIR, exist_ok=True)
    iq_files = sorted(glob.glob(os.path.join(cfg.IQ_DIR, "*.npy")))
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

    return samples


class RFCalibrationDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, str, int]], cfg: Config, iq_channels: int = 4, split_name: str = "Val"):
        self.samples = samples
        self.iq_channels = iq_channels
        self.iq_fixed_len = cfg.IQ_FIXED_LEN
        self.iq_global_scale = cfg.IQ_GLOBAL_SCALE
        self.img_size = cfg.IMG_SIZE
        self.split_name = split_name

        self.iq_tensors: List[Optional[torch.Tensor]] = [None] * len(samples)
        self.spec_uint8: List[Optional[torch.Tensor]] = [None] * len(samples)
        self._preload_parallel(cfg.IO_THREADS)

    def _process_single_item(self, idx_and_sample: Tuple[int, Tuple[str, str, int]]):
        idx, (iq_path, crop_path, _) = idx_and_sample
        iq_complex = np.load(iq_path)
        i_chan = np.real(iq_complex).astype(np.float32) / self.iq_global_scale
        q_chan = np.imag(iq_complex).astype(np.float32) / self.iq_global_scale

        if self.iq_channels == 4:
            amp_chan = np.abs(iq_complex).astype(np.float32) / self.iq_global_scale
            phase_chan = np.angle(iq_complex).astype(np.float32) / np.pi
            iq_nd = np.stack([i_chan, q_chan, amp_chan, phase_chan], axis=0)
        else:
            iq_nd = np.stack([i_chan, q_chan], axis=0)

        t_iq = torch.from_numpy(iq_nd).unsqueeze(0)
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
                if done_count % 500 == 0 or done_count == total:
                    pct = (done_count / total) * 100.0
                    elapsed = time.time() - t0
                    sys.stdout.write(
                        f"\r  [VERİ YÜKLEME - {self.split_name:5s}] "
                        f"{done_count * 2:,}/{total * 2:,} dosya (%{pct:5.1f}) | Süre: {elapsed:4.1f}s"
                    )
                    sys.stdout.flush()
        print()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        _, _, label = self.samples[idx]
        t_spec = (self.spec_uint8[idx].to(dtype=torch.float32) / 255.0 - 0.5) / 0.5
        return (self.iq_tensors[idx], t_spec), torch.tensor(label, dtype=torch.long)


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


class IQ1DResNetBranch(nn.Module):
    def __init__(self, in_channels: int = 4, include_classifier: bool = False):
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
        if include_classifier:
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 2)
            )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.layer3(self.layer2(self.layer1(self.stem(x))))).flatten(1)


class Spectrogram2DCNNBranch(nn.Module):
    def __init__(self, include_classifier: bool = False):
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
        if include_classifier:
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 2)
            )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)


class FlexibleHybridRFClassifier(nn.Module):
    def __init__(self, iq_channels: int = 4, include_sub_classifiers: bool = False):
        super().__init__()
        self.iq_branch = IQ1DResNetBranch(in_channels=iq_channels, include_classifier=include_sub_classifiers)
        self.spec_branch = Spectrogram2DCNNBranch(include_classifier=include_sub_classifiers)
        self.fusion_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        iq_x, spec_x = inputs
        combined = torch.cat([self.iq_branch.extract_features(iq_x), self.spec_branch.extract_features(spec_x)], dim=1)
        return self.fusion_head(combined)


# ==============================================================================
# 2. OLASILIK ÇIKARIMI VE KARAR EŞİĞİ (THRESHOLD) TARAMASI
# ==============================================================================
def predict_probabilities(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_list = []
    targets_list = []

    with torch.no_grad():
        for (iq_x, spec_x), targets in loader:
            iq_x = iq_x.to(device, non_blocking=True)
            spec_x = spec_x.to(device, non_blocking=True)
            logits = model((iq_x, spec_x))
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()  # P(Drone Var = 1 | x)
            probs_list.extend(probs)
            targets_list.extend(targets.numpy())

    return np.array(targets_list), np.array(probs_list)


def calibrate_and_evaluate(
    val_y: np.ndarray,
    val_probs: np.ndarray,
    test_y: np.ndarray,
    test_probs: np.ndarray,
    cfg: Config,
    model_label: str,
):
    thresholds = np.linspace(0.005, 0.995, 991)
    val_f1s, val_precs, val_recs, val_fns, val_fps = [], [], [], [], []

    for tau in thresholds:
        preds = (val_probs >= tau).astype(int)
        val_f1s.append(f1_score(val_y, preds, pos_label=1, zero_division=0))
        val_precs.append(precision_score(val_y, preds, pos_label=1, zero_division=0))
        val_recs.append(recall_score(val_y, preds, pos_label=1, zero_division=0))
        cm = confusion_matrix(val_y, preds)
        val_fns.append(cm[1, 0])
        val_fps.append(cm[0, 1])

    val_f1s = np.array(val_f1s)
    val_precs = np.array(val_precs)
    val_recs = np.array(val_recs)
    val_fns = np.array(val_fns)
    val_fps = np.array(val_fps)

    # 1. En Yüksek F1-Skoru Veren Optimal Eşik (tau_opt*)
    best_f1_idx = int(np.argmax(val_f1s))
    tau_opt = float(thresholds[best_f1_idx])

    # 2. Anti-Drone Kritik Güvenlik Eşiği (Recall >= %99.93 veya FN minimum iken en yüksek F1)
    high_recall_mask = val_recs >= 0.9993
    if np.any(high_recall_mask):
        cand_indices = np.where(high_recall_mask)[0]
        best_safe_idx = cand_indices[int(np.argmax(val_f1s[cand_indices]))]
    else:
        best_safe_idx = int(np.argmax(val_recs - 0.05 * (1.0 - val_precs)))
    tau_safe = float(thresholds[best_safe_idx])

    pr_auc = average_precision_score(test_y, test_probs)

    print("\n" + "=" * 86)
    print(f"YÖNTEM 3: KARAR EŞİĞİ (THRESHOLD) KALİBRASYONU SONUÇLARI ({model_label})")
    print("=" * 86)
    print(f"Test Seti PR-AUC (Average Precision) Skoru : {pr_auc * 100:.4f}% ({pr_auc:.6f})")
    print(f"Doğrulama Setinde Bulunan Optimal F1 Eşiği : tau_opt  = {tau_opt:.4f}")
    print(f"Doğrulama Setinde Anti-Drone Güvenlik Eşiği: tau_safe = {tau_safe:.4f}")
    print("-" * 86)

    modes = [
        ("1) Standart Eşik (tau = 0.5000)", 0.50),
        (f"2) Kalibre Optimal F1 Eşiği (tau* = {tau_opt:.4f})", tau_opt),
        (f"3) Anti-Drone Kritik Güvenlik Eşiği (tau_safe = {tau_safe:.4f})", tau_safe),
    ]

    cms = []
    for title, tau in modes:
        t_preds = (test_probs >= tau).astype(int)
        acc = accuracy_score(test_y, t_preds)
        prec = precision_score(test_y, t_preds, pos_label=1, zero_division=0)
        rec = recall_score(test_y, t_preds, pos_label=1, zero_division=0)
        f1 = f1_score(test_y, t_preds, pos_label=1, zero_division=0)
        cm = confusion_matrix(test_y, t_preds)
        cms.append((title, tau, cm, acc, prec, rec, f1))

        print(f"\n>>> {title}")
        print(
            f"    Accuracy: {acc*100:.2f}% | Precision: {prec*100:.2f}% | "
            f"Recall (Drone Yakalama): {rec*100:.2f}% | F1-Score: {f1*100:.2f}%"
        )
        print(
            f"    Confusion Matrix -> TN: {cm[0,0]:4d} | FP (Yanlış Alarm): {cm[0,1]:2d} || "
            f"FN (Kaçırılan Drone): {cm[1,0]:2d} | TP (Yakalanan Drone): {cm[1,1]:4d}"
        )

    # --- GRAFİK 1: Eşik Taraması (Precision, Recall, F1) ve Precision-Recall Eğrisi ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=130)

    ax1.plot(thresholds, val_precs * 100, label="Precision (%)", color="#1f77b4", linewidth=2)
    ax1.plot(thresholds, val_recs * 100, label="Recall / Drone Yakalama (%)", color="#2ca02c", linewidth=2)
    ax1.plot(thresholds, val_f1s * 100, label="F1-Score (%)", color="#d62728", linewidth=2.2)
    ax1.axvline(0.50, color="gray", linestyle="--", label="Standart Eşik (0.50)")
    ax1.axvline(tau_opt, color="#9467bd", linestyle="-.", linewidth=1.8, label=f"Optimal F1 Eşiği ({tau_opt:.3f})")
    ax1.axvline(tau_safe, color="#ff7f0e", linestyle=":", linewidth=2.0, label=f"Anti-Drone Eşiği ({tau_safe:.3f})")
    ax1.set_ylim(95.0, 100.2)
    ax1.set_xlabel("Karar Eşiği (Decision Threshold τ)")
    ax1.set_ylabel("Metrik Başarısı (%)")
    ax1.set_title("Karar Eşiğine Göre Precision, Recall ve F1 Değişimi")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", fontsize=8.5)

    prec_curve, rec_curve, _ = precision_recall_curve(test_y, test_probs)
    ax2.plot(rec_curve * 100, prec_curve * 100, color="#d62728", linewidth=2.2, label=f"PR Eğrisi (AP = {pr_auc*100:.3f}%)")
    ax2.set_xlim(95.0, 100.1)
    ax2.set_ylim(95.0, 100.1)
    ax2.set_xlabel("Duyarlılık / Recall (%)")
    ax2.set_ylabel("Kesinlik / Precision (%)")
    ax2.set_title("Test Seti Precision-Recall (PR) Eğrisi")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(cfg.CURVE_PLOT_PATH)
    plt.close(fig)
    print(f"\n[BİLGİ] Eşik analizi grafiği kaydedildi: {cfg.CURVE_PLOT_PATH}")

    # --- GRAFİK 2: Yan Yana Confusion Matrix Karşılaştırması ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=130)
    for ax, (title, tau, cm, acc, prec, rec, f1) in zip(axes, cms):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Drone Yok (0)", "Drone Var (1)"])
        disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=False)
        ax.set_title(f"{title}\nAcc: %{acc*100:.2f} | Recall: %{rec*100:.2f} | FN: {cm[1,0]}", fontsize=10)

    plt.tight_layout()
    plt.savefig(cfg.CM_PLOT_PATH)
    print(f"[BİLGİ] Kalibre Confusion Matrix karşılaştırması kaydedildi: {cfg.CM_PLOT_PATH}")
    plt.show()


def main():
    cfg = Config()
    set_seed(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SİSTEM] Kullanılan Cihaz: {device}")
    if device.type == "cuda":
        print(f"[SİSTEM] GPU Modeli: {torch.cuda.get_device_name(0)}")

    # Hangi ağırlık dosyasının mevcut olduğunu kontrol et
    if os.path.exists(cfg.WEIGHTS_2CH):
        weights_path = cfg.WEIGHTS_2CH
        iq_channels = 2
        include_sub = True
        model_label = "Baseline v1 Hibrit Model (best_drone_rf_model.pt)"
    elif os.path.exists(cfg.WEIGHTS_4CH):
        weights_path = cfg.WEIGHTS_4CH
        iq_channels = 4
        include_sub = False
        model_label = "Yöntem 2 4-Kanallı Hibrit Model"
    else:
        raise FileNotFoundError("Eğitilmiş model ağırlık dosyası (.pt) bulunamadı!")

    print(f"[BİLGİ] Yüklenecek Eğitilmiş Model: {model_label}")
    print(f"[BİLGİ] Ağırlık Dosyası: {weights_path}")

    samples = build_dataset_index(cfg)
    labels = [s[2] for s in samples]

    _, temp_samples, _, temp_labels = train_test_split(
        samples, labels, test_size=0.30, random_state=cfg.SEED, stratify=labels
    )
    val_samples, test_samples = train_test_split(
        temp_samples, test_size=0.50, random_state=cfg.SEED, stratify=temp_labels
    )

    print(
        f"[BİLGİ] Kalibrasyon (Val): {len(val_samples):,} çift ({len(val_samples)*2:,} dosya) | "
        f"Değerlendirme (Test): {len(test_samples):,} çift ({len(test_samples)*2:,} dosya)"
    )

    val_ds = RFCalibrationDataset(val_samples, cfg, iq_channels=iq_channels, split_name="Val")
    test_ds = RFCalibrationDataset(test_samples, cfg, iq_channels=iq_channels, split_name="Test")

    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

    model = FlexibleHybridRFClassifier(iq_channels=iq_channels, include_sub_classifiers=include_sub).to(device)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    val_y, val_probs = predict_probabilities(model, val_loader, device)
    test_y, test_probs = predict_probabilities(model, test_loader, device)

    calibrate_and_evaluate(val_y, val_probs, test_y, test_probs, cfg, model_label)


if __name__ == "__main__":
    main()
