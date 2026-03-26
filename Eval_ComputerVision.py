"""
Eval_Models.py  —  Joint Evaluator: Stage 1 Detector + Stage 2 Classifier
══════════════════════════════════════════════════════════════════════════════
Evaluates and visualises the performance of both pipeline stages:

  Stage 1 — YOLOv8s generic detector   (Object_Detection.py)
  Stage 2 — EfficientNet-B0 classifier (Object_Classifier.py)

OUTPUTS  (all written to  output/eval/)
  ├── detection/
  │   ├── confusion_matrix.png      normalised class-level confusion matrix
  │   ├── pr_curves.png             per-class Precision-Recall curves
  │   ├── metrics_bar.png           mAP50 / mAP50-95 / P / R bar chart
  │   └── sample_predictions.png   mosaic of annotated val images
  └── classification/
      ├── confusion_matrix.png      top-N class confusion matrix (heatmap)
      ├── per_class_metrics.png     precision / recall / F1 per class bar chart
      ├── top_errors.png            worst-predicted class crops
      └── summary.txt              plain-text report (copy-paste ready)

USAGE
  python Eval_ComputerVision.py [--det-weights PATH] [--cls-weights PATH]
                        [--data-yaml PATH]   [--crops-dir PATH]
                        [--device cpu|0|0,1] [--n-samples 16]
                        [--topn 30]

  All flags are optional — defaults mirror the paths used in
  Object_Detection.py and Object_Classifier.py.

REQUIREMENTS
  pip install ultralytics torch torchvision pillow matplotlib seaborn
              scikit-learn tqdm
"""

import argparse
import textwrap
import random
import sys
import time
from pathlib import Path

# ─────────────────────────── EARLY IMPORT CHECK ──────────────────── #

def _check_imports():
    missing = []
    for pkg, imp in [
        ("ultralytics",   "ultralytics"),
        ("torch",         "torch"),
        ("torchvision",   "torchvision"),
        ("Pillow",        "PIL"),
        ("matplotlib",    "matplotlib"),
        ("seaborn",       "seaborn"),
        ("scikit-learn",  "sklearn"),
        ("tqdm",          "tqdm"),
    ]:
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)
    if missing:
        sys.exit(f"❌  Missing packages: {', '.join(missing)}\n"
                 f"   Run:  pip install {' '.join(missing)}")

_check_imports()

# ─────────────────────────── IMPORTS ─────────────────────────────── #

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")          # headless — safe on HPC / SSH
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, accuracy_score,
)
from tqdm import tqdm
from ultralytics import YOLO
import hashlib
import colorsys

# ─────────────────────────── DEFAULTS ────────────────────────────── #

BASE_DIR       = Path.cwd() / "output"

DET_WEIGHTS    = Path.cwd() / "runs" / "detect" / "sts_detector" / "weights" / "best.pt"
CLS_WEIGHTS    = BASE_DIR / "stage2_checkpoints" / "best.pt"
CLS_CLASSES    = BASE_DIR / "stage2_checkpoints" / "classes.txt"
DATA_YAML      = BASE_DIR / "stage1_dataset" / "data.yaml"
CROPS_DIR      = BASE_DIR / "cropped_dataset"
STAGE1_VAL_IMG = BASE_DIR / "stage1_dataset" / "images" / "val"

EVAL_DIR       = BASE_DIR / "eval"
DET_EVAL_DIR   = EVAL_DIR / "detection"
CLS_EVAL_DIR   = EVAL_DIR / "classification"

IMGSZ_CLS      = 128
BATCH_CLS      = 64
WORKERS        = 8
VAL_SPLIT      = 0.15
RANDOM_SEED    = 42

# ImageNet normalisation (must match training)
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

# Generic Stage-1 class colours (BGR-like for PIL RGB)

def get_class_color(class_name: str) -> str:
    """Generates a consistent, vibrant hex color based on the class name."""
    # Hash the string to get a consistent integer
    hash_int = int(hashlib.md5(class_name.encode('utf-8')).hexdigest(), 16)
    
    # Hue: 0.0 to 1.0 (covers the whole color wheel)
    hue = (hash_int % 360) / 360.0
    # Saturation: 0.5 to 0.8 (keeps it colorful but not blinding)
    sat = 0.5 + ((hash_int // 360) % 30) / 100.0
    # Value (Brightness): 0.8 to 1.0 (keeps it light enough for black text)
    val = 0.8 + ((hash_int // 10800) % 20) / 100.0
    
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"


# ══════════════════════════════════════════════════════════════════ #
#                        STYLE HELPERS                              #
# ══════════════════════════════════════════════════════════════════ #

def _style():
    """Apply a clean dark-ish style once."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi":       150,
        "axes.titlesize":   13,
        "axes.labelsize":   11,
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
        "legend.fontsize":  9,
        "figure.facecolor": "white",
    })

_style()


def _save(fig: plt.Figure, path: Path, tight: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.savefig(path, bbox_inches="tight")
    else:
        fig.savefig(path)
    plt.close(fig)
    print(f"  ✔  Saved → {path}")


# ══════════════════════════════════════════════════════════════════ #
#                   STAGE 1 — DETECTION EVALUATION                  #
# ══════════════════════════════════════════════════════════════════ #

def evaluate_detector(weights: Path, data_yaml: Path, val_img_dir: Path,
                      device: str, n_samples: int) -> None:
    """Run YOLO val + render visual reports."""
    print("\n" + "═" * 60)
    print("  STAGE 1 — GENERIC DETECTOR EVALUATION")
    print("═" * 60)

    if not weights.exists():
        print(f"  ⚠  Weights not found: {weights}")
        print("     Skipping Stage 1 evaluation.\n")
        return
    if not data_yaml.exists():
        print(f"  ⚠  data.yaml not found: {data_yaml}")
        print("     Skipping Stage 1 evaluation.\n")
        return

    # Auto-detect hardware for YOLO
    if device.lower() == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"

    model = YOLO(str(weights))

    # ── 1. Official YOLO val metrics ──────────────────────────────
    print(f"\n  Running YOLO validation on device: {device} …")
    results = model.val(
        data=str(data_yaml),
        device=device,
        verbose=False,
        plots=False,   # we draw our own
        save=False,
    )

    # Extract scalar metrics safely
    def _get(attr, default=0.0):
        try:
            v = getattr(results, attr, default)
            return float(v) if v is not None else default
        except Exception:
            return default

    mp   = _get("box.mp")
    mr   = _get("box.mr")
    map50    = _get("box.map50")
    map5095  = _get("box.map")

    class_names = model.names  # dict {id: name}
    n_classes   = len(class_names)

    print(f"\n  Precision (mean) : {mp:.4f}")
    print(f"  Recall    (mean) : {mr:.4f}")
    print(f"  mAP@0.50         : {map50:.4f}")
    print(f"  mAP@0.50:0.95    : {map5095:.4f}")

    # ── 2. Overall metrics bar chart ──────────────────────────────
    _plot_detection_metrics_bar(mp, mr, map50, map5095)

    # ── 3. Per-class AP bar chart (if available) ──────────────────
    try:
        maps = results.box.maps          # ndarray, one AP50:95 per class
        _plot_per_class_ap(class_names, maps)
    except Exception as e:
        print(f"  ⚠  Per-class AP unavailable: {e}")

    # ── 4. Confusion matrix (YOLO native) ─────────────────────────
    try:
        _plot_detection_confusion(results, class_names)
    except Exception as e:
        print(f"  ⚠  Could not plot detection confusion matrix: {e}")

    # ── 5. Sample prediction mosaic ───────────────────────────────
    if val_img_dir.exists():
        _plot_detection_samples(model, val_img_dir, device, n_samples)
    else:
        print(f"  ⚠  Val image dir not found: {val_img_dir}")

    print("\n  Stage 1 evaluation complete.\n")


def _plot_detection_metrics_bar(mp, mr, map50, map5095):
    labels  = ["Precision", "Recall", "mAP@50", "mAP@50:95"]
    values  = [mp, mr, map50, map5095]
    colors  = ["#42A5F5", "#66BB6A", "#FFA726", "#EF5350"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, width=0.55, zorder=3)
    ax.set_ylim(0, 1.05)
    ax.set_title("Stage 1 — Detection Summary Metrics", fontweight="bold")
    ax.set_ylabel("Score")
    ax.grid(axis="y", zorder=0)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    _save(fig, DET_EVAL_DIR / "metrics_bar.png")


def _plot_per_class_ap(class_names: dict, maps: np.ndarray):
    names  = [class_names[i] for i in range(len(maps))]
    colors = [get_class_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.9 + 1), 4))
    bars = ax.bar(names, maps, color=colors, width=0.6, zorder=3)
    ax.set_ylim(0, 1.05)
    ax.set_title("Stage 1 — Per-Class AP@50:95", fontweight="bold")
    ax.set_ylabel("AP@50:95")
    ax.grid(axis="y", zorder=0)
    plt.xticks(rotation=25, ha="right")

    for bar, val in zip(bars, maps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    _save(fig, DET_EVAL_DIR / "per_class_ap.png")


def _plot_detection_confusion(results, class_names: dict):
    """Pull the raw confusion matrix from YOLO results and plot it."""
    cm_obj = results.confusion_matrix
    matrix = cm_obj.matrix                   # shape (nc+1, nc+1) with background
    nc     = len(class_names)
    # Slice off the background row/col if present
    if matrix.shape[0] == nc + 1:
        matrix = matrix[:nc, :nc]

    # Normalise row-wise
    row_sums = matrix.sum(axis=1, keepdims=True).clip(min=1)
    norm_cm  = matrix / row_sums

    names = [class_names[i] for i in range(nc)]
    fig, ax = plt.subplots(figsize=(max(6, nc), max(5, nc - 1)))
    sns.heatmap(
        norm_cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=names, yticklabels=names,
        linewidths=0.4, linecolor="white",
        vmin=0, vmax=1, ax=ax,
    )
    ax.set_title("Stage 1 — Detection Confusion Matrix (normalised)", fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    _save(fig, DET_EVAL_DIR / "confusion_matrix.png")


def _plot_detection_samples(model, val_img_dir: Path, device: str, n_samples: int):
    """Mosaic of n_samples validation images with YOLO predictions drawn on."""
    images = sorted(val_img_dir.glob("*.png")) + sorted(val_img_dir.glob("*.jpg"))
    if not images:
        print(f"  ⚠  No images found in {val_img_dir}")
        return
    random.seed(RANDOM_SEED)
    chosen = random.sample(images, min(n_samples, len(images)))

    THUMB = 320
    cols  = min(4, len(chosen))
    rows  = (len(chosen) + cols - 1) // cols

    canvas_w = cols * THUMB
    canvas_h = rows * THUMB
    canvas   = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    class_names = model.names

    for idx, img_path in enumerate(chosen):
        result = model(str(img_path), device=device, verbose=False)[0]
        pil    = Image.open(img_path).convert("RGB")
        draw   = ImageDraw.Draw(pil)

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label  = f"{class_names.get(cls_id, str(cls_id))} {conf:.2f}"
            class_name_str = class_names.get(cls_id, str(cls_id))
            color = get_class_color(class_name_str)

            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            tw = draw.textlength(label, font=font)
            draw.rectangle([x1, max(0, y1 - 16), x1 + tw + 4, max(0, y1 - 1)],
                           fill=color)
            draw.text((x1 + 2, max(0, y1 - 15)), label, fill="black", font=font)

        thumb = pil.resize((THUMB, THUMB), Image.LANCZOS)
        row, col = divmod(idx, cols)
        canvas.paste(thumb, (col * THUMB, row * THUMB))

    out = DET_EVAL_DIR / "sample_predictions.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"  ✔  Saved → {out}")


# ══════════════════════════════════════════════════════════════════ #
#                 STAGE 2 — CLASSIFICATION EVALUATION               #
# ══════════════════════════════════════════════════════════════════ #

def evaluate_classifier(weights: Path, classes_txt: Path,
                        crops_dir: Path, device_str: str,
                        topn: int) -> None:
    """Load EfficientNet-B0 checkpoint and compute full classification report."""
    print("\n" + "═" * 60)
    print("  STAGE 2 — SPECIFIC CLASSIFIER EVALUATION")
    print("═" * 60)

    for p, label in [(weights, "Weights"), (crops_dir, "Crops dir")]:
        if not p.exists():
            print(f"  ⚠  {label} not found: {p}")
            print("     Skipping Stage 2 evaluation.\n")
            return

    # Auto-detect hardware for Torch
    if device_str.lower() == "auto":
        actual_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        actual_device = device_str if device_str != "cpu" else "cpu"
        
    device = torch.device(actual_device)
    print(f"\n  Running Stage 2 evaluation on device: {actual_device}")

    # ── Load class names ──────────────────────────────────────────
    if classes_txt.exists():
        class_names = classes_txt.read_text(encoding="utf-8").splitlines()
        class_names = [c.strip() for c in class_names if c.strip()]
    else:
        # Fall back to alphabetical order from disk
        class_names = sorted(d.name for d in crops_dir.iterdir() if d.is_dir())
        print("  ⚠  classes.txt not found — using alphabetical folder order.")

    n_classes = len(class_names)
    print(f"\n  Classes to evaluate : {n_classes}")

    # ── Build val split (same deterministic seed as training) ──────
    val_tf = transforms.Compose([
        transforms.Resize((IMGSZ_CLS, IMGSZ_CLS)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    base       = datasets.ImageFolder(str(crops_dir))
    n_total    = len(base)
    n_val      = max(1, int(n_total * VAL_SPLIT))
    generator  = torch.Generator().manual_seed(RANDOM_SEED)
    indices    = torch.randperm(n_total, generator=generator).tolist()
    val_idx    = indices[:n_val]

    val_ds     = Subset(datasets.ImageFolder(str(crops_dir), transform=val_tf), val_idx)
    val_loader = DataLoader(val_ds, batch_size=BATCH_CLS, shuffle=False,
                            num_workers=WORKERS, pin_memory=(device.type == "cuda"))

    # ── Load model ────────────────────────────────────────────────
    ckpt       = torch.load(weights, map_location="cpu")
    ckpt_names = ckpt.get("classes", class_names)
    n_ckpt     = len(ckpt_names)

    base_model = models.efficientnet_b0(weights=None)
    in_feat    = base_model.classifier[1].in_features
    base_model.classifier[1] = nn.Linear(in_feat, n_ckpt)
    base_model.load_state_dict(ckpt["model"])
    base_model = base_model.to(device).eval()

    # ── Inference pass ────────────────────────────────────────────
    all_preds, all_labels, all_confs = [], [], []
    print("\n  Running inference on validation set …")

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, leave=False, unit="batch"):
            imgs = imgs.to(device)
            logits = base_model(imgs)
            probs  = torch.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_confs.extend(confs.cpu().tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confs  = np.array(all_confs)

    # Map dataset indices to checkpoint class names
    ds_classes  = base.classes                     # alphabetical from disk
    label_names = [ds_classes[i] for i in all_labels]
    pred_names  = [ckpt_names[i] if i < len(ckpt_names) else "?" for i in all_preds]

    acc = accuracy_score(label_names, pred_names)
    print(f"\n  Overall Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Val samples      : {len(all_labels)}")

    # ── Text report ───────────────────────────────────────────────
    report = classification_report(label_names, pred_names, zero_division=0)
    _write_summary(acc, report, n_classes, len(all_labels))

    # ── Per-class metrics bar chart ───────────────────────────────
    prec, rec, f1, sup = precision_recall_fscore_support(
        label_names, pred_names, labels=ckpt_names,
        zero_division=0, average=None,
    )
    _plot_per_class_metrics(ckpt_names, prec, rec, f1, topn)

    # ── Confusion matrix (top-N busiest classes) ──────────────────
    _plot_classification_confusion(label_names, pred_names, ckpt_names, topn)

    # ── Top error crops ───────────────────────────────────────────
    _plot_top_errors(
        val_ds, all_labels, all_preds, all_confs,
        ds_classes, ckpt_names, n_show=16
    )

    # ── Confidence histogram ──────────────────────────────────────
    _plot_confidence_histogram(all_confs, all_labels, all_preds)

    print("\n  Stage 2 evaluation complete.\n")


def _write_summary(acc, report, n_classes, n_samples):
    txt = textwrap.dedent(f"""\
        ╔══════════════════════════════════════════╗
        ║   STAGE 2 — CLASSIFIER EVALUATION REPORT ║
        ╚══════════════════════════════════════════╝

        Overall accuracy : {acc:.4f}  ({acc*100:.2f}%)
        Classes          : {n_classes}
        Val samples      : {n_samples}

        ── Per-class breakdown ──────────────────────

        {report}
    """)
    out = CLS_EVAL_DIR / "summary.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(txt, encoding="utf-8")
    print(f"  ✔  Saved → {out}")


def _plot_per_class_metrics(class_names, prec, rec, f1, topn):
    # Show only top-N by F1 (keeps chart readable for large class counts)
    order  = np.argsort(f1)[::-1][:topn]
    names  = [class_names[i] for i in order]
    p_vals = prec[order]
    r_vals = rec[order]
    f_vals = f1[order]

    x      = np.arange(len(names))
    width  = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.55 + 2), 5))

    ax.bar(x - width, p_vals, width, label="Precision", color="#42A5F5", zorder=3)
    ax.bar(x,         r_vals, width, label="Recall",    color="#66BB6A", zorder=3)
    ax.bar(x + width, f_vals, width, label="F1",        color="#FFA726", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Stage 2 — Per-Class Metrics (top {len(names)} by F1)", fontweight="bold")
    ax.set_ylabel("Score")
    ax.legend(loc="upper right")
    ax.grid(axis="y", zorder=0)
    _save(fig, CLS_EVAL_DIR / "per_class_metrics.png")


def _plot_classification_confusion(label_names, pred_names, class_names, topn):
    # Limit to top-N most frequent ground-truth classes for readability
    from collections import Counter
    top_classes = [c for c, _ in Counter(label_names).most_common(topn)]
    mask_l = np.array([l in top_classes for l in label_names])
    filtered_l = [l for l, m in zip(label_names, mask_l) if m]
    filtered_p = [p for p, m in zip(pred_names,  mask_l) if m]

    cm = confusion_matrix(filtered_l, filtered_p, labels=top_classes)
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    norm_cm  = cm / row_sums

    n = len(top_classes)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.55 + 1), max(8, n * 0.55)))
    sns.heatmap(
        norm_cm, annot=(n <= 25), fmt=".2f", cmap="Blues",
        xticklabels=top_classes, yticklabels=top_classes,
        linewidths=0.2, linecolor="white",
        vmin=0, vmax=1, ax=ax,
        annot_kws={"size": 6},
    )
    ax.set_title(f"Stage 2 — Confusion Matrix (top {n} classes, normalised)",
                 fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    plt.xticks(rotation=45, ha="right", fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    _save(fig, CLS_EVAL_DIR / "confusion_matrix.png")


def _plot_top_errors(val_ds, all_labels, all_preds, all_confs,
                     ds_classes, ckpt_names, n_show=16):
    """Show a grid of the most confidently wrong predictions."""
    wrong_mask = all_labels != all_preds
    if wrong_mask.sum() == 0:
        print("  🎉  No classification errors found on the validation set!")
        return

    wrong_idx   = np.where(wrong_mask)[0]
    wrong_confs = all_confs[wrong_idx]
    order       = np.argsort(wrong_confs)[::-1][:n_show]   # most confident errors
    chosen_idx  = wrong_idx[order]

    cols = 4
    rows = (len(chosen_idx) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 2.5, rows * 2.7),
                             squeeze=False)

    try:
        font_pil = ImageFont.truetype("DejaVuSans-Bold.ttf", 11)
    except Exception:
        font_pil = ImageFont.load_default()

    val_tf_plain = transforms.Compose([transforms.Resize((IMGSZ_CLS, IMGSZ_CLS))])
    raw_ds = val_ds.dataset        # underlying ImageFolder

    for plot_i, sample_i in enumerate(chosen_idx):
        true_lbl  = ds_classes[all_labels[sample_i]]
        pred_lbl  = ckpt_names[all_preds[sample_i]] if all_preds[sample_i] < len(ckpt_names) else "?"
        conf      = all_confs[sample_i]

        img_path, _ = raw_ds.samples[val_ds.indices[sample_i]]
        pil = Image.open(img_path).convert("RGB").resize((IMGSZ_CLS, IMGSZ_CLS))

        r, c  = divmod(plot_i, cols)
        ax    = axes[r][c]
        ax.imshow(pil)
        ax.set_title(
            f"GT: {true_lbl}\nPred: {pred_lbl} ({conf:.2f})",
            fontsize=7, color="red", pad=2,
        )
        ax.axis("off")

    # Hide empty cells
    for empty_i in range(len(chosen_idx), rows * cols):
        r, c = divmod(empty_i, cols)
        axes[r][c].axis("off")

    fig.suptitle(f"Stage 2 — Top {len(chosen_idx)} Most Confident Errors",
                 fontweight="bold", fontsize=12)
    _save(fig, CLS_EVAL_DIR / "top_errors.png")


def _plot_confidence_histogram(all_confs, all_labels, all_preds):
    correct = all_confs[all_labels == all_preds]
    wrong   = all_confs[all_labels != all_preds]

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 41)
    ax.hist(correct, bins=bins, alpha=0.65, color="#66BB6A", label=f"Correct  (n={len(correct)})")
    ax.hist(wrong,   bins=bins, alpha=0.65, color="#EF5350", label=f"Wrong    (n={len(wrong)})")
    ax.set_xlabel("Softmax Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Stage 2 — Confidence Distribution (correct vs. wrong)",
                 fontweight="bold")
    ax.legend()
    ax.grid(axis="y")
    _save(fig, CLS_EVAL_DIR / "confidence_histogram.png")


# ══════════════════════════════════════════════════════════════════ #
#                         PIPELINE SUMMARY                          #
# ══════════════════════════════════════════════════════════════════ #

def print_summary(t_start: float):
    elapsed = time.time() - t_start
    print("\n" + "═" * 60)
    print("  EVALUATION COMPLETE")
    print("═" * 60)
    print(f"  Total time : {elapsed:.1f}s")
    print(f"\n  Output directory:")
    print(f"    {EVAL_DIR}")
    print(f"\n  Detection  → {DET_EVAL_DIR}")
    for p in sorted(DET_EVAL_DIR.glob("*.png")):
        print(f"    {p.name}")
    print(f"\n  Classification → {CLS_EVAL_DIR}")
    for p in sorted(CLS_EVAL_DIR.glob("*.png")):
        print(f"    {p.name}")
    txt = CLS_EVAL_DIR / "summary.txt"
    if txt.exists():
        print(f"    {txt.name}")
    print()


# ══════════════════════════════════════════════════════════════════ #
#                              MAIN                                  #
# ══════════════════════════════════════════════════════════════════ #

def parse_args():
    ap = argparse.ArgumentParser(
        description="Evaluate Stage 1 (YOLO detector) and Stage 2 (EfficientNet classifier)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--device",      type=str,  default="auto",
                    help="Device: 'auto', 'cpu', '0', '0,1' …  (default: auto)")
    ap.add_argument("--det-weights", type=Path, default=DET_WEIGHTS,
                    help=f"Path to Stage 1 best.pt  (default: {DET_WEIGHTS})")
    ap.add_argument("--cls-weights", type=Path, default=CLS_WEIGHTS,
                    help=f"Path to Stage 2 best.pt  (default: {CLS_WEIGHTS})")
    ap.add_argument("--data-yaml",   type=Path, default=DATA_YAML,
                    help=f"Stage 1 data.yaml  (default: {DATA_YAML})")
    ap.add_argument("--crops-dir",   type=Path, default=CROPS_DIR,
                    help=f"Stage 2 crops dir  (default: {CROPS_DIR})")
    ap.add_argument("--val-img-dir", type=Path, default=STAGE1_VAL_IMG,
                    help=f"Stage 1 val images  (default: {STAGE1_VAL_IMG})")
    ap.add_argument("--n-samples",   type=int,  default=16,
                    help="Detection sample mosaic count  (default: 16)")
    ap.add_argument("--topn",        type=int,  default=30,
                    help="Max classes shown in classification plots  (default: 30)")
    ap.add_argument("--skip-det",    action="store_true",
                    help="Skip Stage 1 (detector) evaluation")
    ap.add_argument("--skip-cls",    action="store_true",
                    help="Skip Stage 2 (classifier) evaluation")
    return ap.parse_args()


def _ensure_dirs() -> None:
    """Pre-create every output directory so nothing crashes mid-run."""
    Path("logs").mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    DET_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    CLS_EVAL_DIR.mkdir(parents=True, exist_ok=True)


def main():
    args    = parse_args()
    t_start = time.time()

    _ensure_dirs()

    print("\n" + "═" * 60)
    print("  STS MODEL EVALUATOR  —  v1.0")
    print("═" * 60)
    print(f"  Stage 1 weights  : {args.det_weights}")
    print(f"  Stage 2 weights  : {args.cls_weights}")
    print(f"  data.yaml        : {args.data_yaml}")
    print(f"  Crops dir        : {args.crops_dir}")
    print(f"  Device           : {args.device}")
    print(f"  Top-N classes    : {args.topn}")
    print(f"  Mosaic samples   : {args.n_samples}")

    if not args.skip_det:
        evaluate_detector(
            weights    = args.det_weights,
            data_yaml  = args.data_yaml,
            val_img_dir= args.val_img_dir,
            device     = args.device,
            n_samples  = args.n_samples,
        )

    if not args.skip_cls:
        evaluate_classifier(
            weights    = args.cls_weights,
            classes_txt= CLS_CLASSES,
            crops_dir  = args.crops_dir,
            device_str = args.device,
            topn       = args.topn,
        )

    print_summary(t_start)


if __name__ == "__main__":
    main()