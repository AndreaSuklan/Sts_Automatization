"""
Object_Classifier.py  -  Stage 2: Specific Identity Classifier
═══════════════════════════════════════════════════════════════
Trains an EfficientNet-B0 (pretrained on ImageNet) to identify the
exact name of every crop produced by Yolo_Crop.py.

Each sub-folder in output/cropped_dataset/ is one class, e.g.:
  Card_Strike_R/    Enemy_JawWorm/    Intent_ATTACK/    Power_strength/

CHECKPOINTS SAVED
  output/stage2_checkpoints/
  ├── last.pt      always up to date (resume from here)
  ├── best.pt      highest val accuracy so far
  └── classes.txt  class index -> name mapping for inference

RESUME LOGIC
  If output/stage2_checkpoints/last.pt exists, training resumes
  automatically.  Delete it to force a fresh start.

CLI ARGUMENTS (all optional - defaults match original CONFIG)
  --epochs   INT     number of training epochs           (100)
  --batch    INT     batch size per GPU                  (64)
  --gpu-ids  STR     comma-separated GPU ids or 'cpu'    (0,1)
  --lr       FLOAT   initial learning rate               (0.001)
  --imgsz    INT     crop resize dimension (square)      (128)
"""

import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

# ─────────────────────────── FIXED CONFIG ────────────────────────── #

BASE_DIR  = Path.cwd() / "output"
CROPS_DIR = BASE_DIR / "cropped_dataset"
CKPT_DIR  = BASE_DIR / "stage2_checkpoints"
LAST_CKPT = CKPT_DIR / "last.pt"
BEST_CKPT = CKPT_DIR / "best.pt"

VAL_SPLIT   = 0.15
RANDOM_SEED = 42
WORKERS     = 8

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────────────────────────── #


def _parse_gpu_ids(value: str):
    """
    Parse a GPU-ids string into a list of ints, or an empty list for CPU.

      'cpu'   -> []
      '0'     -> [0]
      '0,1'   -> [0, 1]
    """
    if value.strip().lower() == "cpu":
        return []
    try:
        ids = [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid --gpu-ids value: {value!r}. "
            "Use 'cpu', '0' (single GPU), or '0,1' (multi-GPU)."
        )
    if not ids:
        raise argparse.ArgumentTypeError("--gpu-ids cannot be empty; use 'cpu' for CPU.")
    return ids


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Stage 2 - EfficientNet-B0 Specific Identity Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs",
    )
    ap.add_argument(
        "--batch", type=int, default=64,
        help="Batch size per GPU",
    )
    ap.add_argument(
        "--gpu-ids", type=str, default="0,1", dest="gpu_ids",
        help="GPU ids to use: 'cpu', '0' (single), or '0,1' (multi-GPU)",
    )
    ap.add_argument(
        "--lr", type=float, default=1e-3,
        help="Initial learning rate (Adam)",
    )
    ap.add_argument(
        "--imgsz", type=int, default=128,
        help="Crop resize dimension in pixels (square)",
    )
    return ap.parse_args()


def _ensure_dirs() -> None:
    Path("logs").mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    if not CROPS_DIR.exists() or not any(CROPS_DIR.iterdir()):
        sys.exit(
            f"Crops directory not found or empty: {CROPS_DIR}\n"
            "Run Yolo_Crop.py first to generate the cropped dataset."
        )


def build_dataloaders(imgsz: int, batch: int):
    """Returns (train_loader, val_loader, class_names)."""
    train_tf = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])

    base        = datasets.ImageFolder(str(CROPS_DIR))
    n_total     = len(base)
    n_val       = max(1, int(n_total * VAL_SPLIT))
    n_train     = n_total - n_val
    generator   = torch.Generator().manual_seed(RANDOM_SEED)
    indices     = torch.randperm(n_total, generator=generator).tolist()
    train_idx   = indices[n_val:]
    val_idx     = indices[:n_val]
    class_names = base.classes

    print(f"  Total crops      : {n_total}")
    print(f"  Unique classes   : {len(class_names)}")
    print(f"  Train / Val      : {n_train} / {n_val}")

    train_ds = Subset(datasets.ImageFolder(str(CROPS_DIR), transform=train_tf), train_idx)
    val_ds   = Subset(datasets.ImageFolder(str(CROPS_DIR), transform=val_tf),   val_idx)

    use_pin = len(class_names) > 0  # True when a GPU is available
    train_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True,
        num_workers=WORKERS, pin_memory=use_pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch, shuffle=False,
        num_workers=WORKERS, pin_memory=use_pin,
    )
    return train_loader, val_loader, class_names


def build_model(n_classes: int) -> nn.Module:
    """EfficientNet-B0 with its classifier head replaced."""
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model   = models.efficientnet_b0(weights=weights)
    in_feat = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feat, n_classes)
    return model


def train() -> None:
    args    = _parse_args()
    gpu_ids = _parse_gpu_ids(args.gpu_ids)

    _ensure_dirs()

    # ── Device setup ──────────────────────────────────────────────────
    if gpu_ids and torch.cuda.is_available():
        device    = torch.device(f"cuda:{gpu_ids[0]}")
        use_multi = len(gpu_ids) > 1
    else:
        device    = torch.device("cpu")
        use_multi = False
        gpu_ids   = []

    n_gpus           = len(gpu_ids) if use_multi else (1 if device.type == "cuda" else 0)
    effective_batch  = args.batch * max(n_gpus, 1)

    print("=" * 60)
    print("  Stage 2 - EfficientNet-B0 Specific Classifier")
    print("=" * 60)
    print(f"  Epochs         : {args.epochs}")
    print(f"  Batch/GPU      : {args.batch}  (effective: {effective_batch})")
    print(f"  GPU ids        : {gpu_ids or 'CPU'}")
    print(f"  Learning rate  : {args.lr}")
    print(f"  Image size     : {args.imgsz}")
    print("=" * 60)

    train_loader, val_loader, class_names = build_dataloaders(args.imgsz, args.batch)
    n_classes  = len(class_names)
    base_model = build_model(n_classes)

    # ── Optimizer / scheduler / criterion ─────────────────────────────
    optimizer = torch.optim.Adam(base_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # ── Resume from last checkpoint (load weights BEFORE DataParallel) ─
    start_epoch  = 0
    best_val_acc = 0.0

    if LAST_CKPT.exists():
        print(f"Resuming Stage 2 training from {LAST_CKPT}")
        ckpt = torch.load(LAST_CKPT, map_location="cpu")
        base_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"  Resuming from epoch {start_epoch} | best val acc: {best_val_acc:.2%}\n")
    else:
        print("Starting Stage 2 - EfficientNet-B0 Classifier\n")

    # ── Wrap in DataParallel AFTER loading weights ─────────────────────
    base_model = base_model.to(device)
    model: nn.Module = (
        nn.DataParallel(base_model, device_ids=gpu_ids) if use_multi else base_model
    )

    (CKPT_DIR / "classes.txt").write_text("\n".join(class_names), encoding="utf-8")

    # ── Training loop ──────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        model.train()
        train_loss = train_correct = train_total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * imgs.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total   += imgs.size(0)

        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                val_correct += (model(imgs).argmax(1) == labels).sum().item()
                val_total   += imgs.size(0)

        scheduler.step()

        train_acc = train_correct / train_total
        val_acc   = val_correct   / val_total
        elapsed   = time.time() - t0

        print(
            f"  Epoch {epoch+1:>3}/{args.epochs}"
            f"  loss {train_loss/train_total:.4f}"
            f"  train {train_acc:.2%}"
            f"  val {val_acc:.2%}"
            f"  ({elapsed:.1f}s)"
        )

        # Always save the plain module (unwrap DataParallel) for portability.
        raw_state = (model.module if use_multi else model).state_dict()

        torch.save({
            "epoch":        epoch,
            "model":        raw_state,
            "optimizer":    optimizer.state_dict(),
            "scheduler":    scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "classes":      class_names,
        }, LAST_CKPT)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":   epoch,
                "model":   raw_state,
                "classes": class_names,
            }, BEST_CKPT)
            print(f"  New best val acc: {best_val_acc:.2%} - saved best.pt")

    print(f"\nStage 2 complete.")
    print(f"Best val accuracy : {best_val_acc:.2%}")
    print(f"Best weights      -> {BEST_CKPT}")
    print(f"Class mapping     -> {CKPT_DIR / 'classes.txt'}")


if __name__ == "__main__":
    train()