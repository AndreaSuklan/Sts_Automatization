"""
train_stage2_classifier.py  —  Stage 2: Specific Identity Classifier
══════════════════════════════════════════════════════════════════════
Trains an EfficientNet-B0 (pretrained on ImageNet) to identify the
exact name of every crop produced by Yolo_Crop.py.

Each sub-folder in output/cropped_dataset/ is one class, e.g.:
  Card_Strike_R/    Enemy_JawWorm/    Intent_ATTACK/    Power_strength/

The model learns: "given a tight crop of an object, what exactly IS it?"

CHECKPOINTS SAVED
  output/stage2_checkpoints/
  ├── last.pt   ← always up to date (resume from here)
  ├── best.pt   ← highest val accuracy so far
  └── classes.txt ← class index → name mapping for inference

RESUME LOGIC
  If  output/stage2_checkpoints/last.pt  exists, training resumes
  automatically.  Delete it to force a fresh start.

NOTES ON HYPER-PARAMETERS
  • IMGSZ = 128 is plenty for icon-sized crops; increasing it gives
    diminishing returns and slows training significantly.
  • LR schedule: cosine annealing → no manual decay needed.
  • Card crops should have been extracted with PADDING_PX ≥ 20 in
    Yolo_Crop.py to capture the card art frame (strong visual cue).
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

# ─────────────────────────── CONFIG ─────────────────────────────── #

BASE_DIR        = Path.cwd() / "output"
CROPS_DIR       = BASE_DIR / "cropped_dataset"       # output of Yolo_Crop.py
CKPT_DIR        = BASE_DIR / "stage2_checkpoints"
LAST_CKPT       = CKPT_DIR / "last.pt"
BEST_CKPT       = CKPT_DIR / "best.pt"

# ── Training hyper-parameters ─────────────────────────────────────
IMGSZ      = 128    # crops are small icons; 128px is the sweet spot
BATCH      = 64
EPOCHS     = 100
LR         = 1e-3
VAL_SPLIT  = 0.15
RANDOM_SEED= 42

# GPU ids to use — mirrors Stage 1 config.
# DataParallel splits each batch evenly across all listed GPUs.
# Set to []  to use CPU, or [0] for a single GPU.
GPU_IDS    = [0, 1]
WORKERS    = 8   # match Stage 1

# ────────────────────────────────────────────────────────────────── #

# ImageNet normalisation constants (required for pretrained weights)
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def _ensure_dirs() -> None:
    """Create all directories this script may write to."""
    Path("logs").mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # Provide a clear message if the crops folder is missing rather than
    # letting torchvision throw a cryptic FileNotFoundError later.
    if not CROPS_DIR.exists() or not any(CROPS_DIR.iterdir()):
        sys.exit(
            f"❌  Crops directory not found or empty: {CROPS_DIR}\n"
            "    Run Yolo_Crop.py first to generate the cropped dataset."
        )


def build_dataloaders() -> tuple[DataLoader, DataLoader, list[str]]:
    """
    Returns (train_loader, val_loader, class_names).

    Uses two separate ImageFolder objects so each split can have its
    own transform (augmentation only on train).
    """
    train_tf = transforms.Compose([
        transforms.Resize((IMGSZ, IMGSZ)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1),
        transforms.RandomRotation(degrees=5),   # cards are slightly tilted in-game
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMGSZ, IMGSZ)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])

    # Build the index split once (deterministic seed)
    base       = datasets.ImageFolder(str(CROPS_DIR))
    n_total    = len(base)
    n_val      = max(1, int(n_total * VAL_SPLIT))
    n_train    = n_total - n_val
    generator  = torch.Generator().manual_seed(RANDOM_SEED)
    indices    = torch.randperm(n_total, generator=generator).tolist()
    train_idx  = indices[n_val:]
    val_idx    = indices[:n_val]
    class_names = base.classes    # stable alphabetical order

    print(f"  Total crops      : {n_total}")
    print(f"  Unique classes   : {len(class_names)}")
    print(f"  Train / Val      : {n_train} / {n_val}")

    # Two separate dataset objects (different transforms)
    train_ds = Subset(datasets.ImageFolder(str(CROPS_DIR), transform=train_tf), train_idx)
    val_ds   = Subset(datasets.ImageFolder(str(CROPS_DIR), transform=val_tf),   val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH, shuffle=True,
        num_workers=WORKERS, pin_memory=bool(GPU_IDS)
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH, shuffle=False,
        num_workers=WORKERS, pin_memory=bool(GPU_IDS)
    )
    return train_loader, val_loader, class_names


def build_model(n_classes: int) -> nn.Module:
    """EfficientNet-B0 with its head replaced for our class count."""
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model   = models.efficientnet_b0(weights=weights)
    in_feat = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feat, n_classes)
    return model


def train() -> None:
    _ensure_dirs()

    # ── Device setup ──────────────────────────────────────────────
    if GPU_IDS and torch.cuda.is_available():
        device     = torch.device(f"cuda:{GPU_IDS[0]}")   # primary GPU
        use_multi  = len(GPU_IDS) > 1
    else:
        device    = torch.device("cpu")
        use_multi = False

    n_gpus = len(GPU_IDS) if use_multi else (1 if device.type == "cuda" else 0)
    print(f"  Device : {device}  |  DataParallel GPUs: {GPU_IDS if use_multi else 'disabled'}")
    # Scale effective batch size linearly with GPU count (standard practice)
    effective_batch = BATCH * max(n_gpus, 1)
    print(f"  Batch  : {BATCH} per GPU × {max(n_gpus,1)} GPUs = {effective_batch} effective\n")

    train_loader, val_loader, class_names = build_dataloaders()
    n_classes    = len(class_names)
    base_model   = build_model(n_classes)

    # ── Resume from last checkpoint (load weights BEFORE DataParallel) ──
    start_epoch  = 0
    best_val_acc = 0.0
    optimizer    = torch.optim.Adam(base_model.parameters(), lr=LR)
    scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion    = nn.CrossEntropyLoss()

    if LAST_CKPT.exists():
        print(f"Resuming Stage 2 training from {LAST_CKPT}")
        ckpt = torch.load(LAST_CKPT, map_location="cpu")
        base_model.load_state_dict(ckpt["model"])   # always saved as plain module
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"  Resuming from epoch {start_epoch} | best val acc: {best_val_acc:.2%}\n")
    else:
        print("Starting Stage 2 — EfficientNet-B0 Classifier\n")

    # ── Wrap in DataParallel AFTER loading weights ─────────────────
    base_model = base_model.to(device)
    model: nn.Module = nn.DataParallel(base_model, device_ids=GPU_IDS) if use_multi else base_model

    # Save class list now (needed for inference even before training ends)
    (CKPT_DIR / "classes.txt").write_text("\n".join(class_names), encoding="utf-8")

    # ── Training loop ─────────────────────────────────────────────
    for epoch in range(start_epoch, EPOCHS):
        t0 = time.time()

        # Train
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

        # Validate
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
            f"  Epoch {epoch+1:>3}/{EPOCHS}"
            f"  loss {train_loss/train_total:.4f}"
            f"  train {train_acc:.2%}"
            f"  val {val_acc:.2%}"
            f"  ({elapsed:.1f}s)"
        )

        # Always save the plain module (unwrap DataParallel) so the
        # checkpoint is portable and resume works regardless of GPU count.
        raw_state = (model.module if use_multi else model).state_dict()

        torch.save({
            "epoch":        epoch,
            "model":        raw_state,
            "optimizer":    optimizer.state_dict(),
            "scheduler":    scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "classes":      class_names,
        }, LAST_CKPT)

        # Save best checkpoint separately
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":   epoch,
                "model":   raw_state,
                "classes": class_names,
            }, BEST_CKPT)
            print(f"  ✅  New best val acc: {best_val_acc:.2%} — saved best.pt")

    print(f"\nStage 2 complete.")
    print(f"Best val accuracy : {best_val_acc:.2%}")
    print(f"Best weights      → {BEST_CKPT}")
    print(f"Class mapping     → {CKPT_DIR / 'classes.txt'}")


if __name__ == "__main__":
    train()