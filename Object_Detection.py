"""
Object_Detection.py  -  Stage 1: Generic Object Detector
═════════════════════════════════════════════════════════════════════
Trains YOLOv8s to locate and classify 16 generic object types
in Slay the Spire combat screenshots.

PRE-REQUISITE
  Run Yolo_Crop.py first.  It builds output/stage1_dataset/ with
  hard-linked images (zero extra disk space), remapped labels,
  and data.yaml.  This script just calls YOLO.train() on that.

RESUME
  If runs/detect/<RUN_NAME>/weights/last.pt exists, training
  resumes automatically.  Delete it to start fresh.

CLI ARGUMENTS (all optional)
  --epochs  INT         number of training epochs        (100)
  --batch   INT         batch size                       (64)
  --device  STR         GPU ids, 'cpu', or 'auto'        (auto)
  --imgsz   INT         input image size                 (640)
"""

import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

# ─────────────────────── FIXED CONFIG ───────────────────────── #

YAML_PATH    = Path.cwd() / "output" / "stage1_dataset" / "data.yaml"
RUN_NAME     = "sts_detector"
LAST_WEIGHTS = Path.cwd() / "runs" / "detect" / RUN_NAME / "weights" / "last.pt"

WORKERS          = 8
DATA_PERCENTAGE  = 1

# ────────────────────────────────────────────────────────────── #


def _parse_device(value: str):
    """
    Parse a device string into the format YOLO expects.
      'auto'  -> dynamically detects all available GPUs, or falls back to 'cpu'
      'cpu'   -> 'cpu'
      '0'     -> [0]
      '0,1'   -> [0, 1]
    """
    val_lower = value.strip().lower()
    
    if val_lower == "auto":
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            return [i for i in range(count)]
        return "cpu"
        
    if val_lower == "cpu":
        return "cpu"
        
    try:
        ids = [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid --device value: {value!r}. "
            "Use 'auto', 'cpu', '0' (single GPU), or '0,1' (multi-GPU)."
        )
    if not ids:
        raise argparse.ArgumentTypeError("--device cannot be empty.")
    return ids


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Stage 1 - YOLOv8s Generic Object Detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs",
    )
    ap.add_argument(
        "--batch", type=int, default=64,
        help="Batch size (split evenly across GPUs when using multi-GPU)",
    )
    ap.add_argument(
        "--device", type=str, default="auto",
        help="Device(s) to train on: 'auto', 'cpu', '0', or '0,1'",
    )
    ap.add_argument(
        "--imgsz", type=int, default=640,
        help="Input image size (pixels, square)",
    )
    return ap.parse_args()


def _ensure_dirs() -> None:
    """Create all directories this script may write to or read from."""
    Path("logs").mkdir(parents=True, exist_ok=True)
    (Path.cwd() / "runs" / "detect").mkdir(parents=True, exist_ok=True)
    LAST_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    YAML_PATH.parent.mkdir(parents=True, exist_ok=True)


def train() -> None:
    args   = _parse_args()
    device = _parse_device(args.device)

    _ensure_dirs()

    print("=" * 60)
    print("  Stage 1 - Generic Object Detector (YOLOv8s)")
    print("=" * 60)
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch}")
    print(f"  Device  : {device}")
    print(f"  Img size: {args.imgsz}")
    print("=" * 60)

    if LAST_WEIGHTS.exists():
        print(f"Resuming Stage 1 training from {LAST_WEIGHTS}")
        model = YOLO(str(LAST_WEIGHTS))
        model.train(resume=True, device=device, data=str(YAML_PATH))    
    else:
        if not YAML_PATH.exists():
            print(f"Warning: data.yaml not found at {YAML_PATH}. Ensure Yolo_Crop.py ran successfully.")
            
        print("Starting Stage 1 - Generic Object Detector (YOLOv8s)")
        model = YOLO("yolov8s.pt")
        model.train(
            data     = str(YAML_PATH),
            epochs   = args.epochs,
            imgsz    = args.imgsz,
            batch    = args.batch,
            workers  = WORKERS,
            device   = device,
            name     = RUN_NAME,
            exist_ok = True,
            fraction = DATA_PERCENTAGE,
        )

    print(f"\nStage 1 complete.")
    print(f"Best weights -> runs/detect/{RUN_NAME}/weights/best.pt")


if __name__ == "__main__":
    train()