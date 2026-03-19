"""
Object_Detection.py  -  Stage 1: Generic Object Detector
═════════════════════════════════════════════════════════
Trains YOLOv8s to locate and classify 8 generic object types
in Slay the Spire combat screenshots.

  0 Card  1 Enemy  2 HealthBar  3 Intent
  4 Player  5 Potion  6 Power  7 Relic

PRE-REQUISITE
  Run Yolo_Crop.py first.  It builds output/stage1_dataset/ with
  hard-linked images (zero extra disk space), remapped labels,
  and data.yaml.  This script just calls YOLO.train() on that.

RESUME
  If runs/detect/<RUN_NAME>/weights/last.pt exists, training
  resumes automatically.  Delete it to start fresh.

CLI ARGUMENTS (all optional - defaults match original CONFIG)
  --epochs  INT         number of training epochs        (100)
  --batch   INT         batch size                       (64)
  --device  STR         GPU ids or 'cpu'  e.g. 0,1      (0,1)
  --imgsz   INT         input image size                 (640)
"""

import argparse
from pathlib import Path
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

      'cpu'   -> 'cpu'
      '0'     -> [0]      (single GPU as list so YOLO uses DataParallel path)
      '0,1'   -> [0, 1]
    """
    if value.strip().lower() == "cpu":
        return "cpu"
    try:
        ids = [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid --device value: {value!r}. "
            "Use 'cpu', '0' (single GPU), or '0,1' (multi-GPU)."
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
        "--device", type=str, default="0,1",
        help="Device(s) to train on: 'cpu', '0' (single GPU), or '0,1' (multi-GPU)",
    )
    ap.add_argument(
        "--imgsz", type=int, default=640,
        help="Input image size (pixels, square)",
    )
    return ap.parse_args()


def _ensure_dirs() -> None:
    """Create all directories this script may write to."""
    Path("logs").mkdir(parents=True, exist_ok=True)
    (Path.cwd() / "runs" / "detect").mkdir(parents=True, exist_ok=True)


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
        model.train(resume=True)

    else:
        if not YAML_PATH.exists():
            raise FileNotFoundError(
                f"data.yaml not found at {YAML_PATH}\n"
                "Run Yolo_Crop.py first to build the Stage 1 dataset."
            )

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