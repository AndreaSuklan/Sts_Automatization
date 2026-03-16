"""
train_stage1_detector.py  —  Stage 1: Generic Object Detector
═══════════════════════════════════════════════════════════════
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
"""

from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────── CONFIG ─────────────────────────────── #

YAML_PATH    = Path.cwd() / "output" / "stage1_dataset" / "data.yaml"
RUN_NAME     = "sts_detector"
LAST_WEIGHTS = Path.cwd() / "runs" / "detect" / RUN_NAME / "weights" / "last.pt"

EPOCHS  = 1
IMGSZ   = 640
BATCH   = 16
WORKERS = 8
DEVICE  = "cpu"   # list of GPU ids; use 0 for single GPU, "cpu" for CPU
DATA_PERCENTAGE = 0.01   # for debugging

# ────────────────────────────────────────────────────────────────── #

def train() -> None:
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

        print("Starting Stage 1 — Generic Object Detector (YOLOv8s)")
        model = YOLO("yolov8s.pt")
        model.train(
            data     = str(YAML_PATH),
            epochs   = EPOCHS,
            imgsz    = IMGSZ,
            batch    = BATCH,
            workers  = WORKERS,
            device   = DEVICE,
            name     = RUN_NAME,
            exist_ok = True,
            fraction = DATA_PERCENTAGE,
        )

    print(f"\nStage 1 complete.")
    print(f"Best weights → runs/detect/{RUN_NAME}/weights/best.pt")


if __name__ == "__main__":
    train()