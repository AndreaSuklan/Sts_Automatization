from ultralytics import YOLO
from pathlib import Path

# Paths
MAIN_DIR = Path.cwd()
YAML_PATH = MAIN_DIR / "output" / "data.yaml"
RUN_NAME = "hand_vision"
LAST_WEIGHTS = MAIN_DIR / "runs" / "segment" / RUN_NAME / "weights" / "last.pt"

def train_segmentation_model():

    if LAST_WEIGHTS.exists():
        print("Resuming training from the last saved epoch at {LAST_WEIGHTS}")

        # Load last checkpoint
        model = YOLO(str(LAST_WEIGHTS))

        # Resume training
        results = model.train(resume=True)

    else:
        print("No previous run found")

        model = YOLO("yolov8n-seg.pt")

        results = model.train(
            data=str(YAML_PATH),
            epochs=400,
            imgsz=608,
            batch=128,
            workers=8,
            device=[0,1],
            name=RUN_NAME,
            exist_ok=True
        )

    print("Training complete")
    print(f"Best model weights are saved in '{RUN_NAME}/weights/best.pt'")

if __name__ == "__main__":
    train_segmentation_model()
