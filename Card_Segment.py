from ultralytics import YOLO
from pathlib import Path

# Paths remain untouched
MAIN_DIR = Path.cwd()
YAML_PATH = MAIN_DIR / "output" / "data.yaml"
RUN_NAME = "card_segment"
LAST_WEIGHTS = MAIN_DIR / "runs" / "segment" / RUN_NAME / "weights" / "last.pt"

def train_finder_model():
    if LAST_WEIGHTS.exists():
        print(f"Resuming training from the last saved epoch at {LAST_WEIGHTS}")
        model = YOLO(str(LAST_WEIGHTS))
        results = model.train(resume=True)
    else:
        print("No previous run found. Training Stage 1 Finder.")
        # yolov8s-seg is perfect for finding a single class incredibly fast
        model = YOLO("yolov8s-seg.pt")

        results = model.train(
            data=str(YAML_PATH),
            epochs=100,
            imgsz=640,
            batch=128,
            workers=8,
            device=[0,1],
            name=RUN_NAME,
            exist_ok=True,
            single_cls=True
        )

    print("Finder training complete")
    print(f"Best model weights are saved in 'runs/segment/{RUN_NAME}/weights/best.pt'")

if __name__ == "__main__":
    train_finder_model()