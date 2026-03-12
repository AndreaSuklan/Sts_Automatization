from ultralytics import YOLO
from pathlib import Path

MAIN_DIR = Path.cwd()
CLS_DATA_DIR = MAIN_DIR / "output" / "cropped_cards"
RUN_NAME = "card_classifier"

def train_classification_model():
    print("Training Classifier...")

    model = YOLO("yolov8m-cls.pt")

    results = model.train(
        data=str(CLS_DATA_DIR),
        epochs=50,
        imgsz=224,
        batch=128,
        workers=8,
        device=[0,1],
        name=RUN_NAME,
        exist_ok=True
    )

    print("Classifier training complete")
    print(f"Best model weights are saved in 'runs/classify/{RUN_NAME}/weights/best.pt'")

if __name__ == "__main__":
    train_classification_model()
