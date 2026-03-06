from ultralytics import YOLO
from pathlib import Path

# Set up paths
MAIN_DIR = Path.cwd()
YAML_PATH = MAIN_DIR / "output" / "data.yaml"

def train_segmentation_model():
    # The 'n' stands for nano, which is the fastest and lightest model.
    model = YOLO("yolov8n-seg.pt") 
    
    print("Starting YOLO segmentation training...")
    
    # Start the training loop
    results = model.train(
        data=str(YAML_PATH),
        epochs=400,
        rect=True,
	    imgsz=608,
        batch=-1,
        workers=8,
	    device="0"
    )
    
    print("Training complete!")
    print("Your best model weights are saved in 'runs/segment/train/weights/best.pt'")

if __name__ == "__main__":
    train_segmentation_model()
