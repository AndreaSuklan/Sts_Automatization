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
        epochs=100,        # Start with a low number just to verify the pipeline works
        rect=True,
	imgsz=608,        # Standard input resolution for YOLO
        batch=16,          # Low batch size to prevent memory crashes on your CPU/GPU
        workers=4,
	device="0"      # Change this to "0" if you have an NVIDIA GPU configured for PyTorch
    )
    
    print("Training complete!")
    print("Your best model weights are saved in 'runs/segment/train/weights/best.pt'")

if __name__ == "__main__":
    train_segmentation_model()
