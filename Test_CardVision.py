from ultralytics import YOLO
from pathlib import Path

MAIN_DIR = Path.cwd()
RUN_NAME = "card_segment"

WEIGHTS_PATH = MAIN_DIR / "runs" / "segment" / RUN_NAME / "weights" / "best.pt"

SCREENSHOT_NAME = "Screenshot_3.png" 
IMAGE_PATH = MAIN_DIR / "Screenshots" / SCREENSHOT_NAME

def test_on_screenshot():
    if not WEIGHTS_PATH.exists():
        print(f"Error: Could not find weights at {WEIGHTS_PATH}")
        return
    if not IMAGE_PATH.exists():
        print(f"Error: Could not find image at {IMAGE_PATH}")
        return

    print(f"Loading custom model from {WEIGHTS_PATH}...")
    model = YOLO(str(WEIGHTS_PATH))

    print(f"Running prediction on {IMAGE_PATH}...")
    # 25% of confidence
    results = model(str(IMAGE_PATH), conf=0.25) 

    for result in results:
        
        # Save the annotated image
        save_path = MAIN_DIR / "Screenshots" / f"predicted_{SCREENSHOT_NAME}.jpg"
        result.save(filename=str(save_path))
        print(f"Saved the annotated result to '{save_path}'")

if __name__ == "__main__":
    test_on_screenshot()