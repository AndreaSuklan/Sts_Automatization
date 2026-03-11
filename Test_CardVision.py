from ultralytics import YOLO
from pathlib import Path

MAIN_DIR = Path.cwd()
RUN_NAME = "hand_vision"

WEIGHTS_PATH = MAIN_DIR / "runs" / "segment" / RUN_NAME / "weights" / "best.pt"

SCREENSHOT_NAME = "screenshot_2.jpg" 
IMAGE_PATH = MAIN_DIR / "Screenshots" / SCREENSHOT_NAME

def test_on_screenshot():
    if not WEIGHTS_PATH.exists():
        print(f"Error: Could not find weights at {WEIGHTS_PATH}")
        return
    if not IMAGE_PATH.exists():
        print(f"Error: Could not find image at {IMAGE_PATH}")
        return

    # --- 2. Load your custom trained model ---
    print(f"Loading custom model from {WEIGHTS_PATH}...")
    model = YOLO(str(WEIGHTS_PATH))

    # --- 3. Run inference (prediction) ---
    print(f"Running prediction on {IMAGE_PATH}...")
    # conf=0.25 filters out weak detections; you can tweak this value
    results = model(str(IMAGE_PATH), conf=0.25) 

    # --- 4. Process and view the results ---
    for result in results:
        # Display the image with boxes and masks on your screen
        result.show() 
        
        # Save the annotated image to your disk
        save_path = MAIN_DIR / "Screenshots" / f"predicted_{SCREENSHOT_NAME}.jpg"
        result.save(filename=str(save_path))
        print(f"Saved the annotated result to '{save_path}'")

if __name__ == "__main__":
    test_on_screenshot()