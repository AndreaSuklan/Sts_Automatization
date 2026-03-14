from ultralytics import YOLO
from pathlib import Path

MAIN_DIR = Path.cwd()
RUN_NAME = "card_segment"

WEIGHTS_PATH = MAIN_DIR / "runs" / "segment" / RUN_NAME / "weights" / "best.pt"
IMAGE_PATH = MAIN_DIR / "Screenshots"

# Define the save directory explicitly
SAVE_DIR = IMAGE_PATH / "predictions"

def test_on_folder():
    if not WEIGHTS_PATH.exists():
        print(f"Error: Could not find weights at {WEIGHTS_PATH}")
        return
    if not IMAGE_PATH.exists():
        print(f"Error: Could not find image folder at {IMAGE_PATH}")
        return

    # 1. Ensure the output directory exists before saving!
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading custom model from {WEIGHTS_PATH}...")
    model = YOLO(str(WEIGHTS_PATH))

    # 2. Safely grab both .png and .jpg files
    valid_extensions = {".png", ".jpg", ".jpeg"}
    images = [p for p in IMAGE_PATH.iterdir() if p.is_file() and p.suffix.lower() in valid_extensions]

    if not images:
        print(f"No images found in {IMAGE_PATH}")
        return

    print(f"Found {len(images)} images. Starting predictions...")

    for im in images:
        # verbose=False stops YOLO from spamming the console for every single image
        results = model(str(im), conf=0.25, verbose=False) 

        for result in results:
            # Save the annotated image
            save_path = SAVE_DIR / f"predicted_{im.name}"
            result.save(filename=str(save_path))
            print(f"Saved annotated result to: {save_path.name}")

if __name__ == "__main__":
    test_on_folder()