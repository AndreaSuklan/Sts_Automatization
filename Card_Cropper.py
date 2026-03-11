import os
import yaml
from PIL import Image
from pathlib import Path

# Paths
MAIN_DIR = Path.cwd()
OUTPUT_DIR = MAIN_DIR / "output"
IMG_DIR = OUTPUT_DIR / "images"
LABEL_DIR = OUTPUT_DIR / "labels"
YAML_PATH = OUTPUT_DIR / "data.yaml"

# New output for the classification dataset
CLS_DATA_DIR = OUTPUT_DIR / "cropped_cards" / "train"

def crop_for_classification():
    CLS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(YAML_PATH, 'r') as f:
        data_yaml = yaml.safe_load(f)
    classes = data_yaml.get('names', {})

    for cls_id, cls_name in classes.items():
        (CLS_DATA_DIR / cls_name).mkdir(parents=True, exist_ok=True)

    print("Cropping dataset to prepare for classification. This may take a moment...")
    
    for label_file in LABEL_DIR.glob('*.txt'):
        img_file = IMG_DIR / (label_file.stem + ".jpg")
        
        if not img_file.exists():
            continue
            
        try:
            img = Image.open(img_file)
            img_w, img_h = img.size
        except Exception:
            continue
            
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 3:
                continue
                
            cls_id = int(parts[0])
            cls_name = classes[cls_id]
            
            # Convert normalized polygon points to absolute bounding box
            coords = [float(x) for x in parts[1:]]
            x_coords = coords[0::2]
            y_coords = coords[1::2]
            
            min_x = int(min(x_coords) * img_w)
            max_x = int(max(x_coords) * img_w)
            min_y = int(min(y_coords) * img_h)
            max_y = int(max(y_coords) * img_h)
            
            # Add a small 2-pixel pad to avoid tight crop errors
            min_x, min_y = max(0, min_x - 2), max(0, min_y - 2)
            max_x, max_y = min(img_w, max_x + 2), min(img_h, max_y + 2)
            
            # Crop and save to the specific class folder
            cropped_img = img.crop((min_x, min_y, max_x, max_y))
            save_path = CLS_DATA_DIR / cls_name / f"{label_file.stem}_crop_{i}.jpg"
            cropped_img.save(save_path)

    print(f"Cropping complete! Classification data ready in: {CLS_DATA_DIR}")

if __name__ == "__main__":
    crop_for_classification()