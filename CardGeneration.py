import random
import json
import math
from PIL import Image
from pathlib import Path

# Use pathlib for clean, OS-agnostic path management
MAIN_DIR = Path.cwd()

# --- CONFIGURATION ---
BG_DIR = MAIN_DIR / "events"
CARD_DIR = MAIN_DIR / "1024Portraits"
OUTPUT_DIR = MAIN_DIR / "output"
IMG_OUT = OUTPUT_DIR / "images"

# Create output dirs if they don't exist
IMG_OUT.mkdir(parents=True, exist_ok=True)

# Crawl data and store both the name and the full path
card_data = [] 
allowed_folders = ['colorless', 'curse', 'green']

for folder in allowed_folders:
    folder_path = CARD_DIR / folder
    if folder_path.exists():
        for file_path in folder_path.rglob('*'):
            if file_path.suffix.lower() in ['.png', '.jpg']:
                card_data.append((file_path.name, file_path.as_posix()))

class_map = {name: idx for idx, (name, path) in enumerate(card_data)}
print(f"Successfully loaded {len(card_data)} cards.")

# Generation settings
NUM_IMAGES_TO_GENERATE = 1000
MIN_CARDS = 1
MAX_CARDS = 10
CARD_SCALE_FACTOR = 0.3 

def generate_dataset():
    bg_files = [f for f in BG_DIR.glob('*') if f.suffix.lower() in ['.png', '.jpg']]
    
    if not bg_files:
        print("Error: No backgrounds found!")
        return

    # Initialize COCO Dictionary
    coco_data = {
        "info": {"description": "Slay the Spire Synthetic Hands - Native COCO"},
        "images": [],
        "annotations": [],
        "categories": [{"id": idx, "name": name, "supercategory": "card"} for name, idx in class_map.items()]
    }
    
    annotation_id = 0
        
    for i in range(NUM_IMAGES_TO_GENERATE):
        bg_path = random.choice(bg_files)
        canvas = Image.open(bg_path).convert("RGBA")
        canvas_w, canvas_h = canvas.size
        
        num_cards = random.randint(MIN_CARDS, MAX_CARDS)
        hand = random.choices(card_data, k=num_cards)
        
        # Register image in COCO
        image_filename = f"synthetic_hand_{i}.jpg"
        coco_data["images"].append({
            "id": i,
            "file_name": image_filename,
            "width": canvas_w,
            "height": canvas_h
        })
        
        # 1. Dynamic Spacing & Overlap Logic
        sample_card = Image.open(hand[0][1])
        base_w = int(sample_card.width * CARD_SCALE_FACTOR)
        base_h = int(sample_card.height * CARD_SCALE_FACTOR)
        
        overlap_factor = random.uniform(0.85, 1.0) 
        step_x = base_w * overlap_factor
        total_hand_width = base_w + (step_x * (num_cards - 1))
        
        # If the hand is wider than 95% of the screen, force it to squeeze tighter
        max_allowed_width = canvas_w * 0.95
        new_scale = CARD_SCALE_FACTOR
        while total_hand_width > max_allowed_width:
            new_scale -= 0.05
            base_w = int(sample_card.width * new_scale)
            base_h = int(sample_card.height * new_scale)
            step_x = step_x = base_w * overlap_factor
            total_hand_width = base_w + (step_x * (num_cards - 1))
            
        start_center_x = (canvas_w / 2) - (total_hand_width / 2) + (base_w / 2)
        
        # Calculate maximum possible height after a 20-degree rotation
        rad_20 = math.radians(20)
        max_rot_h = (base_w * math.sin(rad_20)) + (base_h * math.cos(rad_20))
        max_drop_y = (((num_cards - 1) / 2) ** 2) * (base_h * 0.03)
        
        # Anchor the Y position so the lowest possible corner hits exactly 98% down the screen
        base_center_y = (canvas_h * 0.98) - max_drop_y - (max_rot_h / 2)
        
        # 2. Rotation & Placement
        for j, (card_name, card_path) in enumerate(hand):
            card_img = Image.open(card_path).convert("RGBA")
            card_img = card_img.resize((base_w, base_h), Image.Resampling.LANCZOS)
            
            if num_cards == 1:
                angle = 0
            else:
                angle = -20 * ((j - num_cards/2)/num_cards)
                
            rotated_card = card_img.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
            rot_w, rot_h = rotated_card.size
            
            distance_from_center = abs(j - ((num_cards - 1) / 2))
            arc_drop_y = (distance_from_center ** 2) * (base_h * 0.03)
            
            card_center_x = start_center_x + (j * step_x)
            card_center_y = base_center_y + arc_drop_y
            
            # Top-left coordinates for pasting and COCO bounding boxes
            paste_x = int(card_center_x - (rot_w / 2))
            paste_y = int(card_center_y - (rot_h / 2))
            
            canvas.alpha_composite(rotated_card, (paste_x, paste_y))
            
            # 3. Native COCO Bounding Box Extraction
            # COCO format requires [top_left_x, top_left_y, width, height] in absolute pixels
            bbox = [paste_x, paste_y, rot_w, rot_h]
            area = rot_w * rot_h
            
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": i,
                "category_id": class_map[card_name],
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                "segmentation": []
            })
            annotation_id += 1
        
        # 4. Save Image
        canvas.convert("RGB").save(IMG_OUT / image_filename)
        if i%100 == 0:
            print(f"Created {i*100/NUM_IMAGES_TO_GENERATE}% images")
            
    # 5. Save the master COCO JSON file
    json_output_path = OUTPUT_DIR / "dataset.coco.json"
    with open(json_output_path, "w") as f:
        json.dump(coco_data, f, indent=4)
        
    print(f"Successfully generated {NUM_IMAGES_TO_GENERATE} perfectly arced images.")
    print(f"Saved native COCO annotations to {json_output_path}")

if __name__ == "__main__":
    generate_dataset()