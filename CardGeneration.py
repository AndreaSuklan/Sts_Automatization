import random
import json
import math
from PIL import Image, ImageDraw
from pathlib import Path

# Use pathlib for clean, OS-agnostic path management
MAIN_DIR = Path.cwd()

# --- CONFIGURATION ---
BG_DIR = MAIN_DIR / "events"
CARD_DIR = MAIN_DIR / "1024Portraits"
OUTPUT_DIR = MAIN_DIR / "output"
IMG_OUT = OUTPUT_DIR / "images"

DEBUG_BBOX = False 

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
NUM_IMAGES_TO_GENERATE = 10
MIN_CARDS = 1
MAX_CARDS = 10
CARD_SCALE_FACTOR = 0.3 

def generate_dataset():
    bg_files = [f for f in BG_DIR.glob('*') if f.suffix.lower() in ['.png', '.jpg']]
    
    if not bg_files:
        print("Error: No backgrounds found!")
        return

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
        
        # Define the crop area for the bottom 30%
        crop_y_start = int(canvas_h * 0.7)
        crop_h = canvas_h - crop_y_start
        
        num_cards = random.randint(MIN_CARDS, MAX_CARDS)
        hand = random.choices(card_data, k=num_cards)
        
        # Register the cropped image dimensions in COCO
        image_filename = f"synthetic_hand_{i}.jpg"
        coco_data["images"].append({
            "id": i,
            "file_name": image_filename,
            "width": canvas_w,
            "height": crop_h # CHANGED: Uses cropped height
        })
        
        # 1. Dynamic Spacing & Overlap Logic
        sample_card = Image.open(hand[0][1])
        base_w = int(sample_card.width * CARD_SCALE_FACTOR)
        base_h = int(sample_card.height * CARD_SCALE_FACTOR)
        
        overlap_factor = random.uniform(0.85, 1.0) 
        step_x = base_w * overlap_factor
        total_hand_width = base_w + (step_x * (num_cards - 1))
        
        max_allowed_width = canvas_w * 0.95
        new_scale = CARD_SCALE_FACTOR

        while total_hand_width > max_allowed_width:
            new_scale -= 0.01
            base_w = int(sample_card.width * new_scale)
            base_h = int(sample_card.height * new_scale)
            step_x = base_w * overlap_factor
            total_hand_width = base_w + (step_x * (num_cards - 1))
            
        start_center_x = (canvas_w / 2) - (total_hand_width / 2) + (base_w / 2)
        
        rad_20 = math.radians(20)
        max_rot_h = (base_w * math.sin(rad_20)) + (base_h * math.cos(rad_20))
        max_drop_y = (((num_cards - 1) / 2) ** 2) * (base_h * 0.03)
        
        base_center_y = (canvas_h * 0.98) - max_drop_y - (max_rot_h / 2)
        
        # 2. Rotation & Placement
        for j, (card_name, card_path) in enumerate(hand):
            card_img = Image.open(card_path).convert("RGBA")
            card_img = card_img.resize((base_w, base_h), Image.Resampling.LANCZOS)
            
            if num_cards == 1:
                angle = 0
            else:
                angle = -20 * ((j - (num_cards-1)/2)/(num_cards-1)/2)
                
            rotated_card = card_img.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
            rot_w, rot_h = rotated_card.size
            
            distance_from_center = abs(j - ((num_cards - 1) / 2))
            arc_drop_y = (distance_from_center ** 2) * (base_h * 0.03)
            
            card_center_x = start_center_x + (j * step_x)
            card_center_y = base_center_y + arc_drop_y
            
            paste_x = int(card_center_x - (rot_w / 2))
            paste_y = int(card_center_y - (rot_h / 2))
            
            canvas.alpha_composite(rotated_card, (paste_x, paste_y))
            
            # Polygon Math for tight corners
            # Get the absolute center of the pasted rotated image
            cx = paste_x + (rot_w / 2)
            cy = paste_y + (rot_h / 2)
            
            # Unrotated corner coordinates relative to center
            hw, hh = base_w / 2, base_h / 2
            corners = [
                (-hw, -hh), # Top-Left
                (hw, -hh),  # Top-Right
                (hw, hh),   # Bottom-Right
                (-hw, hh)   # Bottom-Left
            ]
            
            # Rotate corners using standard rotation matrix
            # (PIL uses CCW rotation, requiring a negative angle for standard screen-space math)
            rad = math.radians(-angle)
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)
            
            poly_points = []
            poly_points_flat_cropped = []
            
            for x, y in corners:
                rx = cx + (x * cos_a - y * sin_a)
                ry = cy + (x * sin_a + y * cos_a)
                poly_points.append((rx, ry))
                
                # Shift Y coordinates for the cropped COCO annotation
                poly_points_flat_cropped.extend([rx, ry - crop_y_start])

            if DEBUG_BBOX:
                draw = ImageDraw.Draw(canvas)
                # Draw the tight polygon
                draw.polygon(poly_points, outline="lime", width=3)
            
            # 4. Shift BBox Y-coordinate for cropped COCO annotation
            bbox_cropped = [paste_x, paste_y - crop_y_start, rot_w, rot_h]
            area = rot_w * rot_h
            
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": i,
                "category_id": class_map[card_name],
                "bbox": bbox_cropped,
                "area": area,
                "iscrowd": 0,
                "segmentation": [poly_points_flat_cropped] # Added polygon data
            })
            annotation_id += 1
        
        # 5. CHANGED: Crop canvas to bottom 30% before saving
        final_image = canvas.crop((0, crop_y_start, canvas_w, canvas_h))
        final_image.convert("RGB").save(IMG_OUT / image_filename)
        
        if i % 100 == 0 and i > 0:
            print(f"Created {int(i*100/NUM_IMAGES_TO_GENERATE)}% images")
            
    json_output_path = OUTPUT_DIR / "dataset.coco.json"
    with open(json_output_path, "w") as f:
        json.dump(coco_data, f, indent=4)
        
    print(f"Successfully generated {NUM_IMAGES_TO_GENERATE} cropped images.")
    print(f"Saved native COCO annotations to {json_output_path}")

if __name__ == "__main__":
    generate_dataset()