import random
import os
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
LABEL_OUT = OUTPUT_DIR / "labels"

DEBUG_BBOX = False 

# Create output dirs if they don't exist
IMG_OUT.mkdir(parents=True, exist_ok=True)
LABEL_OUT.mkdir(parents=True, exist_ok=True) 

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
        
    for i in range(NUM_IMAGES_TO_GENERATE):
        bg_path = random.choice(bg_files)
        canvas = Image.open(bg_path).convert("RGBA")
        canvas_w, canvas_h = canvas.size
        
        # Define the crop area for the bottom 30%
        crop_y_start = int(canvas_h * 0.7)
        crop_h = canvas_h - crop_y_start
        
        num_cards = random.randint(MIN_CARDS, MAX_CARDS)
        hand = random.choices(card_data, k=num_cards)
        
        image_filename = f"synthetic_hand_{i}.jpg"
        label_filename = f"synthetic_hand_{i}.txt" 
        label_filepath = LABEL_OUT / label_filename
        
        # Clear out the label file if it already exists from a previous run
        if label_filepath.exists():
            label_filepath.unlink()
        
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
            cx = paste_x + (rot_w / 2)
            cy = paste_y + (rot_h / 2)
            
            hw, hh = base_w / 2, base_h / 2
            corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
            
            rad = math.radians(-angle)
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)
            
            poly_points = []
            norm_seg_points = [] # CHANGED: List for normalized YOLO coords
            
            for x, y in corners:
                # Absolute coordinates on the full canvas
                rx = cx + (x * cos_a - y * sin_a)
                ry = cy + (x * sin_a + y * cos_a)
                poly_points.append((rx, ry))
                
                # CHANGED: Calculate normalized coordinates for cropped image
                shifted_y = ry - crop_y_start
                norm_x = rx / canvas_w
                norm_y = shifted_y / crop_h
                
                # Clamp coordinates between 0.0 and 1.0 to prevent YOLO crash on edge clipping
                norm_x = max(0.0, min(1.0, norm_x))
                norm_y = max(0.0, min(1.0, norm_y))
                
                # Append formatted string to the list
                norm_seg_points.extend([f"{norm_x:.6f}", f"{norm_y:.6f}"])

            if DEBUG_BBOX:
                draw = ImageDraw.Draw(canvas)
                draw.polygon(poly_points, outline="lime", width=3)
            
            # 3. CHANGED: Write the instance segmentation format to .txt
            with open(label_filepath, "a") as f:
                f.write(f"{class_map[card_name]} {' '.join(norm_seg_points)}\n")
        
        # 4. Crop canvas to bottom 30% before saving
        final_image = canvas.crop((0, crop_y_start, canvas_w, canvas_h))
        final_image.convert("RGB").save(IMG_OUT / image_filename)
        
        if i % 100 == 0 and i > 0:
            print(f"Created {int(i*100/NUM_IMAGES_TO_GENERATE)}% images")
            
    # 5. Auto-generate the data.yaml file for YOLO training
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {OUTPUT_DIR.resolve()}\n")
        f.write("train: images\n")
        f.write("val: images\n\n")
        f.write("names:\n")
        sorted_classes = sorted(class_map.items(), key=lambda item: item[1])
        for name, idx in sorted_classes:
            f.write(f"  {idx}: {name}\n")
            
    print(f"Successfully generated {NUM_IMAGES_TO_GENERATE} YOLO segmentation images and labels.")
    print(f"Saved YOLO configuration to {yaml_path}")

if __name__ == "__main__":
    generate_dataset()