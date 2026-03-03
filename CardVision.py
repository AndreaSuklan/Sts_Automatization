import torch
import json
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import DetrImageProcessor, DetrForObjectDetection, TrainingArguments, Trainer
import albumentations as A
from PIL import Image
import numpy as np

# --- CONFIGURATION ---
MAIN_DIR = Path.cwd()
OUTPUT_DIR = MAIN_DIR / "output"
IMG_DIR = OUTPUT_DIR / "images"
ANNOTATION_FILE = OUTPUT_DIR / "dataset.coco.json"

MODEL_NAME = "facebook/detr-resnet-50"
BATCH_SIZE = 1
EPOCHS = 0.1 # Start small, increase if needed

# 1. Load the COCO Data
with open(ANNOTATION_FILE, 'r') as f:
    coco_data = json.load(f)

# Create lookup dictionaries required by DETR
id2label = {item['id']: item['name'] for item in coco_data['categories']}
label2id = {item['name']: item['id'] for item in coco_data['categories']}

# Structure the data for easy access during training
images_dict = {img['id']: img for img in coco_data['images']}
annotations_dict = {}
for ann in coco_data['annotations']:
    img_id = ann['image_id']
    if img_id not in annotations_dict:
        annotations_dict[img_id] = []
    annotations_dict[img_id].append(ann)

image_ids = list(images_dict.keys())

# --- DATASET SPLIT ---
# Simple 80/20 split
split_idx = int(len(image_ids) * 0.8)
train_ids = image_ids[:split_idx]
val_ids = image_ids[split_idx:]

# 2. Image Augmentation (Optional but recommended)
# This adds slight noise/brightness changes to make the model tough
transform = A.Compose([
    A.RandomBrightnessContrast(p=0.2),
    A.GaussNoise(p=0.1)
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

# 3. The Dataset Class
class StSDataset(torch.utils.data.Dataset):
    def __init__(self, image_ids, processor):
        self.image_ids = image_ids
        self.processor = processor

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = images_dict[img_id]
        img_path = IMG_DIR / img_info['file_name']
        
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)
        
        # Get annotations
        anns = annotations_dict.get(img_id, [])
        bboxes = [ann['bbox'] for ann in anns]
        category_ids = [ann['category_id'] for ann in anns]
        
        # Apply augmentations
        transformed = transform(image=image_np, bboxes=bboxes, category_ids=category_ids)
        image_np = transformed['image']
        bboxes = transformed['bboxes']
        category_ids = transformed['category_ids']
        
        # Format for the DETR Processor
        # DETR expects bboxes in Pascal VOC format [x_min, y_min, x_max, y_max] internally
        # We must convert from COCO [x_min, y_min, w, h]
        formatted_anns = []
        for bbox, cat_id in zip(bboxes, category_ids):
            x, y, w, h = bbox
            formatted_anns.append({
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [x, y, x + w, y + h], 
                "area": w * h,
                "iscrowd": 0
            })
            
        target = {'image_id': img_id, 'annotations': formatted_anns}
        
        # The processor handles resizing and normalizing the image for the specific model
        encoding = self.processor(images=image_np, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() 
        target = encoding["labels"][0] 

        return pixel_values, target

# 4. Collate Function
# Batches the data together, padding images if they are different sizes
def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }

# 5. Initialization
processor = DetrImageProcessor.from_pretrained(MODEL_NAME)

train_dataset = StSDataset(train_ids, processor)
val_dataset = StSDataset(val_ids, processor)

model = DetrForObjectDetection.from_pretrained(
    MODEL_NAME,
    revision="no_timm", 
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True # Crucial: We are changing the number of classes from the pre-trained COCO dataset
)

# 6. Training Configuration
training_args = TrainingArguments(
    output_dir= OUTPUT_DIR / "sts_detr_model",
    per_device_train_batch_size=BATCH_SIZE,
    logging_steps=50,
    learning_rate=1e-4,
    weight_decay=1e-4,
    save_total_limit=2,
    remove_unused_columns=False, # Required for DETR
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor,
)

# 7. GO!
if __name__ == "__main__":
    print("Starting DETR training...")
    trainer.train()
    trainer.save_model("sts_detr_model_final")
    print("Training complete and model saved.")