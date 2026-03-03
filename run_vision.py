import torch
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from transformers import DetrImageProcessor, DetrForObjectDetection

# --- CONFIGURATION ---
MAIN_DIR = Path.cwd()
# Point this to where your trainer.save_model() saved the final weights
MODEL_PATH = MAIN_DIR / "output" / "sts_detr_model_final" 
TEST_IMAGE_PATH = MAIN_DIR / "output" / "images" 
PRED_DIR = MAIN_DIR / "output" / "hand_predictions"

CONFIDENCE_THRESHOLD = 0.85 # Ignore guesses the model isn't at least 85% sure about
NUM_TEST_IMAGES = 5

def run_inference(NUM_TEST_IMAGES=5):
    print(f"Loading model from {MODEL_PATH}...")
    processor = DetrImageProcessor.from_pretrained(MODEL_PATH)
    model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
    
    print(f"Analyzing {TEST_IMAGE_PATH.name}...")

    image_files = list(TEST_IMAGE_PATH.glob("*.jpg"))

    for img in range(NUM_TEST_IMAGES):

        image = Image.open(image_files[img]).convert("RGB")
        
        # 1. Preprocess the image for the transformer
        inputs = processor(images=image, return_tensors="pt")
        
        # 2. Run the image through the network
        with torch.no_grad(): # Disables gradient calculation to save memory and speed up inference
            outputs = model(**inputs)
            
        # 3. Post-process the raw mathematical output into usable screen coordinates
        # DETR outputs normalized coordinates. We pass the original image size to convert them back to absolute pixels.
        target_sizes = torch.tensor([image.size[::-1]]) # [height, width]
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=CONFIDENCE_THRESHOLD)[0]

        # 4. Data Extraction & Sorting
        detected_cards = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            class_name = model.config.id2label[label.item()]
            confidence = round(score.item(), 3)
            
            detected_cards.append({
                "class": class_name,
                "confidence": confidence,
                "bbox": box, # [x_min, y_min, x_max, y_max]
                "center_x": (box[0] + box[2]) / 2 # Calculate the horizontal center for sorting
            })
            
        # Crucial Step: Sort the detected cards from left to right based on their X-coordinate.
        # Because cards heavily overlap, DETR might detect the 4th card before the 1st card.
        # Sorting ensures the final JSON array matches the actual hand order left-to-right.
        detected_cards.sort(key=lambda x: x["center_x"])

        # Build the clean JSON array for the RL Brain
        final_json_state = [card["class"] for card in detected_cards]
        
        # 5. Visual Representation (Drawing the boxes)
        draw = ImageDraw.Draw(image)
        try:
            # Tries to use a standard Windows font for readability
            font = ImageFont.truetype("arial.ttf", 18)
        except IOError:
            font = ImageFont.load_default()

        for card in detected_cards:
            box = card["bbox"]
            class_name = card["class"]
            conf = card["confidence"]
            
            # Draw the bounding box (Red, 3 pixels thick)
            draw.rectangle(box, outline="red", width=3)
            
            # Draw the background for the text label
            label_text = f"{class_name} ({conf:.2f})"
            text_bbox = draw.textbbox((box[0], box[1]), label_text, font=font)
            draw.rectangle([text_bbox[0], text_bbox[1] - 20, text_bbox[2], text_bbox[3]], fill="red")
            
            # Draw the text
            draw.text((box[0], box[1] - 20), label_text, fill="white", font=font)

        # 6. Output the results
        print("\n--- INFERENCE RESULTS ---")
        print(f"Cards detected: {len(detected_cards)}")
        print(f"Internal Knowledge Array: {final_json_state}")
        
        # Save and show the visualizer image
        output_img_path = PRED_DIR / f"vision_test_result_{img}.jpg"
        image.save(output_img_path)
        print(f"\nSaved graphical representation to {output_img_path}")

if __name__ == "__main__":
    run_inference()