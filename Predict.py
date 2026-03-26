"""
Predict.py  —  STS Two-Stage Inference + Full C++ State Extraction
═════════════════════════════════════════════════════════════════════
Runs Stage 1 (YOLOv8s detector) + Stage 2 (EfficientNet-B0 classifier)
on every screenshot in a folder. 
Uses EasyOCR and Spatial Grouping to translate bounding boxes into a 
comprehensive, 1-to-1 mathematical state for C++ Reinforcement Learning.

USAGE
  python Predict.py --input  path/to/screenshots/
  python Predict.py --input  path/to/screenshots/ --save-images
"""

from __future__ import annotations

import argparse
import sys
import time
import re
from pathlib import Path
import hashlib
import colorsys

# ─────────────────────── optional-import guard ──────────────────── #
def _check():
    missing = []
    for pkg, imp in [("ultralytics", "ultralytics"), ("torch", "torch"),
                     ("torchvision", "torchvision"), ("Pillow", "PIL"),
                     ("easyocr", "easyocr"), ("numpy", "numpy")]:
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)
    if missing:
        sys.exit(f"Missing packages: {', '.join(missing)}\n"
                 f"Run:  pip install {' '.join(missing)}")

_check()

import numpy as np
import torch
import torch.nn as nn
import easyocr
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms
from ultralytics import YOLO

# ─────────────────────── defaults ───────────────────────────────── #
ROOT          = Path(__file__).resolve().parent
DET_WEIGHTS   = ROOT / "runs" / "detect" / "sts_detector" / "weights" / "best.pt"
CLS_WEIGHTS   = ROOT / "output" / "stage2_checkpoints" / "best.pt"
CLS_CLASSES   = ROOT / "output" / "stage2_checkpoints" / "classes.txt"

IMGSZ_CLS     = 128
_MEAN         = [0.485, 0.456, 0.406]
_STD          = [0.229, 0.224, 0.225]

# ─────────────────────── OCR Initialization ─────────────────────── #
USE_GPU = torch.cuda.is_available()
print(f"Loading EasyOCR (GPU={USE_GPU})...")
READER = easyocr.Reader(['en'], gpu=USE_GPU)

# ─────────────────────── OCR Helpers ────────────────────────────── #
def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert('RGB'))

def extract_fraction(image_crop: Image.Image) -> dict:
    """Reads HP or Energy (e.g., '42/80' or '3/3')."""
    img_array = pil_to_cv2(image_crop)
    results = READER.readtext(img_array, allowlist='0123456789/', detail=0)
    for text in results:
        match = re.search(r'(\d+)/(\d+)', text.replace(' ', ''))
        if match:
            return {"current": int(match.group(1)), "max": int(match.group(2))}
    return {"current": 0, "max": 0}

def extract_single_value(image_crop: Image.Image, allow_x: bool = False, allow_negative: bool = False):
    """Reads a single number. Supports 'X' for cards and '-' for negative powers."""
    img_array = pil_to_cv2(image_crop)
    allowlist = '0123456789'
    if allow_x: allowlist += 'X'
    if allow_negative: allowlist += '-'
        
    results = READER.readtext(img_array, allowlist=allowlist, detail=0)
    for text in results:
        clean = text.replace(' ', '').upper()
        if allow_x and clean == 'X':
            return -1  
            
        match = re.search(r'(-?\d+)', clean)
        if match:
            return int(match.group(1))
            
    return None  

def extract_intent_damage(image_crop: Image.Image) -> dict:
    """Reads intent damage, supporting multi-hits (e.g., '12' or '7x3')."""
    img_array = pil_to_cv2(image_crop)
    results = READER.readtext(img_array, allowlist='0123456789xX', detail=0)
    for text in results:
        clean = text.replace(' ', '').lower()
        match = re.search(r'(\d+)(?:x(\d+))?', clean)
        if match:
            dmg = int(match.group(1))
            hits = int(match.group(2)) if match.group(2) else 1
            return {"damage": dmg, "hits": hits}
    return {"damage": 0, "hits": 0}

def extract_ascension_floor(image_crop: Image.Image) -> dict:
    """Extracts all numbers from the Ascension/Floor display."""
    img_array = pil_to_cv2(image_crop)
    results = READER.readtext(img_array, detail=0)
    combined_text = " ".join(results)
    numbers = re.findall(r'\d+', combined_text)
    asc = int(numbers[0]) if len(numbers) > 0 else 0
    floor = int(numbers[1]) if len(numbers) > 1 else 0
    return {"ascension": asc, "floor": floor}

# ─────────────────────── CLI ────────────────────────────────────── #
def _parse():
    ap = argparse.ArgumentParser(
        description="Two-stage STS inference: detect → classify → OCR state extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input",       required=True,  type=Path, metavar="DIR",
                    help="Folder of screenshot images to run inference on")
    ap.add_argument("--det-weights", default=DET_WEIGHTS, type=Path,
                    help="Stage 1 YOLO weights (best.pt)")
    ap.add_argument("--cls-weights", default=CLS_WEIGHTS, type=Path,
                    help="Stage 2 EfficientNet weights (best.pt)")
    ap.add_argument("--det-conf",    default=0.25, type=float,
                    help="YOLO detection confidence threshold")
    ap.add_argument("--device",      default="cpu", type=str,
                    help="Inference device: 'cpu', '0', etc.")
    ap.add_argument("--save-images", action="store_true",
                    help="Save annotated images to --out folder")
    ap.add_argument("--out",         default=None, type=Path, metavar="DIR",
                    help="Where to save annotated images (default: <input>/predictions)")
    return ap.parse_args()

# ─────────────────────── model loaders ──────────────────────────── #
def load_detector(weights: Path, device: str) -> YOLO:
    print(f"  Loading detector  : {weights}")
    if not weights.exists():
        sys.exit(f"Detector weights not found: {weights}")
    return YOLO(str(weights))

def load_classifier(weights: Path, classes_txt: Path, device: torch.device):
    print(f"  Loading classifier: {weights}")
    if not weights.exists():
        sys.exit(f"Classifier weights not found: {weights}")

    if classes_txt.exists():
        class_names = [l.strip() for l in classes_txt.read_text("utf-8").splitlines() if l.strip()]
    else:
        sys.exit(f"classes.txt not found: {classes_txt}")

    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    ckpt_names = ckpt.get("classes", class_names)
    n = len(ckpt_names)

    base = models.efficientnet_b0(weights=None)
    base.classifier[1] = nn.Linear(base.classifier[1].in_features, n)
    base.load_state_dict(ckpt["model"])
    base = base.to(device).eval()

    tf = transforms.Compose([
        transforms.Resize((IMGSZ_CLS, IMGSZ_CLS)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    return base, ckpt_names, tf

# ─────────────────────── per-image inference ────────────────────── #
def predict_image(img_path: Path,
                  detector, det_conf: float, det_device: str,
                  classifier, cls_names, cls_tf,
                  torch_device: torch.device) -> list[dict]:
    result = detector(str(img_path), conf=det_conf,
                      device=det_device, verbose=False)[0]
    det_names = detector.names

    pil = Image.open(img_path).convert("RGB")
    detections = []

    for box in result.boxes:
        cls_id    = int(box.cls[0])
        det_c     = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        generic   = det_names.get(cls_id, str(cls_id))

        crop = pil.crop((x1, y1, x2, y2))
        tensor = cls_tf(crop).unsqueeze(0).to(torch_device)
        with torch.no_grad():
            probs = torch.softmax(classifier(tensor), dim=1)[0]
        top_conf, top_idx = probs.max(0)
        specific = cls_names[top_idx.item()] if top_idx.item() < len(cls_names) else "?"
        cls_c    = top_conf.item()

        detections.append({
            "generic":  generic,
            "specific": specific,
            "det_conf": det_c,
            "cls_conf": cls_c,
            "box":      (x1, y1, x2, y2),
        })

    return detections

# ─────────────────────── annotation ─────────────────────────────── #
def get_class_color(class_name: str) -> str:
    hash_int = int(hashlib.md5(class_name.encode('utf-8')).hexdigest(), 16)
    hue = (hash_int % 360) / 360.0
    sat = 0.5 + ((hash_int // 360) % 30) / 100.0
    val = 0.8 + ((hash_int // 10800) % 20) / 100.0
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"

def annotate(img_path: Path, detections: list[dict], out_path: Path) -> None:
    pil  = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    for d in detections:
        x1, y1, x2, y2 = d["box"]
        color  = get_class_color(d["generic"])
        label  = f"{d['specific']} ({d['cls_conf']:.2f})"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        tw = int(draw.textlength(label, font=font))
        draw.rectangle([x1, max(0, y1 - 17), x1 + tw + 4, max(0, y1 - 1)], fill=color)
        draw.text((x1 + 2, max(0, y1 - 16)), label, fill="black", font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)

# ─────────────────────── main ───────────────────────────────────── #
def main():
    args = _parse()

    if not args.input.is_dir():
        sys.exit(f"Input folder not found: {args.input}")

    images = sorted(
        p for p in args.input.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not images:
        sys.exit(f"No images found in {args.input}")

    out_dir = args.out or args.input / "predictions"

    if args.device.lower() == "cpu":
        torch_device = torch.device("cpu")
    else:
        torch_device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  STS INFERENCE & C++ STATE EXTRACTION")
    print("=" * 60)
    
    detector = load_detector(args.det_weights, args.device)
    classifier, cls_names, cls_tf = load_classifier(args.cls_weights, CLS_CLASSES, torch_device)

    print()
    t0 = time.time()
    total_dets = 0

    for img_path in images:
        dets = predict_image(
            img_path, detector, args.det_conf, args.device,
            classifier, cls_names, cls_tf, torch_device,
        )
        total_dets += len(dets)

        pil_full = Image.open(img_path).convert("RGB")
        
        # ── 1. Categorize Detections ─────────────────────────────────
        characters = [d for d in dets if d["generic"] in {"Player", "Enemy"}]
        char_attrs = [d for d in dets if d["generic"] in {"PlayerHealthBar", "Block", "Power", "Intent"}]
        
        globals_ui = [d for d in dets if d["generic"] in {
            "Card", "Relic", "EnergyOrb", "AscensionDisplay", 
            "GoldDisplay", "DeckSize", "DrawPile", "DiscardPile"
        }]

        # ── 2. Parse Global UI State ─────────────────────────────────
        game_state = {
            "energy": {"current": 0, "max": 0},
            "ascension": 0,
            "floor": 0,
            "gold": 0,               
            "deck_size": 0,          
            "draw_pile": 0,          
            "discard_pile": 0,       
            "relics": {},   
            "hand": [],     
            "characters": []
        }

        for ui in globals_ui:
            crop = pil_full.crop(ui["box"])
            if ui["generic"] == "EnergyOrb":
                game_state["energy"] = extract_fraction(crop)
            elif ui["generic"] == "AscensionDisplay":
                asc_data = extract_ascension_floor(crop)
                game_state["ascension"] = asc_data["ascension"]
                game_state["floor"] = asc_data["floor"]
            elif ui["generic"] == "Relic":
                count = extract_single_value(crop)
                game_state["relics"][ui["specific"]] = count if count is not None else 0
            elif ui["generic"] == "Card":
                cost = extract_single_value(crop, allow_x=True)
                game_state["hand"].append({"name": ui["specific"], "cost": cost if cost is not None else 0})
            elif ui["generic"] == "GoldDisplay":
                game_state["gold"] = extract_single_value(crop) or 0
            elif ui["generic"] == "DeckSize":
                game_state["deck_size"] = extract_single_value(crop) or 0
            elif ui["generic"] == "DrawPile":
                game_state["draw_pile"] = extract_single_value(crop) or 0
            elif ui["generic"] == "DiscardPile":
                game_state["discard_pile"] = extract_single_value(crop) or 0

        # ── 3. Parse Character State (Spatial Grouping) ──────────────
        for char in characters:
            cx1, cy1, cx2, cy2 = char["box"]
            char_data = {
                "identity": char["specific"],
                "hp_current": 0, "hp_max": 0, "block": 0,
                "powers": {}, 
                "intent_type": "None", "intent_damage": 0, "intent_hits": 0
            }

            for attr in char_attrs:
                ax1, ay1, ax2, ay2 = attr["box"]
                # Spatial check: is the attribute near this character's X coordinates?
                if cx1 - 50 <= ax1 <= cx2 + 50:
                    crop = pil_full.crop((ax1, ay1, ax2, ay2))
                    
                    if attr["generic"] == "PlayerHealthBar":
                        hp = extract_fraction(crop)
                        char_data["hp_current"] = hp["current"]
                        char_data["hp_max"] = hp["max"]
                    elif attr["generic"] == "Block":
                        char_data["block"] = extract_single_value(crop) or 0
                    elif attr["generic"] == "Intent":
                        char_data["intent_type"] = attr["specific"]
                        if "Attack" in attr["specific"]:
                            dmg_data = extract_intent_damage(crop)
                            char_data["intent_damage"] = dmg_data["damage"]
                            char_data["intent_hits"] = dmg_data["hits"]
                    elif attr["generic"] == "Power":
                        stacks = extract_single_value(crop, allow_negative=True)
                        char_data["powers"][attr["specific"]] = stacks if stacks is not None else 1

            game_state["characters"].append(char_data)

        # ── 4. Print the Fully Parsed State ──────────────────────────
        print(f"┌─ {img_path.name}")
        print(f"│  [Global State]")
        print(f"│    Ascension: {game_state['ascension']} | Floor: {game_state['floor']}")
        print(f"│    Energy: {game_state['energy']['current']}/{game_state['energy']['max']}")
        print(f"│    Gold: {game_state['gold']} | Deck: {game_state['deck_size']} | Draw: {game_state['draw_pile']} | Discard: {game_state['discard_pile']}")
        print(f"│    Relics: {game_state['relics']}")
        print(f"│    Hand:   {game_state['hand']}")
        
        for char in game_state["characters"]:
            print(f"│  [{char['identity']}]")
            print(f"│    HP: {char['hp_current']}/{char['hp_max']} | Block: {char['block']}")
            if char['intent_type'] != "None":
                dmg_str = f" ({char['intent_damage']}x{char['intent_hits']})" if char['intent_damage'] > 0 else ""
                print(f"│    Intent: {char['intent_type']}{dmg_str}")
            if char['powers']:
                print(f"│    Powers: {char['powers']}")
        print("└" + "─"*60 + "\n")

        # ── 5. Save Visual Annotations ───────────────────────────────
        if args.save_images:
            annotate(img_path, dets, out_dir / img_path.name)

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"  Done.  {len(images)} images  |  {total_dets} total detections")
    print(f"  {elapsed:.1f}s total  ({elapsed/len(images):.2f}s per image)")
    if args.save_images:
        print(f"  Annotated images → {out_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()