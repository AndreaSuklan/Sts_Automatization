"""
Predict.py  —  Quick inference viewer for the STS two-stage pipeline
═════════════════════════════════════════════════════════════════════
Runs Stage 1 (YOLOv8s detector) + Stage 2 (EfficientNet-B0 classifier)
on every screenshot in a folder and prints a tidy prediction summary.
Optionally saves annotated images to an output folder.

USAGE
  python Predict.py --input  path/to/screenshots/
  python Predict.py --input  path/to/screenshots/ --save-images
  python Predict.py --input  path/to/screenshots/ --save-images --out path/to/output/
  python Predict.py --input  path/to/screenshots/ --det-conf 0.4

DEFAULTS (mirrors the paths used in the rest of the pipeline)
  --det-weights   runs/detect/sts_detector/weights/best.pt
  --cls-weights   output/stage2_checkpoints/best.pt
  --det-conf      0.25
  --device        cpu
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ─────────────────────── optional-import guard ──────────────────── #
def _check():
    missing = []
    for pkg, imp in [("ultralytics", "ultralytics"), ("torch", "torch"),
                     ("torchvision", "torchvision"), ("Pillow", "PIL")]:
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)
    if missing:
        sys.exit(f"Missing packages: {', '.join(missing)}\n"
                 f"Run:  pip install {' '.join(missing)}")

_check()

import torch
import torch.nn as nn
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

# ─────────────────────── CLI ────────────────────────────────────── #
def _parse():
    ap = argparse.ArgumentParser(
        description="Two-stage STS inference: detect → classify → print",
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

    ckpt = torch.load(weights, map_location="cpu")
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
    """
    Returns a list of detections, each a dict:
      { generic, specific, det_conf, cls_conf, box }
    """
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

        # crop + classify
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
COLORS = {
    "Card": "#4FC3F7", "Enemy": "#EF5350", "HealthBar": "#66BB6A",
    "Intent": "#FFA726", "Player": "#AB47BC", "Potion": "#26C6DA",
    "Power": "#FFEE58", "Relic": "#FF7043",
}

def annotate(img_path: Path, detections: list[dict], out_path: Path) -> None:
    pil  = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    for d in detections:
        x1, y1, x2, y2 = d["box"]
        color  = COLORS.get(d["generic"], "#BDBDBD")
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

    # ── device ──────────────────────────────────────────────────────
    if args.device.lower() == "cpu":
        torch_device = torch.device("cpu")
    else:
        torch_device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available()
                                    else "cpu")

    print("=" * 60)
    print("  STS INFERENCE")
    print("=" * 60)
    print(f"  Images   : {len(images)} in {args.input}")
    print(f"  Device   : {torch_device}")
    print(f"  Det conf : {args.det_conf}")
    print(f"  Save     : {args.save_images} → {out_dir if args.save_images else '—'}")
    print("=" * 60)

    detector                      = load_detector(args.det_weights, args.device)
    classifier, cls_names, cls_tf = load_classifier(
        args.cls_weights, CLS_CLASSES, torch_device)

    print()
    t0 = time.time()
    total_dets = 0

    for img_path in images:
        dets = predict_image(
            img_path, detector, args.det_conf, args.device,
            classifier, cls_names, cls_tf, torch_device,
        )
        total_dets += len(dets)

        # ── print results ────────────────────────────────────────────
        print(f"┌─ {img_path.name}  ({len(dets)} detection{'s' if len(dets) != 1 else ''})")
        if not dets:
            print("│   (nothing detected)")
        for d in dets:
            print(f"│   [{d['generic']:<10}]  {d['specific']:<40}"
                  f"  det={d['det_conf']:.2f}  cls={d['cls_conf']:.2f}"
                  f"  box=({d['box'][0]},{d['box'][1]},{d['box'][2]},{d['box'][3]})")
        print()

        # ── optional save ────────────────────────────────────────────
        if args.save_images:
            annotate(img_path, dets, out_dir / img_path.name)

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"  Done.  {len(images)} images  |  {total_dets} total detections"
          f"  |  {elapsed:.1f}s  ({elapsed/len(images):.2f}s/img)")
    if args.save_images:
        print(f"  Annotated images → {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
<<<<<<< HEAD
    main()
=======
    main()
>>>>>>> 5ed1ea2ba0aed767bc07f4297240bace859cd7bc
