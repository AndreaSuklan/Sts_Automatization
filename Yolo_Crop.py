from __future__ import annotations
"""
╔══════════════════════════════════════════════════════════════════╗
║       STS YOLO CROP EXTRACTOR + STAGE 1 DATASET BUILDER         ║
╠══════════════════════════════════════════════════════════════════╣
║  In a single pass over the screenshots this script does TWO      ║
║  things:                                                         ║
║                                                                  ║
║  1. STAGE 2 CROPS                                                ║
║     Crops every bounding box into class-named folders for the    ║
║     EfficientNet classifier (unchanged from the original).       ║
║                                                                  ║
║  2. STAGE 1 DATASET  (no image copies - zero extra disk space)   ║
║     Builds the train/val split, remaps specific class ids to     ║
║     8 generic ones, and hard-links the original screenshots      ║
║     into stage1_dataset/images/{train,val}/.  Hard-links share   ║
║     the same inode as the originals, so disk usage is zero.      ║
║     If source and destination are on different drives the script  ║
║     falls back to a regular copy automatically.                  ║
║     Remapped label .txt files and data.yaml are written here.    ║
╠══════════════════════════════════════════════════════════════════╣
║  CLI ARGUMENTS (all optional - defaults match original CONFIG)   ║
║    --val-split FLOAT   fraction held out for validation (0.15)   ║
║    --padding   INT     extra pixels around each crop (8)         ║
║    --workers   INT     parallel crop workers (cpu_count)         ║
║    --rebuild           delete & fully rebuild stage1_dataset     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import argparse
import os
import sys
import time
import random
import shutil
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is not installed.  Run:  pip install Pillow")


# ══════════════════════════════ FIXED CONFIG ═════════════════════════ #
# Values that are never exposed as CLI args (stable across every run).

BASE_DIR = Path.cwd() / "output"

IMAGES_DIR   = BASE_DIR / "gen_screen_images"
LABELS_DIR   = BASE_DIR / "gen_screen_labels"
CLASSES_FILE = BASE_DIR / "gen_screen_classes.txt"

CROPS_DIR  = BASE_DIR / "cropped_dataset"
STAGE1_DIR = BASE_DIR / "stage1_dataset"

MIN_CROP_SIZE = 16
RESIZE_TO     = None      # None = keep native size
SAVE_FORMAT   = "PNG"
JPEG_QUALITY  = 95
RANDOM_SEED   = 42

GENERIC_CLASSES = [
    "Card",       # 0
    "Enemy",      # 1
    "HealthBar",  # 2
    "Intent",     # 3
    "Player",     # 4
    "Potion",     # 5
    "Power",      # 6
    "Relic",      # 7
]

PREFIX_MAP = {
    "Card_":       "Card",
    "Enemy_":      "Enemy",
    "HealthBar_":  "HealthBar",
    "Intent_":     "Intent",
    "Player_":     "Player",
    "Potion_":     "Potion",
    "Power_":      "Power",
    "Relic_":      "Relic",
}

# ═══════════════════════════════ END FIXED ═══════════════════════════ #

_GENERIC_ID = {name: idx for idx, name in enumerate(GENERIC_CLASSES)}


# ─────────────────────────── CLI parsing ────────────────────────────── #

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="STS YOLO Crop Extractor + Stage 1 Dataset Builder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--val-split", type=float, default=0.15, metavar="FLOAT",
        help="Fraction of screenshots held out for validation",
    )
    ap.add_argument(
        "--padding", type=int, default=8, metavar="INT",
        help="Extra pixels added around every bounding-box crop",
    )
    ap.add_argument(
        "--workers", type=int, default=None, metavar="INT",
        help="Parallel crop worker processes (default: os.cpu_count())",
    )
    ap.add_argument(
        "--rebuild", action="store_true",
        help="Delete and fully rebuild stage1_dataset even if data.yaml exists",
    )
    return ap.parse_args()


# ──────────────────────────── helpers ───────────────────────────── #

def _ensure_dirs() -> None:
    Path("logs").mkdir(parents=True, exist_ok=True)
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    STAGE1_DIR.mkdir(parents=True, exist_ok=True)


def load_classes(classes_file: Path) -> dict:
    mapping: dict = {}
    if not classes_file.exists():
        print(f"Warning: Could not find {classes_file}.")
        print("   Class IDs will be used as folder names (0, 1, 2 ...).")
        return mapping
    for idx, line in enumerate(classes_file.read_text(encoding="utf-8").splitlines()):
        name = line.strip()
        if name:
            safe = name.replace("/", "_").replace("\\", "_").replace(":", "_")
            mapping[idx] = safe
    return mapping


def resolve_generic(specific_name: str):
    for prefix, generic in PREFIX_MAP.items():
        if specific_name.startswith(prefix):
            return generic
    lower = specific_name.lower()
    for generic in GENERIC_CLASSES:
        if generic.lower() in lower:
            return generic
    return None


def yolo_to_pixels(cx, cy, w, h, img_w, img_h, padding=0):
    px_cx = cx * img_w
    px_cy = cy * img_h
    px_w  = w  * img_w
    px_h  = h  * img_h
    x1 = int(px_cx - px_w / 2) - padding
    y1 = int(px_cy - px_h / 2) - padding
    x2 = int(px_cx + px_w / 2) + padding
    y2 = int(px_cy + px_h / 2) + padding
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)


def link_or_copy(src: Path, dst: Path) -> bool:
    try:
        os.link(src, dst)
        return True
    except OSError:
        shutil.copy2(src, dst)
        return False


# ──────────────────────────── Stage 2: crop worker ──────────────────── #

def process_image(args):
    """
    Worker function - crops all bounding boxes from one screenshot.
    Plain strings only (picklable on Windows 'spawn' method).
    Returns (stem, n_saved, n_skipped, errors).
    """
    img_path, label_path, output_dir, classes, padding, min_size, \
        resize_to, fmt, quality = args

    img_path   = Path(img_path)
    label_path = Path(label_path)
    output_dir = Path(output_dir)

    errors  = []
    saved   = 0
    skipped = 0
    stem    = img_path.stem

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as exc:
        return stem, 0, 0, [f"Cannot open image {img_path}: {exc}"]

    img_w, img_h = img.size

    if not label_path.exists():
        return stem, 0, 0, [f"Label file missing: {label_path}"]

    lines = [l.strip() for l in label_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    crop_counter: dict = {}

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"[{stem}] Malformed label line: '{line}'")
            continue

        class_id        = int(parts[0])
        cx, cy, w, h    = map(float, parts[1:])
        class_name      = classes.get(class_id, str(class_id))
        x1, y1, x2, y2 = yolo_to_pixels(cx, cy, w, h, img_w, img_h, padding)

        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            skipped += 1
            continue

        crop = img.crop((x1, y1, x2, y2))
        if resize_to:
            crop = crop.resize((resize_to, resize_to), Image.LANCZOS)

        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        idx = crop_counter.get(class_name, 0)
        crop_counter[class_name] = idx + 1

        ext      = "jpg" if fmt == "JPEG" else "png"
        out_path = class_dir / f"{stem}_crop{idx}.{ext}"

        try:
            save_kwargs = {"format": fmt}
            if fmt == "JPEG":
                save_kwargs["quality"] = quality
            crop.save(out_path, **save_kwargs)
            saved += 1
        except Exception as exc:
            errors.append(f"Failed to save {out_path}: {exc}")
            skipped += 1

    return stem, saved, skipped, errors


# ─────────────────────── Stage 1: dataset builder ───────────────────── #

def build_stage1_dataset(
    pairs,
    specific_classes: dict,
    val_split: float,
    rebuild: bool,
) -> None:
    """
    Build the Stage 1 YOLO-detection dataset in STAGE1_DIR.

    Parameters
    ----------
    pairs           : list of (img_path, lbl_path) tuples
    specific_classes: {class_id: class_name} dict
    val_split       : fraction of images used for validation
    rebuild         : if True, delete and fully rebuild STAGE1_DIR
    """
    yaml_path = STAGE1_DIR / "data.yaml"

    if yaml_path.exists() and not rebuild:
        print(f"  Stage 1 dataset already exists at {STAGE1_DIR}")
        print("  Pass --rebuild to force a full rebuild.\n")
        return

    if rebuild and STAGE1_DIR.exists():
        shutil.rmtree(STAGE1_DIR)
        print("  Removed existing stage1_dataset (--rebuild set).")

    # ── specific-id -> generic-id remap ───────────────────────────────
    id_remap: dict = {}
    unmapped = []
    for spec_id, spec_name in specific_classes.items():
        generic = resolve_generic(spec_name)
        if generic is not None:
            id_remap[spec_id] = _GENERIC_ID[generic]
        else:
            unmapped.append(spec_name)

    print(f"  Specific classes : {len(specific_classes)}")
    print(f"  Generic classes  : {len(GENERIC_CLASSES)}")
    if unmapped:
        print(f"  Unmapped (will be dropped from labels): {unmapped}")

    # ── train / val split ─────────────────────────────────────────────
    rng = random.Random(RANDOM_SEED)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    n_val       = max(1, int(len(shuffled) * val_split))
    split_pairs = {"val": shuffled[:n_val], "train": shuffled[n_val:]}
    print(f"  Train: {len(split_pairs['train'])}  |  Val: {len(split_pairs['val'])}")

    # ── write files ───────────────────────────────────────────────────
    hardlinks = copies = 0

    for split, split_list in split_pairs.items():
        img_out = STAGE1_DIR / "images" / split
        lbl_out = STAGE1_DIR / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path in split_list:
            dst_img = img_out / img_path.name
            if not dst_img.exists():
                used_link = link_or_copy(img_path, dst_img)
                if used_link:
                    hardlinks += 1
                else:
                    copies += 1

            remapped = []
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                orig_id = int(parts[0])
                if orig_id not in id_remap:
                    continue
                remapped.append(f"{id_remap[orig_id]} {' '.join(parts[1:])}")

            (lbl_out / lbl_path.name).write_text("\n".join(remapped), encoding="utf-8")

    # ── data.yaml ─────────────────────────────────────────────────────
    yaml_content = (
        f"path: {STAGE1_DIR.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"\n"
        f"nc: {len(GENERIC_CLASSES)}\n"
        f"names: {GENERIC_CLASSES}\n"
    )
    yaml_path.write_text(yaml_content, encoding="utf-8")

    (STAGE1_DIR / "classes.txt").write_text(
        "\n".join(f"{i}  {name}" for i, name in enumerate(GENERIC_CLASSES)),
        encoding="utf-8",
    )

    mode_str = (
        f"{hardlinks} hard-linked, {copies} copied"
        if copies else
        f"{hardlinks} hard-linked (zero extra disk space)"
    )
    print(f"  Images : {mode_str}")
    print(f"  data.yaml -> {yaml_path}\n")


# ──────────────────────────────── main ──────────────────────────── #

def main():
    args = _parse_args()
    _ensure_dirs()

    print("=" * 65)
    print("  STS YOLO CROP EXTRACTOR + STAGE 1 DATASET BUILDER")
    print("=" * 65)
    print(f"  Images dir   : {IMAGES_DIR}")
    print(f"  Labels dir   : {LABELS_DIR}")
    print(f"  Classes file : {CLASSES_FILE}")
    print(f"  Crops output : {CROPS_DIR}")
    print(f"  Stage1 output: {STAGE1_DIR}")
    print(f"  Padding      : {args.padding} px")
    print(f"  Min crop     : {MIN_CROP_SIZE} px")
    print(f"  Resize to    : {RESIZE_TO or 'native (no resize)'}")
    print(f"  Format       : {SAVE_FORMAT}")
    print(f"  Val split    : {args.val_split}")
    print(f"  Workers      : {args.workers or os.cpu_count()}")
    print(f"  Rebuild      : {args.rebuild}")
    print("=" * 65)

    for directory in (IMAGES_DIR, LABELS_DIR):
        if not directory.exists():
            sys.exit(f"Directory not found: {directory}")

    specific_classes = load_classes(CLASSES_FILE)
    print(f"  Loaded {len(specific_classes)} specific class names.\n")

    image_files = sorted(IMAGES_DIR.glob("*.png"))
    if not image_files:
        sys.exit(f"No .png files found in {IMAGES_DIR}")

    pairs = []
    missing_labels = 0
    for img_path in image_files:
        lbl_path = LABELS_DIR / img_path.with_suffix(".txt").name
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))
        else:
            missing_labels += 1

    print(f"  Found {len(image_files)} images - {len(pairs)} paired with labels.")
    if missing_labels:
        print(f"  {missing_labels} images skipped (no matching .txt label).")
    if not pairs:
        sys.exit("Nothing to process.")

    # ── Stage 1: build detector dataset ───────────────────────────
    print("\n--- Stage 1 dataset -------------------------------------------")
    build_stage1_dataset(
        pairs,
        specific_classes,
        val_split=args.val_split,
        rebuild=args.rebuild,
    )

    # ── Stage 2: crop bounding boxes ──────────────────────────────
    print("--- Stage 2 crops ---------------------------------------------")
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [
        (str(img), str(lbl), str(CROPS_DIR),
         specific_classes, args.padding, MIN_CROP_SIZE,
         RESIZE_TO, SAVE_FORMAT, JPEG_QUALITY)
        for img, lbl in pairs
    ]

    total_saved = total_skipped = done = 0
    all_errors = []
    t0 = time.time()
    print("\n  Cropping ...\n")

    n_workers = args.workers or os.cpu_count()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(process_image, t): t[0] for t in tasks}
        for future in as_completed(futures):
            stem, saved, skipped, errors = future.result()
            total_saved   += saved
            total_skipped += skipped
            all_errors    += errors
            done          += 1

            if done % 10 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                rate    = done / elapsed if elapsed > 0 else 0
                eta     = (len(tasks) - done) / rate if rate > 0 else 0
                print(
                    f"  [{done:>5}/{len(tasks)}]  "
                    f"{total_saved} crops saved  |  "
                    f"{elapsed:.1f}s elapsed  |  ETA {eta:.0f}s",
                    end="\r",
                )

    elapsed = time.time() - t0
    print("\n")
    print("=" * 65)
    print(f"  DONE  in {elapsed:.1f}s")
    print(f"  Crops saved    : {total_saved}")
    print(f"  Crops skipped  : {total_skipped}  (too small or bad box)")
    print(f"  Errors         : {len(all_errors)}")
    print("=" * 65)

    # ── Stage 2 class summary ──────────────────────────────────────
    print("\n  CROP CLASS SUMMARY  (crops per class):\n")
    class_counts = {
        d.name: len(list(d.iterdir()))
        for d in sorted(CROPS_DIR.iterdir())
        if d.is_dir()
    }
    if class_counts:
        col_w = max(len(k) for k in class_counts)
        for name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            bar = "=" * min(count // max(1, max(class_counts.values()) // 40), 40)
            print(f"    {name:<{col_w}}  {count:>5}  {bar}")

    if all_errors:
        log_path = CROPS_DIR / "_crop_errors.txt"
        log_path.write_text("\n".join(all_errors), encoding="utf-8")
        print(f"\n  {len(all_errors)} errors logged -> {log_path}")

    print()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()