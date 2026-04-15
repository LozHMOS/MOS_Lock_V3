"""
yolo_training/generate_synthetic_samples.py
MOSLock V3 — Synthetic Training Image Generator
Mapped Out Solutions® | mos | Lock®

Generates 30 synthetic training images per class (750 total for 25 classes)
with corresponding YOLO .txt annotations.

These are stand-in images only — coloured shapes, labels, and patterns
that allow the YOLO model to train end-to-end before real site photos arrive.
Human-reviewed real annotations from bootstrap_annotations.py replace these.

Output structure:
    dataset/
    ├── images/
    │   ├── train/   (600 images — 24 per class)
    │   ├── val/     (100 images — 4 per class)
    │   └── test/    (50 images  — 2 per class)
    └── labels/
        ├── train/
        ├── val/
        └── test/

Usage:
    python generate_synthetic_samples.py
    python generate_synthetic_samples.py --output /abs/path/dataset --samples 50
    python generate_synthetic_samples.py --seed 42
"""

from __future__ import annotations
import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
except ImportError:
    print("ERROR: Pillow not installed. Run: pip install Pillow")
    sys.exit(1)


# ── Class definitions ────────────────────────────────────────────────────────

CLASS_NAMES = [
    "personal_lock",
    "permit_lock",
    "circuit_breaker_open",
    "circuit_breaker_closed",
    "isolator_switch_open",
    "isolator_switch_closed",
    "fuse_withdrawn",
    "fuse_inserted",
    "earth_switch_applied",
    "earth_switch_removed",
    "disconnect_open",
    "disconnect_closed",
    "compartment_1",
    "compartment_2",
    "compartment_3",
    "compartment_4",
    "compartment_5",
    "compartment_6",
    "compartment_7",
    "compartment_8",
    "compartment_9",
    "compartment_10",
    "indicator_lit",
    "indicator_off",
    "indicator_fault",
]

# Visual properties for each class — shapes and colours
CLASS_VISUAL = {
    "personal_lock":          {"shape": "padlock",     "color": "#FF6B35", "label_bg": "#FF6B35", "size_frac": (0.05, 0.12)},
    "permit_lock":            {"shape": "padlock_lg",  "color": "#F5A623", "label_bg": "#F5A623", "size_frac": (0.08, 0.15)},
    "circuit_breaker_open":   {"shape": "switch",      "color": "#1ABF74", "label_bg": "#1ABF74", "size_frac": (0.06, 0.14)},
    "circuit_breaker_closed": {"shape": "switch",      "color": "#E8392D", "label_bg": "#E8392D", "size_frac": (0.06, 0.14)},
    "isolator_switch_open":   {"shape": "lever",       "color": "#1ABF74", "label_bg": "#1ABF74", "size_frac": (0.07, 0.16)},
    "isolator_switch_closed": {"shape": "lever",       "color": "#E8392D", "label_bg": "#E8392D", "size_frac": (0.07, 0.16)},
    "fuse_withdrawn":         {"shape": "cylinder",    "color": "#1ABF74", "label_bg": "#1ABF74", "size_frac": (0.04, 0.10)},
    "fuse_inserted":          {"shape": "cylinder",    "color": "#E8392D", "label_bg": "#E8392D", "size_frac": (0.04, 0.10)},
    "earth_switch_applied":   {"shape": "earth",       "color": "#007BFF", "label_bg": "#007BFF", "size_frac": (0.06, 0.13)},
    "earth_switch_removed":   {"shape": "earth",       "color": "#F5A623", "label_bg": "#F5A623", "size_frac": (0.06, 0.13)},
    "disconnect_open":        {"shape": "gap",         "color": "#1ABF74", "label_bg": "#1ABF74", "size_frac": (0.07, 0.15)},
    "disconnect_closed":      {"shape": "gap",         "color": "#E8392D", "label_bg": "#E8392D", "size_frac": (0.07, 0.15)},
    "compartment_1":          {"shape": "panel",       "color": "#3ECFCF", "label_bg": "#071539", "size_frac": (0.25, 0.50)},
    "compartment_2":          {"shape": "panel",       "color": "#3ECFCF", "label_bg": "#071539", "size_frac": (0.25, 0.50)},
    "compartment_3":          {"shape": "panel",       "color": "#3ECFCF", "label_bg": "#071539", "size_frac": (0.25, 0.50)},
    "compartment_4":          {"shape": "panel",       "color": "#3ECFCF", "label_bg": "#071539", "size_frac": (0.20, 0.45)},
    "compartment_5":          {"shape": "panel",       "color": "#3ECFCF", "label_bg": "#071539", "size_frac": (0.20, 0.45)},
    "compartment_6":          {"shape": "panel",       "color": "#3ECFCF", "label_bg": "#071539", "size_frac": (0.20, 0.45)},
    "compartment_7":          {"shape": "panel",       "color": "#3ECFCF", "label_bg": "#071539", "size_frac": (0.20, 0.40)},
    "compartment_8":          {"shape": "panel",       "color": "#3ECFCF", "label_bg": "#071539", "size_frac": (0.20, 0.40)},
    "compartment_9":          {"shape": "panel",       "color": "#3ECFCF", "label_bg": "#071539", "size_frac": (0.20, 0.40)},
    "compartment_10":         {"shape": "panel",       "color": "#3ECFCF", "label_bg": "#071539", "size_frac": (0.15, 0.35)},
    "indicator_lit":          {"shape": "circle_lit",  "color": "#E8392D", "label_bg": "#E8392D", "size_frac": (0.03, 0.08)},
    "indicator_off":          {"shape": "circle_off",  "color": "#555555", "label_bg": "#555555", "size_frac": (0.03, 0.08)},
    "indicator_fault":        {"shape": "circle_x",    "color": "#CC0000", "label_bg": "#CC0000", "size_frac": (0.03, 0.08)},
}

# Substation background colours — grey steel, rust, dark panel
BG_COLORS = [
    (180, 178, 170),  # BST008 grey steel
    (160, 160, 155),  # weathered grey
    (140, 135, 128),  # darker panel
    (190, 185, 175),  # light steel
    (100, 100, 95),   # dark cabinet
    (210, 205, 195),  # painted metal
    (80, 78, 72),     # underground dark
    (50, 48, 45),     # very dark (underground)
]


def hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def random_bg_color(rng: random.Random) -> Tuple[int, int, int]:
    base = rng.choice(BG_COLORS)
    jitter = tuple(max(0, min(255, c + rng.randint(-15, 15))) for c in base)
    return jitter


def add_noise(img: Image.Image, rng: random.Random, intensity: float = 0.03) -> Image.Image:
    """Add Gaussian noise to simulate camera noise underground."""
    arr = np.array(img).astype(np.float32)
    noise = np.random.randn(*arr.shape) * 255 * intensity
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def add_blur(img: Image.Image, rng: random.Random) -> Image.Image:
    """Random slight blur to simulate camera motion / focus."""
    if rng.random() < 0.3:
        radius = rng.uniform(0.3, 1.2)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    return img


def adjust_brightness(img: Image.Image, rng: random.Random) -> Image.Image:
    """Simulate underground low-light or direct sunlight."""
    from PIL import ImageEnhance
    factor = rng.uniform(0.45, 1.55)
    return ImageEnhance.Brightness(img).enhance(factor)


def draw_padlock(draw: ImageDraw.Draw, x1: int, y1: int, x2: int, y2: int,
                 color: Tuple, large: bool = False):
    """Draw a simple padlock shape."""
    w = x2 - x1
    h = y2 - y1
    body_top = y1 + h * 0.4
    # Shackle (arch)
    shackle_left  = x1 + w * 0.25
    shackle_right = x1 + w * 0.75
    shackle_top   = y1
    shackle_w = shackle_right - shackle_left
    draw.arc(
        [shackle_left, shackle_top, shackle_right, body_top + h * 0.1],
        start=200, end=340, fill=color, width=max(2, int(w * 0.1))
    )
    # Lock body
    body_r = max(2, int(min(w, h) * 0.08))
    draw.rounded_rectangle([x1, body_top, x2, y2], radius=body_r, fill=color)
    # Keyhole
    kx = (x1 + x2) // 2
    ky = int(body_top + (y2 - body_top) * 0.4)
    kr = max(2, int(w * 0.08))
    hole_color = tuple(max(0, c - 60) for c in color)
    draw.ellipse([kx - kr, ky - kr, kx + kr, ky + kr], fill=hole_color)
    draw.rectangle([kx - kr // 2, ky, kx + kr // 2, ky + kr * 2], fill=hole_color)


def draw_switch(draw: ImageDraw.Draw, x1: int, y1: int, x2: int, y2: int,
                color: Tuple, closed: bool = False):
    """Draw a circuit breaker symbol."""
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    lw = max(2, int((x2 - x1) * 0.08))
    # Contact lines
    draw.line([x1, cy, x1 + (x2 - x1) * 0.35, cy], fill=color, width=lw)
    draw.line([x1 + (x2 - x1) * 0.65, cy, x2, cy], fill=color, width=lw)
    if closed:
        # Closed: line connects
        draw.line([x1 + (x2 - x1) * 0.35, cy, x1 + (x2 - x1) * 0.65, cy],
                  fill=color, width=lw)
    else:
        # Open: gap in middle, angled line
        draw.line([x1 + (x2 - x1) * 0.35, cy, x1 + (x2 - x1) * 0.55, y1 + (y2 - y1) * 0.3],
                  fill=color, width=lw)
    # Contact dots
    r = max(3, int((x2 - x1) * 0.05))
    draw.ellipse([x1 + (x2 - x1) * 0.33 - r, cy - r, x1 + (x2 - x1) * 0.33 + r, cy + r],
                 fill=color)
    draw.ellipse([x1 + (x2 - x1) * 0.67 - r, cy - r, x1 + (x2 - x1) * 0.67 + r, cy + r],
                 fill=color)


def draw_indicator(draw: ImageDraw.Draw, x1: int, y1: int, x2: int, y2: int,
                   state: str, color: Tuple):
    """Draw a live line indicator light."""
    r = min(x2 - x1, y2 - y1) // 2
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    if state == "lit":
        # Bright glow
        for i in range(3):
            outer_r = r + i * 3
            glow_alpha = max(50, 150 - i * 50)
            glow_color = color + (glow_alpha,)
            draw.ellipse([cx - outer_r, cy - outer_r, cx + outer_r, cy + outer_r],
                         outline=(*color, 200), width=2)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif state == "off":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color,
                     outline=(100, 100, 100), width=2)
    else:  # fault
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
        # X mark
        lw = max(2, r // 3)
        draw.line([cx - r * 0.6, cy - r * 0.6, cx + r * 0.6, cy + r * 0.6],
                  fill=(255, 255, 255), width=lw)
        draw.line([cx + r * 0.6, cy - r * 0.6, cx - r * 0.6, cy + r * 0.6],
                  fill=(255, 255, 255), width=lw)


def draw_panel(draw: ImageDraw.Draw, x1: int, y1: int, x2: int, y2: int,
               color: Tuple, label: str):
    """Draw a substation panel/compartment rectangle."""
    # Panel body — slightly lighter fill
    fill = tuple(min(255, c + 40) for c in (100, 100, 98))
    draw.rectangle([x1, y1, x2, y2], fill=fill, outline=color, width=3)
    # Top stripe
    draw.rectangle([x1, y1, x2, y1 + 20], fill=(7, 21, 57))
    # Bolts in corners
    bolt_r = 4
    for bx, by in [(x1 + 10, y1 + 30), (x2 - 10, y1 + 30),
                   (x1 + 10, y2 - 10),  (x2 - 10, y2 - 10)]:
        draw.ellipse([bx - bolt_r, by - bolt_r, bx + bolt_r, by + bolt_r],
                     fill=(150, 150, 145))
    # Ventilation slots
    slot_y = y1 + (y2 - y1) * 0.5
    for i in range(3):
        sy = int(slot_y + i * 12)
        draw.rectangle([x1 + 15, sy, x2 - 15, sy + 4], fill=(80, 80, 78))
    # Door handle
    hx = (x1 + x2) // 2
    draw.rectangle([hx - 15, y2 - 40, hx + 15, y2 - 30],
                   fill=(160, 155, 148), outline=(120, 118, 112), width=1)


def generate_image(
    class_id:  int,
    class_name: str,
    img_size:  int,
    rng:       random.Random,
) -> Tuple[Image.Image, List[float]]:
    """
    Generate one synthetic training image for class_id.
    Returns (PIL.Image, [cx, cy, w, h]) in YOLO normalised format.
    """
    vis = CLASS_VISUAL[class_name]
    color_rgb = hex_to_rgb(vis["color"])
    bg_rgb    = random_bg_color(rng)

    # Background — add grid lines to simulate panel surface
    img = Image.new("RGB", (img_size, img_size), bg_rgb)
    draw = ImageDraw.Draw(img)

    # Background texture — faint grid
    grid_color = tuple(max(0, c - 15) for c in bg_rgb)
    for gx in range(0, img_size, rng.randint(20, 60)):
        draw.line([(gx, 0), (gx, img_size)], fill=grid_color, width=1)
    for gy in range(0, img_size, rng.randint(20, 60)):
        draw.line([(0, gy), (img_size, gy)], fill=grid_color, width=1)

    # Place random distractor elements (bolts, cables, text)
    for _ in range(rng.randint(2, 6)):
        dx = rng.randint(0, img_size - 1)
        dy = rng.randint(0, img_size - 1)
        dr = rng.randint(2, 8)
        draw.ellipse([dx - dr, dy - dr, dx + dr, dy + dr],
                     fill=tuple(max(0, c - 30) for c in bg_rgb))

    # Object size
    min_frac, max_frac = vis["size_frac"]
    obj_w = int(img_size * rng.uniform(min_frac, max_frac))
    obj_h = int(obj_w * rng.uniform(0.7, 1.4))
    obj_w = max(16, min(obj_w, img_size - 20))
    obj_h = max(16, min(obj_h, img_size - 20))

    # Position — centred with random offset (must fit in image)
    max_x = img_size - obj_w - 10
    max_y = img_size - obj_h - 10
    x1 = rng.randint(10, max(11, max_x))
    y1 = rng.randint(10, max(11, max_y))
    x2 = x1 + obj_w
    y2 = y1 + obj_h

    shape = vis["shape"]

    if shape in ("padlock", "padlock_lg"):
        draw_padlock(draw, x1, y1, x2, y2, color_rgb, large=(shape == "padlock_lg"))
        # Add lock number label for permit lock
        if shape == "padlock_lg":
            lock_num = rng.randint(1, 20)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            except Exception:
                font = ImageFont.load_default()
            draw.text((x1 + obj_w * 0.3, y1 + obj_h * 0.65), str(lock_num),
                      fill=(255, 255, 255), font=font)

    elif shape in ("switch",):
        closed = "closed" in class_name
        draw_switch(draw, x1, y1, x2, y2, color_rgb, closed=closed)

    elif shape == "lever":
        # Isolator switch — vertical bar that rotates
        lw = max(3, obj_w // 8)
        cx = (x1 + x2) // 2
        if "open" in class_name:
            # Rotated 45° (open)
            draw.line([cx, y2, x1 + obj_w * 0.7, y1], fill=color_rgb, width=lw)
        else:
            # Vertical (closed/connected)
            draw.line([cx, y1, cx, y2], fill=color_rgb, width=lw)
        # Pivot point
        draw.ellipse([cx - 5, y2 - 5, cx + 5, y2 + 5], fill=color_rgb)

    elif shape == "cylinder":
        # Fuse — draw as rounded rectangle + end caps
        if obj_w > obj_h:
            draw.rounded_rectangle([x1, y1, x2, y2], radius=obj_h // 3, fill=color_rgb)
            draw.ellipse([x1, y1, x1 + obj_h, y2], fill=tuple(min(255, c + 30) for c in color_rgb))
            draw.ellipse([x2 - obj_h, y1, x2, y2], fill=tuple(min(255, c + 30) for c in color_rgb))
        else:
            draw.rounded_rectangle([x1, y1, x2, y2], radius=obj_w // 3, fill=color_rgb)

    elif shape == "earth":
        # Earth switch — horizontal lines stacked (earth symbol)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        lw = max(2, obj_w // 12)
        lens = [obj_w, int(obj_w * 0.7), int(obj_w * 0.4)]
        for i, l in enumerate(lens):
            ey = y1 + int(obj_h * (0.3 + i * 0.25))
            draw.line([cx - l // 2, ey, cx + l // 2, ey], fill=color_rgb, width=lw)
        # Vertical line
        draw.line([cx, y1, cx, y1 + int(obj_h * 0.3)], fill=color_rgb, width=lw)

    elif shape == "gap":
        # Disconnect — two contacts with gap
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        lw = max(2, obj_h // 8)
        gap_half = obj_w // 6
        draw.line([x1, cy, cx - gap_half, cy], fill=color_rgb, width=lw)
        draw.line([cx + gap_half, cy, x2, cy], fill=color_rgb, width=lw)
        if "closed" in class_name:
            draw.line([cx - gap_half, cy, cx + gap_half, cy], fill=color_rgb, width=lw)
        r = max(4, lw)
        draw.ellipse([cx - gap_half - r, cy - r, cx - gap_half + r, cy + r], fill=color_rgb)
        draw.ellipse([cx + gap_half - r, cy - r, cx + gap_half + r, cy + r], fill=color_rgb)

    elif shape == "panel":
        comp_num = class_name.replace("compartment_", "")
        draw_panel(draw, x1, y1, x2, y2, color_rgb, f"C{comp_num}")

    elif shape.startswith("circle"):
        state_map = {"circle_lit": "lit", "circle_off": "off", "circle_x": "fault"}
        draw_indicator(draw, x1, y1, x2, y2, state_map.get(shape, "lit"), color_rgb)

    # Class label overlay (helps model learn text context)
    short_label = class_name.replace("_", " ").upper()
    label_bg_rgb = hex_to_rgb(vis["label_bg"])
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
    lx = x1
    ly = max(0, y1 - 16)
    label_w = len(short_label) * 7 + 4
    draw.rectangle([lx, ly, lx + label_w, ly + 15], fill=label_bg_rgb)
    draw.text((lx + 2, ly + 1), short_label, fill=(255, 255, 255), font=font)

    # Augmentation
    img = adjust_brightness(img, rng)
    img = add_noise(img, rng)
    img = add_blur(img, rng)

    # Compute YOLO normalised annotation
    cx_n = (x1 + x2) / 2 / img_size
    cy_n = (y1 + y2) / 2 / img_size
    w_n  = (x2 - x1) / img_size
    h_n  = (y2 - y1) / img_size

    # Clip to [0, 1]
    cx_n = max(0.0, min(1.0, cx_n))
    cy_n = max(0.0, min(1.0, cy_n))
    w_n  = max(0.01, min(1.0, w_n))
    h_n  = max(0.01, min(1.0, h_n))

    return img, [class_id, cx_n, cy_n, w_n, h_n]


def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic MOSLock training images")
    p.add_argument("--output",  type=str, default="./dataset",
                   help="Output dataset root directory")
    p.add_argument("--samples", type=int, default=30,
                   help="Synthetic images per class (default: 30, i.e. 750 total)")
    p.add_argument("--imgsz",   type=int, default=640,
                   help="Image size in pixels (default: 640)")
    p.add_argument("--seed",    type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--train_frac", type=float, default=0.8,
                   help="Fraction of images for training set (default: 0.8)")
    p.add_argument("--val_frac",   type=float, default=0.133,
                   help="Fraction for validation (default: 0.133, rest = test)")
    return p.parse_args()


def main():
    args = parse_args()
    rng  = random.Random(args.seed)
    np.random.seed(args.seed)

    out_root = Path(args.output)
    splits   = ["train", "val", "test"]
    for split in splits:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Split indices per class
    n_train = int(args.samples * args.train_frac)
    n_val   = int(args.samples * args.val_frac)
    n_test  = args.samples - n_train - n_val

    total_generated = 0
    manifest = []

    print(f"\n  MOSLock V3 — Synthetic Image Generator")
    print(f"  Classes: {len(CLASS_NAMES)}  ×  {args.samples} samples = {len(CLASS_NAMES) * args.samples} images")
    print(f"  Output:  {out_root.resolve()}")
    print(f"  Split:   train={n_train}  val={n_val}  test={n_test} per class\n")

    for class_id, class_name in enumerate(CLASS_NAMES):
        split_counts = {"train": n_train, "val": n_val, "test": n_test}
        for split, count in split_counts.items():
            for i in range(count):
                # Vary seed per image
                img_rng = random.Random(args.seed + class_id * 10000 + splits.index(split) * 1000 + i)
                np.random.seed(args.seed + class_id * 10000 + splits.index(split) * 1000 + i)

                img, annotation = generate_image(class_id, class_name, args.imgsz, img_rng)

                fname_stem = f"{class_name}_{split}_{i:04d}"
                img_path   = out_root / "images" / split / f"{fname_stem}.jpg"
                lbl_path   = out_root / "labels" / split / f"{fname_stem}.txt"

                img.save(str(img_path), "JPEG", quality=92)

                class_idx, cx, cy, w, h = annotation
                with open(lbl_path, "w") as f:
                    f.write(f"{class_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

                manifest.append({
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "image": str(img_path.relative_to(out_root)),
                    "label": str(lbl_path.relative_to(out_root)),
                })

                total_generated += 1

        print(f"  [{class_id+1:2}/{len(CLASS_NAMES)}] {class_name:<35} "
              f"train={n_train} val={n_val} test={n_test}")

    # Write manifest JSON
    manifest_path = out_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"total": total_generated, "images": manifest}, f, indent=2)

    print(f"\n  ✓ Generated {total_generated} images in {out_root.resolve()}")
    print(f"  ✓ Manifest: {manifest_path}")
    print(f"\n  Next steps:")
    print(f"    1. Review and correct annotations in LabelImg or CVAT")
    print(f"       (bootstrap_annotations.py can pre-annotate real photos)")
    print(f"    2. Copy real annotated images into dataset/images/train/")
    print(f"    3. Run: python train.py --data ../data.yaml")


if __name__ == "__main__":
    main()
