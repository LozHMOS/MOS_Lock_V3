"""
yolo_training/bootstrap_annotations.py
MOSLock V3 — Claude Vision API Pre-Annotation Tool
Mapped Out Solutions® | mos | Lock®

Uses the Anthropic Claude claude-opus-4-5 vision API to automatically pre-annotate
real substation equipment photos in YOLO .txt format for human review in
CVAT or LabelImg.

Workflow:
  1. Place real photos in --input folder
  2. Run this script — Claude analyses each photo and outputs YOLO labels
  3. Open labels in LabelImg/CVAT alongside the original images
  4. A human reviewer accepts, adjusts, or rejects each bounding box
  5. Reviewed images become ground-truth training data

Output:
    <output_dir>/
    ├── <image_stem>.txt        YOLO label files
    ├── <image_stem>_review.json  Claude's full annotations + confidence notes
    └── bootstrap_report.txt    Summary of what was found per image

Usage:
    export ANTHROPIC_API_KEY=your_key_here
    python bootstrap_annotations.py --input ./real_photos --output ./auto_labels
    python bootstrap_annotations.py --input ./real_photos --output ./auto_labels --dry-run
"""

from __future__ import annotations
import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Class names — must match data.yaml ──────────────────────────────────────
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

CLASS_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

ANNOTATION_PROMPT = """You are an expert electrical safety system helping to annotate
substation equipment photos for a computer vision model (MOSLock).

The image shows a mobile substation (BST008 type, Glencore Coal Assets Australia).
Your task is to identify and locate all visible equipment from the following 25 classes:

LOCKS:
- personal_lock: individual worker padlock (colour-coded: typically red, blue, or yellow)
- permit_lock: permit holder's lock (larger, numbered, attached to lock box)

ISOLATION POINTS (one of two states each):
- circuit_breaker_open / circuit_breaker_closed
- isolator_switch_open / isolator_switch_closed
- fuse_withdrawn / fuse_inserted
- earth_switch_applied / earth_switch_removed
- disconnect_open / disconnect_closed

COMPARTMENTS (large panel zones):
- compartment_1: HV Incoming (high voltage incoming section)
- compartment_2: Transformer
- compartment_3: LV Outgoing
- compartment_4: Control
- compartment_5: Auxiliary
- compartment_6: Bus Section
- compartment_7: Metering
- compartment_8: Protection Relay
- compartment_9: Cable Termination
- compartment_10: Earth Bar

LIVE LINE INDICATORS (small round indicators):
- indicator_lit: illuminated (voltage present — DANGER)
- indicator_off: dark/unlit (no voltage — SAFE)
- indicator_fault: damaged or obscured indicator (requires manual check)

Return ONLY a JSON object in this exact format:
{
  "annotations": [
    {
      "class_name": "class_name_exactly_as_listed",
      "bbox_norm": [cx, cy, width, height],
      "confidence_note": "brief note on why you classified it this way",
      "review_flag": false
    }
  ],
  "image_notes": "overall notes about image quality, lighting, what's visible",
  "uncertain_regions": ["description of any areas you're unsure about"]
}

RULES for bbox_norm [cx, cy, width, height]:
- All values between 0.0 and 1.0 (normalised relative to image dimensions)
- cx, cy = centre of bounding box
- width, height = box dimensions
- Compartments are large boxes (typically width > 0.15, height > 0.20)
- Locks are small boxes (typically width < 0.08, height < 0.10)
- Indicators are small circles (typically width and height < 0.06)
- Set review_flag: true if you are uncertain about the class or location

Be conservative — only annotate what you can see clearly. Omit objects you cannot
confirm. It is better to miss an object than to create a false positive.

IMPORTANT SAFETY NOTE: This system is used to verify electrical isolation before
workers access high voltage equipment. Annotation accuracy directly affects worker safety.
"""


def load_image_b64(image_path: Path) -> Tuple[str, str]:
    """Load image file → base64 string + media type."""
    suffix = image_path.suffix.lower()
    media_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
    }
    media_type = media_map.get(suffix, "image/jpeg")
    with open(image_path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("utf-8")
    return b64, media_type


def call_claude_vision(
    client,
    image_path: Path,
    max_retries: int = 3,
) -> Optional[Dict]:
    """Call Claude claude-opus-4-5 API with the image and annotation prompt."""
    b64_data, media_type = load_image_b64(image_path)

    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type":       "base64",
                                    "media_type": media_type,
                                    "data":       b64_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": ANNOTATION_PROMPT,
                            }
                        ],
                    }
                ],
            )
            response_text = message.content[0].text.strip()

            # Extract JSON from response (handle markdown code blocks)
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            data = json.loads(response_text)
            return data

        except json.JSONDecodeError as e:
            print(f"    WARNING: JSON parse error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff
        except Exception as e:
            err_str = str(e)
            if "rate_limit" in err_str.lower() or "529" in err_str:
                wait = 30 * (attempt + 1)
                print(f"    Rate limited — waiting {wait}s before retry {attempt + 1}/{max_retries}")
                time.sleep(wait)
            else:
                print(f"    ERROR calling Claude API: {e}")
                return None

    return None


def annotations_to_yolo(annotations: List[Dict], image_path: Path) -> List[str]:
    """Convert Claude annotation dicts → YOLO label file lines."""
    lines = []
    for ann in annotations:
        class_name = ann.get("class_name", "").strip()
        if class_name not in CLASS_INDEX:
            print(f"      SKIP: unknown class '{class_name}'")
            continue

        bbox = ann.get("bbox_norm", [])
        if len(bbox) != 4:
            print(f"      SKIP: invalid bbox for {class_name}: {bbox}")
            continue

        cx, cy, w, h = bbox
        if not all(0.0 <= v <= 1.0 for v in [cx, cy, w, h]):
            print(f"      SKIP: bbox values out of range for {class_name}: {bbox}")
            continue

        if w < 0.005 or h < 0.005:
            print(f"      SKIP: bbox too small for {class_name}: {bbox}")
            continue

        class_id = CLASS_INDEX[class_name]
        review   = ann.get("review_flag", False)
        note     = ann.get("confidence_note", "")

        lines.append(
            f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
            f"  # {class_name}{' [REVIEW]' if review else ''} — {note}"
        )

    return lines


def process_image(
    image_path: Path,
    output_dir: Path,
    client,
    dry_run:    bool = False,
) -> Dict:
    """Process one image: call Claude, write YOLO label + review JSON."""
    print(f"  Processing: {image_path.name}")

    if dry_run:
        print(f"    [DRY RUN] Would call Claude API for {image_path.name}")
        return {"image": str(image_path), "status": "dry_run", "annotations": []}

    result = call_claude_vision(client, image_path)

    if result is None:
        print(f"    FAILED: No annotations generated for {image_path.name}")
        return {"image": str(image_path), "status": "failed", "annotations": []}

    annotations = result.get("annotations", [])
    image_notes = result.get("image_notes", "")
    uncertain   = result.get("uncertain_regions", [])

    review_flags = sum(1 for a in annotations if a.get("review_flag", False))
    print(f"    Found {len(annotations)} annotations "
          f"({review_flags} flagged for review)")
    if image_notes:
        print(f"    Notes: {image_notes[:120]}")

    # Write YOLO label file
    label_path = output_dir / f"{image_path.stem}.txt"
    yolo_lines = annotations_to_yolo(annotations, image_path)
    with open(label_path, "w") as f:
        f.write(f"# Auto-annotated by MOSLock bootstrap_annotations.py\n")
        f.write(f"# Image: {image_path.name}\n")
        f.write(f"# HUMAN REVIEW REQUIRED before use in training\n")
        f.write(f"# Claude notes: {image_notes}\n")
        f.write(f"# Uncertain regions: {'; '.join(uncertain)}\n")
        f.write(f"# Format: class_id cx cy w h  (YOLO normalised)\n\n")
        # Write clean lines (without comments) for LabelImg compatibility
        for line in yolo_lines:
            clean = line.split("  #")[0].strip()
            f.write(f"{clean}\n")

    # Write full review JSON (with notes and flags) for annotator use
    review_path = output_dir / f"{image_path.stem}_review.json"
    review_data = {
        "source_image":     str(image_path),
        "label_file":       str(label_path),
        "claude_model":     "claude-opus-4-5",
        "image_notes":      image_notes,
        "uncertain_regions": uncertain,
        "annotations":      annotations,
        "yolo_lines":       yolo_lines,
    }
    with open(review_path, "w") as f:
        json.dump(review_data, f, indent=2)

    return {
        "image":        str(image_path),
        "status":       "success",
        "annotations":  annotations,
        "num_detections": len(annotations),
        "review_flags": review_flags,
        "label_path":   str(label_path),
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Pre-annotate real substation photos using Claude Vision API"
    )
    p.add_argument("--input",   type=str, required=True,
                   help="Directory containing real equipment photos (.jpg/.png)")
    p.add_argument("--output",  type=str, default="./auto_labels",
                   help="Output directory for YOLO labels (default: ./auto_labels)")
    p.add_argument("--api-key", type=str, default="",
                   help="Anthropic API key (default: reads ANTHROPIC_API_KEY env var)")
    p.add_argument("--dry-run", action="store_true",
                   help="Parse images without calling API (test mode)")
    p.add_argument("--delay",   type=float, default=1.5,
                   help="Seconds to wait between API calls (default: 1.5)")
    p.add_argument("--ext",     type=str, default="jpg,jpeg,png,webp",
                   help="Image file extensions to process (comma-separated)")
    return p.parse_args()


def main():
    args = parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir.resolve()}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather image files
    extensions = [f".{e.strip().lower()}" for e in args.ext.split(",")]
    image_files = []
    for ext in extensions:
        image_files.extend(sorted(input_dir.glob(f"*{ext}")))
        image_files.extend(sorted(input_dir.glob(f"*{ext.upper()}")))
    image_files = sorted(set(image_files))

    if not image_files:
        print(f"No image files found in {input_dir.resolve()}")
        print(f"Looking for extensions: {extensions}")
        sys.exit(1)

    print(f"\n  MOSLock V3 — Bootstrap Annotations")
    print(f"  Input:  {input_dir.resolve()}")
    print(f"  Output: {output_dir.resolve()}")
    print(f"  Images: {len(image_files)}")
    print(f"  Model:  claude-opus-4-5\n")

    # Initialise Claude client
    client = None
    if not args.dry_run:
        try:
            import anthropic
        except ImportError:
            print("ERROR: anthropic not installed. Run: pip install anthropic")
            sys.exit(1)

        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("ERROR: No API key provided.")
            print("Set ANTHROPIC_API_KEY environment variable or use --api-key flag.")
            sys.exit(1)

        client = anthropic.Anthropic(api_key=api_key)

    # Process each image
    results = []
    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}]")
        r = process_image(img_path, output_dir, client, dry_run=args.dry_run)
        results.append(r)
        if not args.dry_run and i < len(image_files):
            time.sleep(args.delay)

    # Write summary report
    report_path = output_dir / "bootstrap_report.txt"
    n_success = sum(1 for r in results if r["status"] == "success")
    n_failed  = sum(1 for r in results if r["status"] == "failed")
    total_ann = sum(r.get("num_detections", 0) for r in results)
    total_rev = sum(r.get("review_flags", 0) for r in results)

    with open(report_path, "w") as f:
        f.write("MOSLock V3 — Bootstrap Annotation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Images processed: {len(image_files)}\n")
        f.write(f"  Success:        {n_success}\n")
        f.write(f"  Failed:         {n_failed}\n")
        f.write(f"Total annotations: {total_ann}\n")
        f.write(f"Review flags:      {total_rev}\n\n")
        f.write("Per-image summary:\n")
        for r in results:
            img = Path(r["image"]).name
            status = r["status"]
            n = r.get("num_detections", 0)
            rv = r.get("review_flags", 0)
            f.write(f"  {img:<40} {status:<10} {n:3} annotations "
                    f"({rv} for review)\n")
        f.write("\nNext steps:\n")
        f.write("  1. Open images + .txt files in LabelImg (YOLO format)\n")
        f.write("     pip install labelImg && labelImg\n")
        f.write("  2. Review all [REVIEW]-flagged boxes first\n")
        f.write("  3. Check *_review.json for Claude's uncertainty notes\n")
        f.write("  4. Move approved annotations to dataset/labels/train/\n")
        f.write("  5. Run python train.py\n")

    print(f"\n{'='*60}")
    print(f"  Bootstrap complete!")
    print(f"  {n_success}/{len(image_files)} images annotated")
    print(f"  {total_ann} total annotations ({total_rev} flagged for review)")
    print(f"  Report: {report_path.resolve()}")
    print(f"\n  ⚠ HUMAN REVIEW REQUIRED before using annotations in training")
    print(f"    Open in LabelImg: pip install labelImg && labelImg")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
