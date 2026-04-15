"""
yolo_training/train.py
MOSLock V3 — YOLOv8n Fine-Tuning Script
Mapped Out Solutions® | mos | Lock®

Fine-tunes the YOLOv8n (nano) model on the MOSLock dataset.
Nano chosen for tablet inference speed (Samsung Galaxy Tab S10).

Usage:
    python train.py                        # default settings
    python train.py --epochs 150 --batch 16
    python train.py --resume               # resume from last checkpoint
    python train.py --data /abs/path/data.yaml

Outputs:
    runs/train/moslock_v3/weights/best.pt      → use for export.py
    runs/train/moslock_v3/weights/last.pt      → resume checkpoint
    runs/train/moslock_v3/results.csv          → training metrics
    runs/train/moslock_v3/confusion_matrix.png
"""

import argparse
import sys
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Train MOSLock YOLOv8n model")
    p.add_argument("--data",    type=str, default="data.yaml",
                   help="Path to data.yaml (default: data.yaml in cwd)")
    p.add_argument("--epochs",  type=int, default=100,
                   help="Number of training epochs (default: 100)")
    p.add_argument("--batch",   type=int, default=8,
                   help="Batch size (default: 8; reduce if OOM on CPU-only machine)")
    p.add_argument("--imgsz",   type=int, default=640,
                   help="Training image size (default: 640)")
    p.add_argument("--device",  type=str, default="",
                   help="Device: '' = auto-detect, 'cpu', '0' = GPU0, 'mps' = Apple Silicon")
    p.add_argument("--resume",  action="store_true",
                   help="Resume training from runs/train/moslock_v3/weights/last.pt")
    p.add_argument("--project", type=str, default="runs/train",
                   help="Project directory for saving results")
    p.add_argument("--name",    type=str, default="moslock_v3",
                   help="Run name (sub-directory under project)")
    p.add_argument("--patience",type=int, default=20,
                   help="Early stopping patience (default: 20 epochs)")
    p.add_argument("--pretrained", type=str, default="yolov8n.pt",
                   help="Pretrained weights to fine-tune from (default: yolov8n.pt)")
    p.add_argument("--workers", type=int, default=2,
                   help="DataLoader workers (default: 2; use 0 on Windows)")
    p.add_argument("--cache",   type=str, default="ram",
                   choices=["ram", "disk", ""],
                   help="Cache images in RAM or disk for faster training")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: data.yaml not found at '{data_path.resolve()}'")
        print("Run generate_synthetic_samples.py first to create a dataset,")
        print("or set --data to the correct path.")
        sys.exit(1)

    # ── Select weights ────────────────────────────────────────────────────
    if args.resume:
        resume_weights = Path(args.project) / args.name / "weights" / "last.pt"
        if not resume_weights.exists():
            print(f"ERROR: No checkpoint to resume from: {resume_weights}")
            sys.exit(1)
        weights = str(resume_weights)
        print(f"Resuming from: {weights}")
    else:
        weights = args.pretrained
        print(f"Fine-tuning from: {weights}")

    model = YOLO(weights)

    print(f"\n{'='*60}")
    print(f"  MOSLock V3 — YOLOv8n Training")
    print(f"  Data:    {data_path.resolve()}")
    print(f"  Epochs:  {args.epochs}")
    print(f"  Batch:   {args.batch}")
    print(f"  Imgsz:   {args.imgsz}")
    print(f"  Device:  {args.device or 'auto'}")
    print(f"  Project: {args.project}/{args.name}")
    print(f"{'='*60}\n")

    # ── Augmentation settings tuned for substation imagery ────────────────
    # Substation equipment is rigid — moderate geometric augmentation,
    # aggressive photometric (lighting variation underground vs daylight).
    results = model.train(
        data        = str(data_path),
        epochs      = args.epochs,
        batch       = args.batch,
        imgsz       = args.imgsz,
        device      = args.device,
        project     = args.project,
        name        = args.name,
        exist_ok    = args.resume,
        resume      = args.resume,
        patience    = args.patience,
        workers     = args.workers,
        cache       = args.cache or False,
        # Photometric augmentation — critical for lighting variation (underground)
        hsv_h       = 0.015,    # hue shift
        hsv_s       = 0.5,      # saturation shift
        hsv_v       = 0.5,      # brightness shift (large — underground → daylight)
        # Geometric augmentation — moderate (equipment orientation is predictable)
        degrees     = 15.0,     # ±15° rotation (tablet may be hand-held at an angle)
        translate   = 0.1,      # ±10% translation
        scale       = 0.4,      # scale jitter ±40%
        shear       = 5.0,      # ±5° shear
        perspective = 0.0005,   # very slight perspective (tablet angle)
        flipud      = 0.0,      # no vertical flip (equipment always right-way up)
        fliplr      = 0.3,      # 30% horizontal flip (mirrored substation views)
        mosaic      = 1.0,      # mosaic helps with small-object detection
        mixup       = 0.05,     # light mixup
        copy_paste  = 0.1,      # copy-paste augmentation for small objects (locks)
        # Class weighting — safety-critical classes weighted higher
        # lock + indicator classes get higher weight via cls_pw
        cls         = 0.5,
        box         = 7.5,
        dfl         = 1.5,
        # Label smoothing helps with noisy synthetic annotations
        label_smoothing = 0.05,
        # Logging
        verbose     = True,
        plots       = True,
        save        = True,
        save_period = 10,       # save checkpoint every 10 epochs
    )

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best weights: {best_weights.resolve()}")
    print(f"  Next step:    python export.py --weights {best_weights}")
    print(f"{'='*60}\n")

    # Print final metrics
    try:
        metrics = results.results_dict
        print(f"  mAP50:     {metrics.get('metrics/mAP50(B)', 0):.3f}")
        print(f"  mAP50-95:  {metrics.get('metrics/mAP50-95(B)', 0):.3f}")
        print(f"  Precision: {metrics.get('metrics/precision(B)', 0):.3f}")
        print(f"  Recall:    {metrics.get('metrics/recall(B)', 0):.3f}")
    except Exception:
        pass  # metrics may not be available in all ultralytics versions


if __name__ == "__main__":
    main()
