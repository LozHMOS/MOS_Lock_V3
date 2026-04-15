"""
yolo_training/export.py
MOSLock V3 — Model Export Script
Mapped Out Solutions® | mos | Lock®

Exports trained YOLOv8n .pt weights to:
  • ONNX   → CPU inference via onnxruntime (primary tablet deployment format)
  • TFLite → optional Android on-device inference (requires tensorflow)

Usage:
    python export.py                               # default: best.pt → ONNX
    python export.py --weights runs/train/moslock_v3/weights/best.pt
    python export.py --weights best.pt --tflite    # also export TFLite
    python export.py --weights best.pt --imgsz 320 # smaller for faster inference

Output files:
    ./models/moslock_yolov8n.onnx     (used by moslock_v3.py Live Mode)
    ./models/moslock_yolov8n.tflite   (optional Android deployment)
    ./models/moslock_yolov8n_metadata.json
"""

import argparse
import json
import sys
import shutil
from pathlib import Path
from datetime import datetime


def parse_args():
    p = argparse.ArgumentParser(description="Export MOSLock YOLOv8n to ONNX + TFLite")
    p.add_argument("--weights", type=str,
                   default="runs/train/moslock_v3/weights/best.pt",
                   help="Path to trained .pt weights file")
    p.add_argument("--imgsz",   type=int, default=640,
                   help="Export image size (default: 640; use 320 for faster tablet inference)")
    p.add_argument("--tflite",  action="store_true",
                   help="Also export TFLite (requires tensorflow installed)")
    p.add_argument("--output",  type=str, default="./models",
                   help="Output directory for exported models")
    p.add_argument("--dynamic", action="store_true",
                   help="Enable dynamic batch size in ONNX (default: fixed batch=1)")
    p.add_argument("--simplify", action="store_true", default=True,
                   help="Simplify ONNX graph (default: True)")
    p.add_argument("--half",    action="store_true",
                   help="Export FP16 (only useful with GPU inference; not for CPU/tablet)")
    return p.parse_args()


def export_onnx(model, args, output_dir: Path) -> Path:
    """Export to ONNX format for onnxruntime inference on tablet."""
    print("\n── Exporting ONNX ──────────────────────────────────────")
    onnx_path = model.export(
        format   = "onnx",
        imgsz    = args.imgsz,
        dynamic  = args.dynamic,
        simplify = args.simplify,
        half     = args.half,
        opset    = 17,           # opset 17 = supported by onnxruntime 1.16+
    )
    onnx_path = Path(onnx_path)
    dest = output_dir / "moslock_yolov8n.onnx"
    shutil.copy2(onnx_path, dest)
    print(f"  ✓ ONNX saved: {dest.resolve()}")
    print(f"  Size: {dest.stat().st_size / 1024 / 1024:.1f} MB")
    return dest


def export_tflite(model, args, output_dir: Path) -> Path:
    """Export to TFLite for Android on-device inference."""
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        print("  WARNING: tensorflow not installed. Skipping TFLite export.")
        print("  Install: pip install tensorflow==2.16.2")
        return None

    print("\n── Exporting TFLite ─────────────────────────────────────")
    tflite_path = model.export(
        format = "tflite",
        imgsz  = args.imgsz,
        half   = False,     # INT8 / FP16 requires calibration data — use FP32 for now
    )
    tflite_path = Path(tflite_path)
    dest = output_dir / "moslock_yolov8n.tflite"
    shutil.copy2(tflite_path, dest)
    print(f"  ✓ TFLite saved: {dest.resolve()}")
    print(f"  Size: {dest.stat().st_size / 1024 / 1024:.1f} MB")
    return dest


def write_metadata(output_dir: Path, args, onnx_dest, tflite_dest):
    """Write a JSON metadata file alongside the exported models."""
    from yolo_training_data_yaml import CLASS_NAMES  # lazy import
    class_names = CLASS_NAMES if CLASS_NAMES else list(range(25))

    try:
        from ultralytics import YOLO
        m = YOLO(args.weights)
        class_names = list(m.names.values())
    except Exception:
        pass

    metadata = {
        "product":      "MOSLock",
        "version":      "3.0",
        "model":        "YOLOv8n",
        "company":      "Mapped Out Solutions®",
        "exported_at":  datetime.utcnow().isoformat() + "Z",
        "source_weights": str(Path(args.weights).resolve()),
        "imgsz":        args.imgsz,
        "num_classes":  25,
        "class_names":  class_names,
        "onnx_file":    str(onnx_dest.name) if onnx_dest else None,
        "tflite_file":  str(tflite_dest.name) if tflite_dest else None,
        "runtime_target": "onnxruntime (CPU)",
        "tablet_target":  "Samsung Galaxy Tab S10 (Android)",
        "standards":    [
            "Glencore GCAA Fatal Hazard Protocol 7",
            "AS/NZS 4836:2023",
            "AS 4871.1",
        ],
        "confidence_threshold_default":  0.90,
        "confidence_threshold_minimum":  0.50,
        "notes": (
            "indicator_fault class (id=24) triggers STOP regardless of confidence threshold. "
            "Do not raise the threshold for this class above 0.70."
        ),
    }

    meta_path = output_dir / "moslock_yolov8n_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  ✓ Metadata: {meta_path.resolve()}")


def validate_onnx(onnx_path: Path):
    """Basic ONNX graph validation."""
    try:
        import onnx
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        print("  ✓ ONNX graph validation passed")
    except ImportError:
        print("  (onnx package not installed — skipping graph validation)")
    except Exception as e:
        print(f"  WARNING: ONNX validation failed: {e}")


def test_onnx_inference(onnx_path: Path, imgsz: int):
    """Run a dummy inference to confirm the ONNX model loads correctly."""
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("  (onnxruntime/numpy not installed — skipping inference test)")
        return

    print("  Running dummy inference test...")
    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    input_name  = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    # Replace dynamic dimensions with fixed values
    shape = [1 if (isinstance(d, str) or d is None) else d for d in input_shape]
    shape[2] = imgsz
    shape[3] = imgsz
    dummy = np.random.randn(*shape).astype(np.float32)
    outputs = sess.run(None, {input_name: dummy})
    print(f"  ✓ Inference test passed — output shape: {outputs[0].shape}")


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"ERROR: Weights file not found: {weights_path.resolve()}")
        print("Run train.py first, then export.")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  MOSLock V3 — Model Export")
    print(f"  Source: {weights_path.resolve()}")
    print(f"  Output: {output_dir.resolve()}")
    print(f"  Format: ONNX{' + TFLite' if args.tflite else ''}")
    print(f"  imgsz:  {args.imgsz}")
    print(f"{'='*60}")

    model = YOLO(str(weights_path))

    onnx_dest   = export_onnx(model, args, output_dir)
    tflite_dest = export_tflite(model, args, output_dir) if args.tflite else None

    validate_onnx(onnx_dest)
    test_onnx_inference(onnx_dest, args.imgsz)
    write_metadata(output_dir, args, onnx_dest, tflite_dest)

    print(f"\n{'='*60}")
    print(f"  Export complete!")
    print(f"  Place '{onnx_dest.name}' in the 'models/' folder next to moslock_v3.py")
    print(f"  Set model path in Live Mode sidebar: ./models/moslock_yolov8n.onnx")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
