"""
yolo_training/inference.py
MOSLock V3 — Single-Image ONNX Inference Module
Mapped Out Solutions® | mos | Lock®

Provides a clean, dependency-minimal inference function that:
  • Loads a YOLOv8n ONNX model via onnxruntime
  • Runs inference on a single image (file path, URL, PIL Image, or numpy array)
  • Returns structured detections: class name, confidence, bounding box
  • Handles pre/post-processing internally (letterbox, NMS)

Usage as module (from moslock_v3.py):
    from inference import MOSLockInference
    engine = MOSLockInference("./models/moslock_yolov8n.onnx", conf_thresh=0.90)
    detections = engine.infer(image)   # returns List[Detection]

Usage from command line:
    python inference.py --model ./models/moslock_yolov8n.onnx --image photo.jpg
    python inference.py --model ./models/moslock_yolov8n.onnx --image photo.jpg --show
"""

from __future__ import annotations
import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np


# ── 25 class names — must match data.yaml order ─────────────────────────
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

# Classes with safety consequences that trigger special handling
CRITICAL_CLASSES = {"indicator_fault", "indicator_lit"}
SAFE_OPEN_CLASSES = {
    "circuit_breaker_open", "isolator_switch_open", "fuse_withdrawn",
    "earth_switch_applied", "disconnect_open",
}
LOCK_CLASSES = {"personal_lock", "permit_lock"}


@dataclass
class Detection:
    """Single detection result from ONNX inference."""
    class_id:    int
    class_name:  str
    confidence:  float
    # Bounding box in pixel coordinates [x1, y1, x2, y2]
    bbox:        Tuple[float, float, float, float]
    # Normalised box [cx, cy, w, h] relative to original image dims
    bbox_norm:   Tuple[float, float, float, float] = field(default_factory=tuple)

    @property
    def is_critical(self) -> bool:
        return self.class_name in CRITICAL_CLASSES

    @property
    def is_lock(self) -> bool:
        return self.class_name in LOCK_CLASSES

    @property
    def is_isolation_open(self) -> bool:
        return self.class_name in SAFE_OPEN_CLASSES

    def to_dict(self) -> dict:
        x1, y1, x2, y2 = self.bbox
        return {
            "class_id":   self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox_x1":    round(x1, 1),
            "bbox_y1":    round(y1, 1),
            "bbox_x2":    round(x2, 1),
            "bbox_y2":    round(y2, 1),
            "is_critical": self.is_critical,
        }


class MOSLockInference:
    """
    ONNX runtime inference engine for MOSLock YOLOv8n.
    Singleton-friendly — load once, call infer() many times.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        conf_thresh: float = 0.90,
        iou_thresh:  float = 0.45,
        imgsz:       int   = 640,
    ):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime not installed. Run: pip install onnxruntime"
            )

        self.model_path  = Path(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh  = iou_thresh
        self.imgsz       = imgsz
        self._session    = None
        self._ort        = ort

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {self.model_path.resolve()}\n"
                "Run export.py to generate the model from trained weights."
            )

        self._load_session()

    def _load_session(self):
        """Load ONNX session — prefer CPU execution for tablet compatibility."""
        providers = self._ort.get_available_providers()
        # Prefer CoreML (iPad) → DirectML (Windows) → CPU
        preferred = []
        if "CoreMLExecutionProvider" in providers:
            preferred.append("CoreMLExecutionProvider")
        if "DirectMLExecutionProvider" in providers:
            preferred.append("DirectMLExecutionProvider")
        preferred.append("CPUExecutionProvider")

        self._session = self._ort.InferenceSession(
            str(self.model_path),
            providers=preferred,
        )
        self._input_name  = self._session.get_inputs()[0].name
        self._input_shape = self._session.get_inputs()[0].shape
        self._num_classes = len(CLASS_NAMES)

    def _load_image(self, source: Union[str, Path, "PIL.Image.Image", np.ndarray]) -> np.ndarray:
        """Load image from any supported source type → RGB numpy array."""
        if isinstance(source, np.ndarray):
            img = source.copy()
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)  # greyscale → RGB
            if img.shape[2] == 4:
                img = img[:, :, :3]  # RGBA → RGB
            return img

        try:
            from PIL import Image as PILImage
            if isinstance(source, PILImage.Image):
                return np.array(source.convert("RGB"))
        except ImportError:
            pass

        import cv2
        path_str = str(source)
        img = cv2.imread(path_str)
        if img is None:
            raise ValueError(f"Could not load image: {path_str}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _letterbox(
        self,
        img:  np.ndarray,
        new_shape: Tuple[int, int] = (640, 640),
        color: Tuple[int, int, int] = (114, 114, 114),
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Resize and pad image to new_shape maintaining aspect ratio.
        Returns: (padded_image, scale_ratio, (pad_w, pad_h))
        """
        h, w = img.shape[:2]
        target_h, target_w = new_shape

        # Scale ratio (new / old)
        r = min(target_h / h, target_w / w)
        new_unpad_w = int(round(w * r))
        new_unpad_h = int(round(h * r))

        pad_w = (target_w - new_unpad_w) / 2
        pad_h = (target_h - new_unpad_h) / 2

        try:
            import cv2
            resized = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(img)
            resized = np.array(pil_img.resize((new_unpad_w, new_unpad_h), PILImage.BILINEAR))

        top    = int(round(pad_h - 0.1))
        bottom = int(round(pad_h + 0.1))
        left   = int(round(pad_w - 0.1))
        right  = int(round(pad_w + 0.1))

        padded = np.full((target_h, target_w, 3), color, dtype=np.uint8)
        padded[top:top + new_unpad_h, left:left + new_unpad_w] = resized

        return padded, r, (pad_w, pad_h)

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Letterbox → normalise [0,1] → NCHW float32."""
        padded, _, _ = self._letterbox(img, (self.imgsz, self.imgsz))
        tensor = padded.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))   # HWC → CHW
        tensor = np.expand_dims(tensor, axis=0)     # → NCHW
        return tensor

    def _nms(
        self,
        boxes:  np.ndarray,
        scores: np.ndarray,
        iou_threshold: float,
    ) -> List[int]:
        """CPU NMS — returns kept indices."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep  = []

        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w    = np.maximum(0.0, xx2 - xx1)
            h    = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            inds  = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def _postprocess(
        self,
        outputs:    np.ndarray,
        orig_shape: Tuple[int, int],
    ) -> List[Detection]:
        """
        Parse YOLOv8 ONNX output → Detection list.
        YOLOv8 ONNX output shape: [1, num_classes+4, num_anchors]
        """
        preds = outputs[0]  # shape: [1, 4+nc, anchors] or [1, anchors, 4+nc]

        # Handle both transposed formats
        if preds.ndim == 3:
            preds = preds[0]           # → [4+nc, anchors] or [anchors, 4+nc]
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T            # → [anchors, 4+nc]

        orig_h, orig_w = orig_shape
        scale_w = self.imgsz / orig_w
        scale_h = self.imgsz / orig_h
        scale   = min(scale_w, scale_h)
        pad_w   = (self.imgsz - orig_w * scale) / 2
        pad_h   = (self.imgsz - orig_h * scale) / 2

        boxes_all    = []
        scores_all   = []
        class_ids_all = []

        for row in preds:
            cx, cy, w, h = row[:4]
            class_scores = row[4:]
            class_id     = int(np.argmax(class_scores))
            confidence   = float(class_scores[class_id])

            if confidence < self.conf_thresh:
                continue

            # Convert cx,cy,w,h → x1,y1,x2,y2 in model-input space
            x1_m = cx - w / 2
            y1_m = cy - h / 2
            x2_m = cx + w / 2
            y2_m = cy + h / 2

            # Map back to original image space
            x1 = (x1_m - pad_w) / scale
            y1 = (y1_m - pad_h) / scale
            x2 = (x2_m - pad_w) / scale
            y2 = (y2_m - pad_h) / scale

            # Clip to image bounds
            x1 = max(0.0, min(float(x1), orig_w))
            y1 = max(0.0, min(float(y1), orig_h))
            x2 = max(0.0, min(float(x2), orig_w))
            y2 = max(0.0, min(float(y2), orig_h))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes_all.append([x1, y1, x2, y2])
            scores_all.append(confidence)
            class_ids_all.append(class_id)

        if not boxes_all:
            return []

        boxes_np  = np.array(boxes_all, dtype=np.float32)
        scores_np = np.array(scores_all, dtype=np.float32)

        keep = self._nms(boxes_np, scores_np, self.iou_thresh)

        detections = []
        for idx in keep:
            x1, y1, x2, y2 = boxes_all[idx]
            cid = class_ids_all[idx]
            cname = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"class_{cid}"
            conf  = scores_all[idx]

            # Normalised box
            cx_n = ((x1 + x2) / 2) / orig_w
            cy_n = ((y1 + y2) / 2) / orig_h
            w_n  = (x2 - x1) / orig_w
            h_n  = (y2 - y1) / orig_h

            detections.append(Detection(
                class_id   = cid,
                class_name = cname,
                confidence = conf,
                bbox       = (x1, y1, x2, y2),
                bbox_norm  = (cx_n, cy_n, w_n, h_n),
            ))

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def infer(
        self,
        source: Union[str, Path, "PIL.Image.Image", np.ndarray],
    ) -> List[Detection]:
        """
        Run inference on a single image.
        Returns list of Detection objects sorted by confidence (highest first).
        """
        img      = self._load_image(source)
        orig_h, orig_w = img.shape[:2]
        tensor   = self._preprocess(img)
        outputs  = self._session.run(None, {self._input_name: tensor})
        return self._postprocess(outputs, (orig_h, orig_w))

    def set_confidence(self, conf: float):
        """Update the confidence threshold at runtime (for slider)."""
        self.conf_thresh = max(0.01, min(0.99, conf))


# ── CLI entry point ────────────────────────────────────────────────────────

def _draw_detections(img_array: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """Draw bboxes on image — returns RGB numpy array."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(img_array).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Colour map per class group
    COLOR_MAP = {
        "personal_lock": "#FF6B35",
        "permit_lock":   "#F5A623",
        "indicator_lit": "#E8392D",
        "indicator_off": "#1ABF74",
        "indicator_fault": "#CC0000",
    }

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = COLOR_MAP.get(det.class_name, "#3ECFCF")
        if det.class_name.startswith("circuit_breaker") or \
           det.class_name.startswith("isolator") or \
           det.class_name.startswith("fuse") or \
           det.class_name.startswith("earth") or \
           det.class_name.startswith("disconnect"):
            color = "#1ABF74" if "open" in det.class_name or "applied" in det.class_name or "withdrawn" in det.class_name else "#E8392D"
        elif det.class_name.startswith("compartment"):
            color = "#3ECFCF"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{det.class_name} {det.confidence:.0%}"
        draw.rectangle([x1, y1 - 18, x1 + len(label) * 7, y1], fill=color)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        except Exception:
            font = ImageFont.load_default()
        draw.text((x1 + 2, y1 - 16), label, fill="white", font=font)

    return np.array(img)


def main():
    p = argparse.ArgumentParser(description="MOSLock V3 — Single image ONNX inference")
    p.add_argument("--model",  type=str, required=True, help="Path to .onnx model")
    p.add_argument("--image",  type=str, required=True, help="Path to image file")
    p.add_argument("--conf",   type=float, default=0.90, help="Confidence threshold")
    p.add_argument("--iou",    type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--imgsz",  type=int,   default=640,  help="Inference image size")
    p.add_argument("--show",   action="store_true",      help="Display result image")
    p.add_argument("--save",   type=str,   default="",   help="Save annotated image to path")
    p.add_argument("--json",   action="store_true",      help="Output JSON to stdout")
    args = p.parse_args()

    engine = MOSLockInference(args.model, args.conf, args.iou, args.imgsz)

    try:
        img_array = engine._load_image(args.image)
    except Exception as e:
        print(f"ERROR loading image: {e}", file=sys.stderr)
        sys.exit(1)

    detections = engine.infer(args.image)

    if args.json:
        import json
        output = {
            "image": args.image,
            "model": args.model,
            "conf_threshold": args.conf,
            "num_detections": len(detections),
            "detections": [d.to_dict() for d in detections],
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\n  Image:      {args.image}")
        print(f"  Model:      {args.model}")
        print(f"  Threshold:  {args.conf:.0%}")
        print(f"  Detections: {len(detections)}\n")
        if not detections:
            print("  (no detections above threshold)")
        for i, det in enumerate(detections, 1):
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            flag = " ⚠ CRITICAL" if det.is_critical else ""
            print(f"  {i:2}. {det.class_name:<30} {det.confidence:.1%}  "
                  f"[{x1},{y1},{x2},{y2}]{flag}")

    if args.show or args.save:
        annotated = _draw_detections(img_array, detections)
        if args.save:
            from PIL import Image
            Image.fromarray(annotated).save(args.save)
            print(f"\n  Saved: {args.save}")
        if args.show:
            try:
                from PIL import Image
                Image.fromarray(annotated).show()
            except Exception:
                import cv2
                import cv2 as cv2_mod
                cv2_mod.imshow("MOSLock Detections", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                cv2_mod.waitKey(0)
                cv2_mod.destroyAllWindows()


if __name__ == "__main__":
    main()
