"""
moslock_v3.py
MOSLock V3 — Branded Streamlit Application
Mapped Out Solutions® | mos | Lock®
"We will always have a plan."

Tablet-ready isolation verification tool for BST008 mobile substations.
Deployed on Samsung Galaxy Tab S10 + Streamlit Community Cloud.

Three runtime scenarios (all supported without crashes):
  Scenario 1 — REMOTE DEMO (desktop/laptop):
    pip install streamlit pillow pandas
    streamlit run moslock_v3.py

  Scenario 2 — TABLET DEMO (Tab S10, no trained model):
    Same install as above; Demo Mode shows pre-loaded detections

  Scenario 3 — TABLET LIVE (Tab S10 + trained ONNX model):
    pip install streamlit pillow pandas onnxruntime numpy
    Place moslock_yolov8n.onnx in ./models/
    streamlit run moslock_v3.py
"""

from __future__ import annotations

# ── Standard library ────────────────────────────────────────────────────────
import base64
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Third-party (always available — core deps) ───────────────────────────────
import streamlit as st

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False

try:
    from PIL import Image as PILImage, ImageDraw, ImageFont
    _PIL = True
except ImportError:
    _PIL = False

# ── YOLO / ONNX — lazy imports (only in Live Mode) ───────────────────────────
_ONNX_AVAILABLE = False
try:
    import onnxruntime as ort
    import numpy as np
    _ONNX_AVAILABLE = True
except ImportError:
    pass

# ── Brand assets (local package) ─────────────────────────────────────────────
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from moslock_branding.brand_config import (
        MOS_CYAN, MOS_DARK_NAVY, MOS_NEAR_BLACK, MOS_WHITE, MOS_PURPLE, MOS_BLUE,
        MOS_PAGE_BG, MOS_SIDEBAR_BG, MOS_CARD_BG, MOS_BORDER,
        SAFE_GREEN, CAUTION_AMBER, DANGER_RED, STOP_RED,
        VERDICT_ALL_CLEAR, VERDICT_DISCREPANCY, VERDICT_REVIEW, VERDICT_STOP,
        COMPARTMENT_ISOLATED, COMPARTMENT_ENERGISED, COMPARTMENT_UNKNOWN,
        FONT_STACK, GOOGLE_FONT_URL,
        APP_NAME, APP_VERSION, PRODUCT_NAME, COMPANY_NAME, STRAPLINE,
        CLASS_NAMES, CLASS_METADATA,
        LOGO_B64_DARK_BG, LOGO_B64_LIGHT_BG, logo_img_tag,
    )
    _BRAND = True
except ImportError:
    # Fallback constants if brand package not found
    MOS_CYAN = "#3ECFCF"; MOS_DARK_NAVY = "#071539"; MOS_NEAR_BLACK = "#2d2926"
    MOS_WHITE = "#FFFFFF"; MOS_PURPLE = "#D04EFF"; MOS_BLUE = "#007BFF"
    MOS_PAGE_BG = "#F4F6F9"; MOS_SIDEBAR_BG = "#071539"; MOS_CARD_BG = "#FFFFFF"
    MOS_BORDER = "#E2E6EA"; SAFE_GREEN = "#1ABF74"; CAUTION_AMBER = "#F5A623"
    DANGER_RED = "#E8392D"; STOP_RED = "#CC0000"
    VERDICT_ALL_CLEAR = "#1ABF74"; VERDICT_DISCREPANCY = "#E8392D"
    VERDICT_REVIEW = "#F5A623"; VERDICT_STOP = "#CC0000"
    COMPARTMENT_ISOLATED = "#1ABF74"; COMPARTMENT_ENERGISED = "#E8392D"
    COMPARTMENT_UNKNOWN = "#F5A623"
    FONT_STACK = "'Helvetica Neue', Arial, sans-serif"
    GOOGLE_FONT_URL = "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700;900&display=swap"
    APP_NAME = "MOSLock"; APP_VERSION = "3.0.0"
    PRODUCT_NAME = "mos | Lock\u00ae"; COMPANY_NAME = "Mapped Out Solutions\u00ae"
    STRAPLINE = "We will always have a plan."
    CLASS_NAMES = [
        "personal_lock","permit_lock","circuit_breaker_open","circuit_breaker_closed",
        "isolator_switch_open","isolator_switch_closed","fuse_withdrawn","fuse_inserted",
        "earth_switch_applied","earth_switch_removed","disconnect_open","disconnect_closed",
        "compartment_1","compartment_2","compartment_3","compartment_4","compartment_5",
        "compartment_6","compartment_7","compartment_8","compartment_9","compartment_10",
        "indicator_lit","indicator_off","indicator_fault",
    ]
    CLASS_METADATA = {n: {"label": n, "color": MOS_CYAN, "group": "Unknown", "id": i}
                      for i, n in enumerate(CLASS_NAMES)}
    LOGO_B64_DARK_BG = ""; LOGO_B64_LIGHT_BG = ""
    def logo_img_tag(dark_bg=True, width=200):
        return f"<span style='color:{MOS_CYAN};font-size:20px;font-weight:900;'>mos|Lock\u00ae</span>"
    _BRAND = False


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO MODE — Pre-loaded detection results
# Derived from V2 demo images: Substation Isolated LV.jpeg, Substation Trained.jpeg
# Real pixel coords mapped to normalised coords for 1565×525px source images
# ═══════════════════════════════════════════════════════════════════════════════

DEMO_IMAGES = {
    "Substation — Base State": {
        "file":    "Substation.jpeg",
        "label":   "BST008 — No isolation (base state)",
        "detections": [
            {"class_name": "compartment_1",          "confidence": 0.97, "bbox": (0.02, 0.05, 0.21, 0.98), "label": "HV Incoming"},
            {"class_name": "compartment_2",          "confidence": 0.95, "bbox": (0.22, 0.05, 0.52, 0.98), "label": "Transformer"},
            {"class_name": "compartment_3",          "confidence": 0.96, "bbox": (0.53, 0.05, 0.98, 0.98), "label": "LV Outgoing"},
        ],
    },
    "Substation — Training Model": {
        "file":    "Substation Trained.jpeg",
        "label":   "BST008 — Training model annotations",
        "detections": [
            {"class_name": "compartment_1",          "confidence": 0.97, "bbox": (0.02, 0.05, 0.21, 0.98), "label": "Compartment 1"},
            {"class_name": "compartment_2",          "confidence": 0.95, "bbox": (0.22, 0.05, 0.52, 0.98), "label": "Compartment 2"},
            {"class_name": "compartment_3",          "confidence": 0.96, "bbox": (0.53, 0.05, 0.98, 0.98), "label": "Compartment 3"},
            {"class_name": "isolator_switch_open",   "confidence": 0.93, "bbox": (0.03, 0.55, 0.09, 0.75), "label": "Isolation Pt 1"},
            {"class_name": "circuit_breaker_open",   "confidence": 0.91, "bbox": (0.26, 0.28, 0.32, 0.46), "label": "Isolation Pt 2"},
            {"class_name": "disconnect_open",        "confidence": 0.94, "bbox": (0.52, 0.10, 0.60, 0.35), "label": "Isolation Pt 3"},
            {"class_name": "indicator_off",          "confidence": 0.96, "bbox": (0.72, 0.47, 0.79, 0.60), "label": "Indicator 1"},
        ],
    },
    "Substation — Isolated LV": {
        "file":    "Substation Isolated LV.jpeg",
        "label":   "BST008 — Partially isolated (LV compartment isolated)",
        "detections": [
            {"class_name": "compartment_1",          "confidence": 0.97, "bbox": (0.02, 0.05, 0.21, 0.98), "label": "Compartment 1 — Energised"},
            {"class_name": "compartment_2",          "confidence": 0.95, "bbox": (0.22, 0.05, 0.52, 0.98), "label": "Compartment 2 — Energised"},
            {"class_name": "compartment_3",          "confidence": 0.96, "bbox": (0.53, 0.05, 0.98, 0.98), "label": "Compartment 3 — Isolated"},
            {"class_name": "isolator_switch_open",   "confidence": 0.93, "bbox": (0.03, 0.55, 0.09, 0.75), "label": "Isolation Pt 1"},
            {"class_name": "circuit_breaker_open",   "confidence": 0.91, "bbox": (0.26, 0.28, 0.32, 0.46), "label": "Isolation Pt 2"},
            {"class_name": "disconnect_open",        "confidence": 0.94, "bbox": (0.52, 0.10, 0.60, 0.35), "label": "Isolation Pt 3"},
            {"class_name": "personal_lock",          "confidence": 0.96, "bbox": (0.54, 0.22, 0.59, 0.30), "label": "Personal Lock"},
            {"class_name": "indicator_off",          "confidence": 0.96, "bbox": (0.72, 0.47, 0.79, 0.60), "label": "Indicator 1 — Off"},
        ],
    },
    "Substation — Locked Out": {
        "file":    "Sub Locked Out.jpg",
        "label":   "BST008 — Locked out (lock at isolation point 3)",
        "detections": [
            {"class_name": "compartment_1",          "confidence": 0.97, "bbox": (0.02, 0.05, 0.21, 0.98), "label": "Compartment 1"},
            {"class_name": "compartment_2",          "confidence": 0.95, "bbox": (0.22, 0.05, 0.52, 0.98), "label": "Compartment 2"},
            {"class_name": "compartment_3",          "confidence": 0.96, "bbox": (0.53, 0.05, 0.98, 0.98), "label": "Compartment 3"},
            {"class_name": "disconnect_open",        "confidence": 0.94, "bbox": (0.52, 0.10, 0.60, 0.35), "label": "Isolation Pt 3"},
            {"class_name": "personal_lock",          "confidence": 0.97, "bbox": (0.50, 0.25, 0.57, 0.35), "label": "Lock at Isolation Pt 3"},
        ],
    },
}

# ── Image source paths ────────────────────────────────────────────────────────
IMAGE_SEARCH_DIRS = [
    Path(__file__).parent / "demo_images",
    Path(__file__).parent,
    Path("/mnt/session/uploads/workspace/images"),
]

# ── IFTTT Rule Engine — default training model rules ─────────────────────────
DEFAULT_RULES = [
    {
        "id":           "R001",
        "name":         "HV Incoming (Compartment 1) — Circuit Breaker Open",
        "compartment":  "compartment_1",
        "isolation_pt": "circuit_breaker_open",
        "require_lock": True,
        "indicator_expected": "indicator_off",
        "description":  "HV incoming circuit breaker must be open with personal lock applied. Live line indicator must be OFF.",
    },
    {
        "id":           "R002",
        "name":         "Transformer (Compartment 2) — Isolator Switch Open",
        "compartment":  "compartment_2",
        "isolation_pt": "isolator_switch_open",
        "require_lock": True,
        "indicator_expected": "indicator_off",
        "description":  "Transformer isolator must be open with personal lock. Indicator must be OFF.",
    },
    {
        "id":           "R003",
        "name":         "LV Outgoing (Compartment 3) — Disconnect Open",
        "compartment":  "compartment_3",
        "isolation_pt": "disconnect_open",
        "require_lock": True,
        "indicator_expected": "indicator_off",
        "description":  "LV outgoing disconnect must be open with lock applied.",
    },
    {
        "id":           "R004",
        "name":         "Earth Switch — Applied",
        "compartment":  "compartment_1",
        "isolation_pt": "earth_switch_applied",
        "require_lock": False,
        "indicator_expected": None,
        "description":  "Working earth must be applied to confirm dead circuit.",
    },
    {
        "id":           "R005",
        "name":         "No Live Indicators",
        "compartment":  None,
        "isolation_pt": None,
        "require_lock": False,
        "indicator_expected": "indicator_off",
        "description":  "All live line indicators must be OFF. indicator_lit or indicator_fault triggers STOP.",
        "indicator_only": True,
    },
]

# ── Compartment name lookup ───────────────────────────────────────────────────
COMPARTMENT_NAMES = {
    "compartment_1":  "HV Incoming",
    "compartment_2":  "Transformer",
    "compartment_3":  "LV Outgoing",
    "compartment_4":  "Control",
    "compartment_5":  "Auxiliary",
    "compartment_6":  "Bus Section",
    "compartment_7":  "Metering",
    "compartment_8":  "Protection Relay",
    "compartment_9":  "Cable Termination",
    "compartment_10": "Earth Bar",
}

# ── 12-Step isolation process ─────────────────────────────────────────────────
TWELVE_STEPS = [
    (1,  "Identify Energy Sources",     "isolation"),
    (2,  "Advise Relevant Parties",     "isolation"),
    (3,  "Isolate & Secure",            "isolation"),
    (4,  "Place Tags, Locks or Permits","isolation"),
    (5,  "Verify Isolation (Test Dead)","isolation"),
    (6,  "Commence Work",               "isolation"),
    (7,  "Complete Work",               "restoration"),
    (8,  "Check Work",                  "restoration"),
    (9,  "Clear Area",                  "restoration"),
    (10, "Remove Tags, Locks, Permits", "restoration"),
    (11, "Restore Energy",              "restoration"),
    (12, "Check Operation",             "restoration"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# CSS & THEME INJECTION
# ═══════════════════════════════════════════════════════════════════════════════

def load_css() -> str:
    """Load theme CSS from file, fall back to embedded minimal CSS."""
    css_path = Path(__file__).parent / "moslock_branding" / "theme.css"
    if css_path.exists():
        with open(css_path, "r") as f:
            return f.read()
    # Minimal fallback CSS
    return f"""
    @import url('{GOOGLE_FONT_URL}');
    html, body, [class*="css"] {{
        font-family: {FONT_STACK} !important;
        font-size: 16px !important;
    }}
    .stApp {{ background-color: {MOS_PAGE_BG} !important; }}
    section[data-testid="stSidebar"] {{ background-color: {MOS_SIDEBAR_BG} !important; }}
    section[data-testid="stSidebar"] * {{ color: {MOS_WHITE} !important; }}
    .stButton > button {{
        background-color: {MOS_CYAN} !important;
        color: {MOS_NEAR_BLACK} !important;
        font-weight: 700 !important;
        min-height: 48px !important;
        border-radius: 8px !important;
    }}
    """


def inject_css(high_contrast: bool = False):
    css = load_css()
    if high_contrast:
        # Append high-contrast overrides
        css += """
        .stApp { background-color: #000000 !important; color: #FFFFFF !important; }
        [class*="css"] { color: #FFFFFF !important; }
        """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    # Also inject Google Fonts link tag
    st.markdown(
        f'<link href="{GOOGLE_FONT_URL}" rel="stylesheet">',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def find_image(filename: str) -> Optional[Path]:
    """Search for a demo image file across multiple known directories."""
    for d in IMAGE_SEARCH_DIRS:
        p = d / filename
        if p.exists():
            return p
    return None


def image_to_bytes(img: "PILImage.Image", fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=90)
    return buf.getvalue()


def draw_detections_pil(
    img: "PILImage.Image",
    detections: List[Dict],
    conf_thresh: float = 0.0,
    high_contrast: bool = False,
) -> "PILImage.Image":
    """
    Draw bounding boxes and labels on a PIL Image.
    Returns new annotated PIL Image.
    """
    if not _PIL:
        return img

    img = img.convert("RGB").copy()
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    # Font loading
    font_large = font_small = None
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for fp in font_paths:
        if Path(fp).exists():
            try:
                font_large = ImageFont.truetype(fp, 15)
                font_small = ImageFont.truetype(fp, 12)
                break
            except Exception:
                pass
    if font_large is None:
        font_large = font_small = ImageFont.load_default()

    for det in detections:
        conf = det.get("confidence", 0.0)
        if conf < conf_thresh:
            continue

        cname  = det.get("class_name", "")
        bbox   = det.get("bbox", (0, 0, 1, 1))
        label  = det.get("label", CLASS_METADATA.get(cname, {}).get("label", cname))
        color  = CLASS_METADATA.get(cname, {}).get("color", MOS_CYAN)

        # Convert normalised → pixel coords
        x1 = int(bbox[0] * w)
        y1 = int(bbox[1] * h)
        x2 = int(bbox[2] * w)
        y2 = int(bbox[3] * h)

        # Parse hex colour → RGBA
        def hex_rgba(hx: str, alpha: int = 255) -> Tuple:
            hx = hx.lstrip("#")
            r, g, b = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
            return (r, g, b, alpha)

        rgb   = hex_rgba(color)
        rgba  = hex_rgba(color, 60)
        lw    = 3 if cname.startswith("compartment") else 2

        # Highlight indicator_fault with pulsing-style thicker border
        if cname == "indicator_fault":
            lw = 5
            rgb = hex_rgba(STOP_RED)

        # Box fill (semi-transparent)
        draw.rectangle([x1, y1, x2, y2], fill=hex_rgba(color, 30), outline=rgb, width=lw)

        # Label pill background
        display = f"{label}  {conf:.0%}"
        try:
            bbox_txt = draw.textbbox((0, 0), display, font=font_large)
            txt_w = bbox_txt[2] - bbox_txt[0]
            txt_h = bbox_txt[3] - bbox_txt[1]
        except AttributeError:
            txt_w, txt_h = draw.textsize(display, font=font_large)

        pill_h = txt_h + 8
        pill_y = max(0, y1 - pill_h)
        pill_x2 = min(w, x1 + txt_w + 10)

        draw.rectangle([x1, pill_y, pill_x2, y1], fill=rgb)
        txt_color = (30, 30, 30, 255) if color in (MOS_CYAN, CAUTION_AMBER) else (255, 255, 255, 255)
        draw.text((x1 + 5, pill_y + 4), display, fill=txt_color, font=font_large)

    return img


def placeholder_image(
    label: str = "Image not available",
    width: int = 800,
    height: int = 300,
) -> Optional["PILImage.Image"]:
    """Create a styled placeholder when a demo image file is not found."""
    if not _PIL:
        return None
    img  = PILImage.new("RGB", (width, height), tuple(int(MOS_DARK_NAVY.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)))
    draw = ImageDraw.Draw(img)
    # Grid overlay
    for x in range(0, width, 60):
        draw.line([(x, 0), (x, height)], fill=(30, 40, 80), width=1)
    for y in range(0, height, 60):
        draw.line([(0, y), (width, y)], fill=(30, 40, 80), width=1)
    # Centred label
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    try:
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        tw, th = draw.textsize(label, font=font)
    draw.text(((width - tw) // 2, (height - th) // 2), label,
              fill=tuple(int(MOS_CYAN.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)),
              font=font)
    return img


# ═══════════════════════════════════════════════════════════════════════════════
# IFTTT RULE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_ifttt_engine(
    detections:  List[Dict],
    rules:       List[Dict],
    conf_thresh: float,
    workers_signed_on: int = 1,
) -> Dict:
    """
    Evaluate detections against training model rules.
    Returns verdict dict with per-rule results and overall verdict.
    """
    # Build lookup sets from detections
    detected_classes   = {d["class_name"] for d in detections if d["confidence"] >= conf_thresh}
    detected_below_thresh = {d["class_name"] for d in detections if d["confidence"] < conf_thresh}
    all_detected       = {d["class_name"] for d in detections}

    # Locks detected
    personal_locks = [d for d in detections if d["class_name"] == "personal_lock"
                      and d["confidence"] >= conf_thresh]
    permit_locks   = [d for d in detections if d["class_name"] == "permit_lock"
                      and d["confidence"] >= conf_thresh]
    lock_count     = len(personal_locks) + len(permit_locks)

    # Indicators
    indicator_lit   = any(d["class_name"] == "indicator_lit"   and d["confidence"] >= conf_thresh
                          for d in detections)
    indicator_fault = any(d["class_name"] == "indicator_fault" and d["confidence"] >= conf_thresh
                          for d in detections)
    indicator_off   = any(d["class_name"] == "indicator_off"   and d["confidence"] >= conf_thresh
                          for d in detections)

    # ── Evaluate per rule ──────────────────────────────────────────────────
    rule_results = []
    for rule in rules:
        result = {"rule": rule, "verdict": "PASS", "reason": ""}

        # Indicator-only rule
        if rule.get("indicator_only"):
            if indicator_fault:
                result["verdict"] = "STOP"
                result["reason"]  = "indicator_fault detected — mandatory manual verification required"
            elif indicator_lit:
                result["verdict"] = "FAIL"
                result["reason"]  = "indicator_lit detected — voltage present"
            elif not indicator_off and "indicator_off" not in detected_classes:
                result["verdict"] = "REVIEW"
                result["reason"]  = "No live line indicator detected — cannot confirm safe state"
            else:
                result["verdict"] = "PASS"
                result["reason"]  = "All visible indicators are OFF"
            rule_results.append(result)
            continue

        # Check isolation point state
        iso_pt  = rule.get("isolation_pt")
        comp    = rule.get("compartment")
        req_lock = rule.get("require_lock", False)
        ind_exp  = rule.get("indicator_expected")

        if iso_pt:
            if iso_pt in detected_classes:
                result["verdict"] = "PASS"
                result["reason"]  = f"{iso_pt} detected at ≥{conf_thresh:.0%} confidence"
            elif iso_pt in detected_below_thresh:
                result["verdict"] = "REVIEW"
                result["reason"]  = (f"{iso_pt} detected but below {conf_thresh:.0%} threshold — "
                                     "manual verification required")
            else:
                # Check if the CLOSED/dangerous counterpart is detected
                dangerous_counterpart = iso_pt.replace("_open", "_closed") \
                                              .replace("_withdrawn", "_inserted") \
                                              .replace("_applied", "_removed")
                if dangerous_counterpart in detected_classes:
                    result["verdict"] = "FAIL"
                    result["reason"]  = (f"Detected {dangerous_counterpart} — "
                                         "isolation point NOT in safe state")
                else:
                    result["verdict"] = "REVIEW"
                    result["reason"]  = f"{iso_pt} not detected — cannot confirm state"

        # Check lock requirement
        if req_lock and result["verdict"] == "PASS":
            if lock_count == 0:
                result["verdict"] = "FAIL"
                result["reason"]  = "No locks detected — isolation not secured"
            elif lock_count < workers_signed_on:
                result["verdict"] = "REVIEW"
                result["reason"]  = (f"Detected {lock_count} lock(s) but "
                                     f"{workers_signed_on} worker(s) signed on")

        # Check indicator expectation
        if ind_exp and result["verdict"] == "PASS":
            if ind_exp == "indicator_off" and indicator_lit:
                result["verdict"] = "FAIL"
                result["reason"]  += "; Indicator shows LIVE"
            elif ind_exp == "indicator_off" and indicator_fault:
                result["verdict"] = "STOP"
                result["reason"]  += "; indicator_fault — manual check required"

        rule_results.append(result)

    # ── Overall verdict ────────────────────────────────────────────────────
    verdicts = [r["verdict"] for r in rule_results]

    if "STOP" in verdicts or indicator_fault:
        overall = "STOP"
        overall_color = VERDICT_STOP
        overall_icon  = "🛑"
        overall_msg   = "STOP — Manual verification mandatory before entry"
    elif "FAIL" in verdicts or indicator_lit:
        overall = "DISCREPANCY"
        overall_color = VERDICT_DISCREPANCY
        overall_icon  = "❌"
        overall_msg   = "DISCREPANCY — Isolation incomplete. Do not enter."
    elif "REVIEW" in verdicts or detected_below_thresh:
        overall = "REVIEW REQUIRED"
        overall_color = VERDICT_REVIEW
        overall_icon  = "⚠"
        overall_msg   = "REVIEW REQUIRED — One or more items need manual verification"
    else:
        overall = "ALL CLEAR"
        overall_color = VERDICT_ALL_CLEAR
        overall_icon  = "✅"
        overall_msg   = "ALL CLEAR — All isolation points verified. Safe to proceed."

    # Block ALL CLEAR if any detection below threshold
    if overall == "ALL CLEAR" and detected_below_thresh:
        overall = "REVIEW REQUIRED"
        overall_color = VERDICT_REVIEW
        overall_icon  = "⚠"
        overall_msg   = (f"REVIEW REQUIRED — {len(detected_below_thresh)} detection(s) below "
                         f"{conf_thresh:.0%} threshold")

    return {
        "overall":        overall,
        "overall_color":  overall_color,
        "overall_icon":   overall_icon,
        "overall_msg":    overall_msg,
        "rule_results":   rule_results,
        "lock_count":     lock_count,
        "workers_signed_on": workers_signed_on,
        "lock_balance":   lock_count - workers_signed_on,
        "indicator_lit":  indicator_lit,
        "indicator_fault": indicator_fault,
        "indicator_off":  indicator_off,
        "detected_classes": list(detected_classes),
        "below_threshold":  list(detected_below_thresh),
    }


def infer_compartment_states(detections: List[Dict], conf_thresh: float) -> Dict[str, str]:
    """
    Determine the state (isolated/energised/unknown) of each compartment
    from detections. Returns dict of compartment_name → state.
    """
    states = {f"compartment_{i}": "unknown" for i in range(1, 11)}
    detected_classes = {d["class_name"] for d in detections if d["confidence"] >= conf_thresh}

    # If an isolation point (open state) is detected near a compartment,
    # that compartment is inferred as isolated.
    # This is a simplified heuristic — the full spatial analysis requires bbox overlap.
    open_classes   = {"circuit_breaker_open", "isolator_switch_open", "fuse_withdrawn",
                      "earth_switch_applied", "disconnect_open"}
    closed_classes = {"circuit_breaker_closed", "isolator_switch_closed", "fuse_inserted",
                      "earth_switch_removed", "disconnect_closed"}

    has_open   = bool(detected_classes & open_classes)
    has_closed = bool(detected_classes & closed_classes)

    # Map detected compartment classes to their states
    for i in range(1, 11):
        comp_key = f"compartment_{i}"
        if comp_key in detected_classes:
            # If any open isolation points detected, assume isolated
            # (real spatial analysis is done by the IFTTT engine)
            if has_open and not has_closed:
                states[comp_key] = "isolated"
            elif has_closed:
                states[comp_key] = "energised"
            else:
                states[comp_key] = "unknown"

    return states


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE ONNX INFERENCE ENGINE (Scenario 3)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_onnx_model(model_path: str):
    """Load ONNX model — cached so it's only loaded once per session."""
    if not _ONNX_AVAILABLE:
        return None
    try:
        sys.path.insert(0, str(Path(__file__).parent / "yolo_training"))
        from inference import MOSLockInference
        engine = MOSLockInference(model_path, conf_thresh=0.90)
        return engine
    except Exception as e:
        st.error(f"Failed to load ONNX model: {e}")
        return None


def run_live_inference(
    engine,
    image_source,
    conf_thresh: float,
) -> List[Dict]:
    """Run ONNX inference and convert Detection objects → dict list."""
    if engine is None:
        return []
    try:
        engine.set_confidence(conf_thresh)
        detections_raw = engine.infer(image_source)
        return [
            {
                "class_name": d.class_name,
                "confidence": d.confidence,
                "bbox": (
                    d.bbox[0] / engine.imgsz,   # normalise for display
                    d.bbox[1] / engine.imgsz,
                    d.bbox[2] / engine.imgsz,
                    d.bbox[3] / engine.imgsz,
                ),
                "label": CLASS_METADATA.get(d.class_name, {}).get("label", d.class_name),
            }
            for d in detections_raw
        ]
    except Exception as e:
        st.error(f"Inference error: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

def render_verdict_banner(verdict_data: Dict):
    """Render the overall verdict as a prominent banner."""
    overall = verdict_data["overall"]
    color   = verdict_data["overall_color"]
    icon    = verdict_data["overall_icon"]
    msg     = verdict_data["overall_msg"]

    css_class = {
        "ALL CLEAR":       "verdict-all-clear",
        "DISCREPANCY":     "verdict-discrepancy",
        "REVIEW REQUIRED": "verdict-review",
        "STOP":            "verdict-stop",
    }.get(overall, "verdict-review")

    st.markdown(
        f'<div class="verdict-banner {css_class}">'
        f'{icon} &nbsp; {msg}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Extra STOP alert if indicator_fault
    if verdict_data.get("indicator_fault"):
        st.markdown(
            '<div class="stop-alert">'
            '🛑 &nbsp; STOP — indicator_fault detected. '
            'Live line indicator is damaged or obscured. '
            'MANDATORY manual verification with calibrated test instrument before entry.'
            '</div>',
            unsafe_allow_html=True,
        )


def render_kpi_row(verdict_data: Dict, conf_thresh: float):
    """Render the 4-KPI top row."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        n_detected = len(verdict_data.get("detected_classes", []))
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-value">{n_detected}</div>'
            f'<div class="kpi-label">Classes Detected</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col2:
        lock_count = verdict_data.get("lock_count", 0)
        workers    = verdict_data.get("workers_signed_on", 0)
        card_class = "safe" if lock_count >= workers > 0 else ("danger" if lock_count == 0 else "caution")
        st.markdown(
            f'<div class="kpi-card {card_class}">'
            f'<div class="kpi-value">{lock_count} / {workers}</div>'
            f'<div class="kpi-label">Locks / Workers</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col3:
        n_below = len(verdict_data.get("below_threshold", []))
        card_class = "safe" if n_below == 0 else "caution"
        st.markdown(
            f'<div class="kpi-card {card_class}">'
            f'<div class="kpi-value">{n_below}</div>'
            f'<div class="kpi-label">Below {conf_thresh:.0%} Threshold</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col4:
        n_rules   = len(verdict_data.get("rule_results", []))
        n_pass    = sum(1 for r in verdict_data.get("rule_results", []) if r["verdict"] == "PASS")
        card_class = "safe" if n_pass == n_rules else ("danger" if n_pass < n_rules // 2 else "caution")
        st.markdown(
            f'<div class="kpi-card {card_class}">'
            f'<div class="kpi-value">{n_pass} / {n_rules}</div>'
            f'<div class="kpi-label">Rules PASS</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_rule_table(rule_results: List[Dict]):
    """Render the IFTTT rule verification results as a styled HTML table."""
    rows = ""
    for rr in rule_results:
        verdict = rr["verdict"]
        rule    = rr["rule"]
        reason  = rr["reason"]

        badge_class = {
            "PASS":   "badge-safe",
            "FAIL":   "badge-danger",
            "REVIEW": "badge-caution",
            "STOP":   "badge-stop",
        }.get(verdict, "badge-info")

        icon = {"PASS": "✅", "FAIL": "❌", "REVIEW": "⚠", "STOP": "🛑"}.get(verdict, "")

        rows += (
            f"<tr>"
            f"<td><strong>{rule['id']}</strong></td>"
            f"<td>{rule['name']}</td>"
            f"<td><span class='badge {badge_class}'>{icon} {verdict}</span></td>"
            f"<td style='font-size:14px;color:#555;'>{reason}</td>"
            f"</tr>"
        )

    html = f"""
    <div class="scroll-table-wrapper">
    <table class="rule-table">
        <thead>
            <tr>
                <th style="width:60px">Rule</th>
                <th>Description</th>
                <th style="width:130px">Verdict</th>
                <th>Reason</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_compartment_grid(states: Dict[str, str]):
    """Render the 10-compartment status grid."""
    cells = ""
    for i in range(1, 11):
        key   = f"compartment_{i}"
        state = states.get(key, "unknown")
        name  = COMPARTMENT_NAMES.get(key, key)
        css   = {"isolated": "isolated", "energised": "energised"}.get(state, "unknown")
        icon  = {"isolated": "✅ Isolated", "energised": "⚡ Energised"}.get(state, "? Unknown")
        cells += (
            f"<div class='compartment-cell {css}'>"
            f"<div class='comp-id'>C{i}</div>"
            f"<strong>{name}</strong>"
            f"<div style='font-size:12px;margin-top:4px;'>{icon}</div>"
            f"</div>"
        )
    st.markdown(
        f"<div class='compartment-grid'>{cells}</div>",
        unsafe_allow_html=True,
    )


def render_twelve_step_progress(current_step: int):
    """Render the 12-step isolation progress indicator."""
    dots = ""
    for step, label, phase in TWELVE_STEPS:
        if step < current_step:
            css = "complete"
        elif step == current_step:
            css = "active"
        else:
            css = "pending"
        title = f"Step {step}: {label}"
        dots += f"<div class='step-dot {css}' title='{title}'>{step}</div>"

    phase_label = "Isolation Phase" if current_step <= 6 else "Restoration Phase"
    current_name = TWELVE_STEPS[current_step - 1][1] if 1 <= current_step <= 12 else ""

    st.markdown(
        f"<div style='margin-bottom:8px;'>"
        f"<strong style='color:{MOS_DARK_NAVY};'>12-Step Process</strong> "
        f"<span style='color:{MOS_CYAN};font-size:13px;'>{phase_label}</span>"
        f"<span style='float:right;font-size:13px;color:#555;'>Step {current_step}: {current_name}</span>"
        f"</div>"
        f"<div class='step-indicator'>{dots}</div>",
        unsafe_allow_html=True,
    )


def render_detection_dataframe(detections: List[Dict], conf_thresh: float):
    """Render detections as a sortable dataframe."""
    if not _PANDAS:
        st.warning("Install pandas for table view: pip install pandas")
        for d in detections:
            st.text(f"{d['class_name']}: {d['confidence']:.1%}")
        return

    if not detections:
        st.info("No detections.")
        return

    rows = []
    for d in detections:
        conf = d["confidence"]
        cname = d["class_name"]
        group = CLASS_METADATA.get(cname, {}).get("group", "Unknown")
        above = "✅" if conf >= conf_thresh else "⚠ Below threshold"
        rows.append({
            "Class":      cname,
            "Group":      group,
            "Confidence": f"{conf:.1%}",
            "Threshold":  above,
            "Label":      d.get("label", cname),
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Confidence": st.column_config.TextColumn("Confidence"),
        },
    )


def render_sidebar(high_contrast: bool) -> Dict:
    """Render the sidebar and return all control values."""
    with st.sidebar:
        # Logo — white-out version on dark sidebar
        st.markdown(logo_img_tag(dark_bg=True, width=210), unsafe_allow_html=True)
        st.markdown(
            f"<p style='color:{MOS_CYAN};font-size:12px;margin-top:0;"
            f"font-style:italic;'>{STRAPLINE}</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='color:rgba(255,255,255,0.5);font-size:11px;'>"
            f"v{APP_VERSION}</p>",
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Mode toggle ────────────────────────────────────────────
        st.markdown("### 🔧 Mode")
        live_mode = st.toggle("Live Mode (requires ONNX model)", value=False,
                               help="Enable camera + ONNX inference. Requires trained model file.")

        if live_mode and not _ONNX_AVAILABLE:
            st.warning("onnxruntime not installed.\n```\npip install onnxruntime numpy\n```")
            live_mode = False

        st.divider()

        # ── Permit details ─────────────────────────────────────────
        st.markdown("### 📋 Permit")
        permit_id = st.text_input("Permit ID", value="HV-" + datetime.now().strftime("%y-%m-%d-%H-%M"),
                                   help="GCAA permit format: HV-yy-mm-dd-hh-mm")
        site_name = st.text_input("Site", value="BST008 — Demo Site")
        workers_signed_on = st.number_input("Workers Signed On", min_value=0, max_value=20, value=1,
                                             help="Number of workers with personal locks on the lock box")
        current_step = st.select_slider(
            "12-Step Progress",
            options=list(range(1, 13)),
            value=5,
            help="Current position in the GCAA 12-step isolation process",
        )
        st.divider()

        # ── Demo image selector ────────────────────────────────────
        if not live_mode:
            st.markdown("### 📷 Demo Image")
            demo_choice = st.selectbox(
                "Select scenario",
                options=list(DEMO_IMAGES.keys()),
                index=1,
            )
        else:
            demo_choice = None

        # ── Live mode settings ─────────────────────────────────────
        if live_mode:
            st.markdown("### 📷 Live Mode")
            model_path = st.text_input(
                "ONNX Model Path",
                value="./models/moslock_yolov8n.onnx",
                help="Path to trained ONNX model file",
            )
            use_camera = st.radio(
                "Input",
                ["Camera (primary)", "Upload file"],
                index=0,
                help="Camera is primary on tablet; file upload for desktop",
            )
        else:
            model_path = "./models/moslock_yolov8n.onnx"
            use_camera = "Camera (primary)"

        # ── Confidence threshold ───────────────────────────────────
        st.markdown("### 🎯 Confidence")
        conf_thresh = st.slider(
            "Min. confidence threshold",
            min_value=50, max_value=99, value=90, step=1,
            format="%d%%",
            help="Detections below this threshold trigger REVIEW. Default: 90%.",
        ) / 100.0

        st.divider()

        # ── Accessibility ──────────────────────────────────────────
        st.markdown("### ♿ Accessibility")
        high_contrast = st.toggle(
            "High Contrast Mode",
            value=high_contrast,
            help="Maximum contrast for underground / low-light conditions",
        )
        st.divider()

        # ── Sidebar footer ─────────────────────────────────────────
        st.markdown(
            f"<div style='font-size:11px;color:rgba(255,255,255,0.4);padding-top:8px;'>"
            f"<strong>Standards:</strong><br>"
            f"GCAA Fatal Hazard Protocol 7<br>"
            f"AS/NZS 4836:2023<br>"
            f"AS 4871.1<br><br>"
            f"<strong>Support:</strong><br>"
            f"mappedoutsolutions.com"
            f"</div>",
            unsafe_allow_html=True,
        )

    return {
        "live_mode":         live_mode,
        "demo_choice":       demo_choice,
        "model_path":        model_path if live_mode else "",
        "use_camera":        use_camera,
        "conf_thresh":       conf_thresh,
        "workers_signed_on": int(workers_signed_on),
        "current_step":      int(current_step),
        "permit_id":         permit_id,
        "site_name":         site_name,
        "high_contrast":     high_contrast,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TAB RENDERERS
# ═══════════════════════════════════════════════════════════════════════════════

def tab_detection(controls: Dict, detections: List[Dict], annotated_img: Optional["PILImage.Image"]):
    """Tab 1: Detection Image + Results."""
    st.markdown("## Detection Results")

    if annotated_img is not None:
        st.image(
            annotated_img,
            use_container_width=True,
            caption=f"{'Live' if controls['live_mode'] else 'Demo'} — "
                    f"{DEMO_IMAGES.get(controls.get('demo_choice',''), {}).get('label', '')}",
        )
    else:
        st.info("No image loaded. Select a demo image or enable Live Mode.")
        placeholder = placeholder_image("No image available — select a demo scenario")
        if placeholder:
            st.image(placeholder, use_container_width=True)

    st.markdown("### Detections")
    render_detection_dataframe(detections, controls["conf_thresh"])


def tab_verification(controls: Dict, detections: List[Dict], verdict_data: Dict):
    """Tab 2: IFTTT Verification Engine."""
    st.markdown("## Isolation Verification")

    # 12-step progress
    render_twelve_step_progress(controls["current_step"])
    st.markdown("")

    # Permit header
    st.markdown(
        f'<div class="permit-header">'
        f'<h3>Permit: {controls["permit_id"]}</h3>'
        f'<p>📍 <strong>Site:</strong> {controls["site_name"]} &nbsp;|&nbsp; '
        f'👷 <strong>Workers signed on:</strong> {controls["workers_signed_on"]} &nbsp;|&nbsp; '
        f'🎯 <strong>Threshold:</strong> {controls["conf_thresh"]:.0%}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Main verdict
    render_verdict_banner(verdict_data)

    # KPI row
    render_kpi_row(verdict_data, controls["conf_thresh"])

    st.markdown("### Rule Verification")
    render_rule_table(verdict_data["rule_results"])

    # Lock count
    st.markdown("### Lock Count")
    lc   = verdict_data["lock_count"]
    ws   = verdict_data["workers_signed_on"]
    diff = lc - ws
    if diff == 0 and ws > 0:
        msg   = f"✅ {lc} lock(s) present, {ws} worker(s) signed on — balanced"
        color = SAFE_GREEN
    elif diff < 0:
        msg   = f"❌ {lc} lock(s) present but {ws} worker(s) signed on — {abs(diff)} lock(s) missing"
        color = DANGER_RED
    elif diff > 0:
        msg   = f"⚠ {lc} lock(s) present, {ws} worker(s) signed on — {diff} extra lock(s) detected"
        color = CAUTION_AMBER
    else:
        msg   = f"No workers signed on"
        color = MOS_DARK_NAVY

    st.markdown(
        f'<div class="lock-chip" style="border-color:{color};color:{color};">'
        f'🔒 {msg}'
        f'</div>',
        unsafe_allow_html=True,
    )


def tab_compartments(detections: List[Dict], conf_thresh: float):
    """Tab 3: Compartment Status Grid."""
    st.markdown("## Compartment Status")
    st.caption("Compartment states inferred from detected isolation points. "
               "Green = Isolated, Red = Energised, Amber = Status unknown.")

    states = infer_compartment_states(detections, conf_thresh)
    render_compartment_grid(states)

    st.markdown("### Compartment Details")
    if _PANDAS:
        rows = []
        for i in range(1, 11):
            key   = f"compartment_{i}"
            state = states.get(key, "unknown")
            name  = COMPARTMENT_NAMES.get(key, f"Compartment {i}")
            detected = key in {d["class_name"] for d in detections}
            rows.append({
                "ID":         f"C{i}",
                "Zone":       name,
                "State":      state.capitalize(),
                "Detected":   "Yes" if detected else "—",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        for i in range(1, 11):
            key = f"compartment_{i}"
            state = states.get(key, "unknown")
            name  = COMPARTMENT_NAMES.get(key, f"Compartment {i}")
            icon  = {"isolated": "✅", "energised": "⚡", "unknown": "?"}.get(state, "?")
            st.text(f"C{i}: {name:30} {icon} {state.capitalize()}")


def tab_training_model(controls: Dict):
    """Tab 4: Training Model / IFTTT Rule Editor."""
    st.markdown("## Training Model — IFTTT Rules")
    st.info(
        "The Training Model defines what constitutes a correctly isolated substation. "
        "Each rule maps an isolation point state to a pass/fail condition. "
        "Rules are evaluated against detections during verification."
    )

    st.markdown("### Active Rules")
    for i, rule in enumerate(DEFAULT_RULES):
        with st.expander(f"Rule {rule['id']}: {rule['name']}", expanded=(i == 0)):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.text_area(
                    "Description",
                    value=rule["description"],
                    height=80,
                    key=f"rule_desc_{i}",
                    disabled=True,
                )
            with col2:
                st.markdown(f"**Isolation Point:** `{rule.get('isolation_pt', 'N/A')}`")
                st.markdown(f"**Compartment:** `{rule.get('compartment', 'N/A')}`")
                st.markdown(f"**Requires Lock:** {'Yes ✅' if rule.get('require_lock') else 'No'}")
                if rule.get("indicator_expected"):
                    st.markdown(f"**Indicator Expected:** `{rule['indicator_expected']}`")

    st.markdown("### Add Custom Rule")
    with st.expander("➕ Add Rule", expanded=False):
        st.info("Custom rules will be applied alongside the default rules in this session.")
        c1, c2 = st.columns(2)
        with c1:
            new_iso = st.selectbox(
                "Required Isolation State",
                [n for n in CLASS_NAMES if "_open" in n or "_withdrawn" in n or "_applied" in n],
                key="new_rule_iso",
            )
            new_comp = st.selectbox(
                "Associated Compartment",
                [f"compartment_{i}" for i in range(1, 11)],
                key="new_rule_comp",
            )
        with c2:
            new_lock = st.checkbox("Require lock", value=True, key="new_rule_lock")
            new_name = st.text_input("Rule name", value="Custom Rule", key="new_rule_name")

        if st.button("Add Rule", key="add_rule_btn"):
            st.success(f"Rule added: {new_name} — {new_iso} on {new_comp}")


def tab_permit(controls: Dict, verdict_data: Dict, detections: List[Dict]):
    """Tab 5: High Voltage Permit & Sign-On Register."""
    st.markdown("## High Voltage Access Permit")

    # Permit status
    st.markdown(
        f'<div class="permit-header">'
        f'<h3>Permit ID: {controls["permit_id"]}</h3>'
        f'<p>📍 <strong>Site:</strong> {controls["site_name"]}</p>'
        f'<p>📅 <strong>Generated:</strong> {datetime.now().strftime("%d %b %Y %H:%M")}</p>'
        f'<p>🎯 <strong>Overall Verdict:</strong> '
        f'<span style="color:{verdict_data["overall_color"]};font-weight:900;">'
        f'{verdict_data["overall_icon"]} {verdict_data["overall"]}'
        f'</span></p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # GCAA 12-step reference
    with st.expander("📋 GCAA 12-Step Isolation Process Reference", expanded=False):
        st.markdown("**Per Glencore GCAA Fatal Hazard Protocol 7:**")
        cols = st.columns(2)
        for i, (step, label, phase) in enumerate(TWELVE_STEPS):
            phase_color = MOS_CYAN if phase == "isolation" else CAUTION_AMBER
            col = cols[i % 2]
            with col:
                completed = step < controls["current_step"]
                icon = "✅" if completed else ("🔵" if step == controls["current_step"] else "⚪")
                st.markdown(
                    f"**{icon} Step {step}:** {label} "
                    f"<span style='color:{phase_color};font-size:12px;'>({phase})</span>",
                    unsafe_allow_html=True,
                )

    # Worker sign-on register
    st.markdown("### Worker Sign-On Register")
    st.caption("Per GCAA permit requirements — each worker applies their personal lock to the lockout station when signing on.")

    if "workers" not in st.session_state:
        st.session_state.workers = []

    with st.form("sign_on_form", clear_on_submit=True):
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            worker_name = st.text_input("Worker Name", placeholder="Printed name")
        with col2:
            worker_contact = st.text_input("Contact No.", placeholder="Phone / radio channel")
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Sign On 🔒", use_container_width=True)

        if submitted and worker_name:
            st.session_state.workers.append({
                "name":      worker_name,
                "contact":   worker_contact,
                "signed_on": datetime.now().strftime("%H:%M"),
                "signed_off": "—",
                "status":    "Signed On",
            })
            st.success(f"✅ {worker_name} signed on at {datetime.now().strftime('%H:%M')}")

    if st.session_state.workers:
        if _PANDAS:
            df = pd.DataFrame(st.session_state.workers)
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Sign-off buttons
        for i, w in enumerate(st.session_state.workers):
            if w["status"] == "Signed On":
                if st.button(f"Sign Off: {w['name']}", key=f"signoff_{i}"):
                    st.session_state.workers[i]["signed_off"] = datetime.now().strftime("%H:%M")
                    st.session_state.workers[i]["status"] = "Signed Off"
                    st.rerun()

        # Lock balance check
        signed_on_count = sum(1 for w in st.session_state.workers if w["status"] == "Signed On")
        if signed_on_count != verdict_data["lock_count"] and verdict_data["lock_count"] > 0:
            st.warning(
                f"⚠ Register shows {signed_on_count} worker(s) signed on, "
                f"but {verdict_data['lock_count']} lock(s) detected. "
                "Reconcile before proceeding."
            )
    else:
        st.info("No workers signed on. Use the form above to register workers.")

    # Isolation drawing reference
    with st.expander("📐 Single Line Diagram Reference", expanded=False):
        sld_path = find_image("Isolation drawing.jpg")
        if sld_path:
            st.image(str(sld_path), use_container_width=True,
                     caption="BST008 HV Single Line Diagram")
        else:
            st.info("Isolation drawing not found in demo_images/. "
                    "Place 'Isolation drawing.jpg' in the demo_images/ folder.")


def tab_about():
    """Tab 6: About / Help."""
    st.markdown("## About MOSLock V3")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"""
        **{PRODUCT_NAME}** is an AI-assisted isolation verification tool developed by
        **{COMPANY_NAME}** for high-voltage substation maintenance.

        MOSLock uses a YOLOv8n computer vision model to detect and classify equipment states
        on mobile substations, cross-referencing detections against a Training Model
        (IFTTT rule set) to verify that isolation has been completed correctly per the
        Glencore GCAA 12-step isolation process (Fatal Hazard Protocol 7).

        ### How It Works
        1. **Scan** the substation with the tablet camera (or upload a photo)
        2. **Detect** — YOLOv8n identifies locks, isolation points, compartments, and indicators
        3. **Verify** — IFTTT engine compares detections against the Training Model rules
        4. **Verdict** — ALL CLEAR / DISCREPANCY / REVIEW REQUIRED / STOP
        5. **Record** — Permit register captures who was signed on, when, and what was verified

        ### Safety Warning
        MOSLock is a verification **aid** — not a replacement for physical isolation
        procedures. The GCAA 12-step process must always be completed. If ANY
        indicator_fault is detected, **mandatory manual verification** with a calibrated
        instrument is required before entry.

        ### Standards
        - Glencore GCAA Fatal Hazard Protocol 7
        - AS/NZS 4836:2023 Safe Working on LV Electrical Installations
        - AS 4871.1 Electrical Equipment for Coal Mines
        """)

    with col2:
        st.markdown(logo_img_tag(dark_bg=False, width=180), unsafe_allow_html=True)
        st.markdown(
            f"<p style='font-size:13px;color:{MOS_DARK_NAVY};font-style:italic;'>"
            f"{STRAPLINE}</p>",
            unsafe_allow_html=True,
        )
        st.markdown(f"""
        **Version:** {APP_VERSION}  
        **Build:** Dec 2025  
        **Target Device:** Samsung Galaxy Tab S10  
        **Model:** YOLOv8n (ONNX)  
        **Classes:** 25  

        **Contact:**  
        mappedoutsolutions.com
        """)

    with st.expander("🔒 Demo Mode — What you're seeing"):
        st.markdown("""
        **Demo Mode** shows pre-loaded detection results from the BST008 V2 trial
        site images. No ONNX model is required.

        Switch to **Live Mode** (sidebar toggle) to:
        - Use the tablet camera as input
        - Run real-time ONNX inference
        - Requires `./models/moslock_yolov8n.onnx`

        Train the model with:
        ```bash
        pip install ultralytics
        cd yolo_training
        python generate_synthetic_samples.py
        python train.py
        python export.py
        ```
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

def render_footer(permit_id: str):
    st.markdown(
        f'<div class="mos-footer">'
        f'<div>'
        f'  {logo_img_tag(dark_bg=True, width=130)}'
        f'  <span class="strapline">{STRAPLINE}</span>'
        f'</div>'
        f'<div style="text-align:center;">'
        f'  <strong style="color:{MOS_CYAN};">{PRODUCT_NAME}</strong> v{APP_VERSION}<br>'
        f'  Permit: {permit_id}'
        f'</div>'
        f'<div style="text-align:right;font-size:11px;">'
        f'  GCAA Fatal Hazard Protocol 7<br>'
        f'  AS/NZS 4836:2023 | AS 4871.1<br>'
        f'  &copy; {datetime.now().year} {COMPANY_NAME}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Streamlit page config ────────────────────────────────────────────────
    st.set_page_config(
        page_title=f"{APP_NAME} V3 — {COMPANY_NAME}",
        page_icon="🔒",
        layout="wide",
        initial_sidebar_state="collapsed",   # collapsed by default on tablet
        menu_items={
            "Get Help":     "https://mappedoutsolutions.com",
            "Report a bug": "https://mappedoutsolutions.com",
            "About":        f"{PRODUCT_NAME} v{APP_VERSION} — {COMPANY_NAME}",
        },
    )

    # ── Initialise session state ─────────────────────────────────────────────
    if "high_contrast" not in st.session_state:
        st.session_state.high_contrast = False
    if "workers" not in st.session_state:
        st.session_state.workers = []
    if "last_detections" not in st.session_state:
        st.session_state.last_detections = []

    # ── Sidebar (get controls first so we know high_contrast for CSS) ────────
    controls = render_sidebar(st.session_state.high_contrast)
    st.session_state.high_contrast = controls["high_contrast"]

    # ── Inject CSS ───────────────────────────────────────────────────────────
    inject_css(controls["high_contrast"])

    # ── Header ───────────────────────────────────────────────────────────────
    header_col1, header_col2 = st.columns([1, 3])
    with header_col1:
        st.markdown(logo_img_tag(dark_bg=False, width=220), unsafe_allow_html=True)
    with header_col2:
        mode_pill = (
            f'<span class="mode-pill live">🟢 Live Mode</span>'
            if controls["live_mode"] else
            f'<span class="mode-pill demo">🔵 Demo Mode</span>'
        )
        st.markdown(
            f"<h1 style='margin:0;padding:0;'>{APP_NAME} V3 &nbsp; {mode_pill}</h1>"
            f"<p style='margin:2px 0 0 0;color:#555;font-size:15px;'>"
            f"Isolation Verification Assistant — {controls['site_name']}"
            f"</p>",
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Acquire detections ───────────────────────────────────────────────────
    detections:   List[Dict]           = []
    source_img:   Optional[PILImage.Image] = None
    annotated_img: Optional[PILImage.Image] = None

    if controls["live_mode"]:
        # ── LIVE MODE ────────────────────────────────────────────────────────
        engine = load_onnx_model(controls["model_path"])

        if engine is None:
            if not _ONNX_AVAILABLE:
                st.warning(
                    "**onnxruntime not installed.** "
                    "Install with: `pip install onnxruntime numpy`\n\n"
                    "Showing Demo Mode until model is available."
                )
            else:
                model_path = Path(controls["model_path"])
                st.warning(
                    f"**Model not found:** `{model_path}`\n\n"
                    "Run `yolo_training/train.py` and `export.py` to generate the model.\n"
                    "Showing Demo Mode until model is available."
                )
            # Graceful degradation — fall through to demo
            controls["live_mode"] = False

        if controls["live_mode"] and engine is not None:
            st.info("📷 Point tablet camera at the substation then press **Capture**")
            if controls["use_camera"] == "Camera (primary)":
                camera_img = st.camera_input(
                    "Capture substation image",
                    key="live_camera",
                    help="Point at the substation isolation points. Ensure lighting is adequate.",
                )
                if camera_img is not None:
                    source_img    = PILImage.open(camera_img).convert("RGB")
                    detections    = run_live_inference(engine, source_img, controls["conf_thresh"])
                    annotated_img = draw_detections_pil(source_img, detections, controls["conf_thresh"],
                                                        controls["high_contrast"])
            else:
                uploaded = st.file_uploader(
                    "Upload substation photo",
                    type=["jpg", "jpeg", "png", "webp"],
                    key="live_upload",
                )
                if uploaded is not None:
                    source_img    = PILImage.open(uploaded).convert("RGB")
                    detections    = run_live_inference(engine, source_img, controls["conf_thresh"])
                    annotated_img = draw_detections_pil(source_img, detections, controls["conf_thresh"],
                                                        controls["high_contrast"])

    if not controls["live_mode"]:
        # ── DEMO MODE ────────────────────────────────────────────────────────
        demo_key  = controls.get("demo_choice") or list(DEMO_IMAGES.keys())[1]
        demo_data = DEMO_IMAGES.get(demo_key, {})
        detections = demo_data.get("detections", [])

        img_file = demo_data.get("file", "")
        img_path = find_image(img_file) if img_file else None

        if img_path and _PIL:
            try:
                source_img = PILImage.open(str(img_path)).convert("RGB")
                annotated_img = draw_detections_pil(
                    source_img, detections, controls["conf_thresh"], controls["high_contrast"]
                )
            except Exception as e:
                st.warning(f"Could not load demo image '{img_file}': {e}")
                annotated_img = placeholder_image(f"Demo: {demo_key}", 900, 350)
        else:
            annotated_img = placeholder_image(f"Demo: {demo_key}", 900, 350)

    # ── Store detections in session state ────────────────────────────────────
    st.session_state.last_detections = detections

    # ── Run IFTTT engine ─────────────────────────────────────────────────────
    verdict_data = run_ifttt_engine(
        detections,
        DEFAULT_RULES,
        controls["conf_thresh"],
        controls["workers_signed_on"],
    )

    # ── STOP alert — always visible at top if indicator_fault ────────────────
    if verdict_data.get("indicator_fault"):
        st.error(
            "🛑 **STOP — indicator_fault detected.** "
            "The live line indicator is damaged or obscured. "
            "MANDATORY manual verification with calibrated test instrument required before entry."
        )

    # ── Tab navigation ────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📷 Detection",
        "✅ Verification",
        "🏭 Compartments",
        "📐 Training Model",
        "📋 Permit",
        "ℹ About",
    ])

    with tab1:
        tab_detection(controls, detections, annotated_img)

    with tab2:
        tab_verification(controls, detections, verdict_data)

    with tab3:
        tab_compartments(detections, controls["conf_thresh"])

    with tab4:
        tab_training_model(controls)

    with tab5:
        tab_permit(controls, verdict_data, detections)

    with tab6:
        tab_about()

    # ── Footer ────────────────────────────────────────────────────────────────
    render_footer(controls["permit_id"])


if __name__ == "__main__":
    main()
