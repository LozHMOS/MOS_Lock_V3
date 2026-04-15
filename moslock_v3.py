"""
moslock_v3.py
MOSLock V3 — Mapped Out Solutions® | mos | Lock®
"We will always have a plan."

Tablet-ready isolation verification tool for BST008 mobile substations.

Run:  streamlit run moslock_v3.py
Deps: pip install streamlit pillow pandas
"""

from __future__ import annotations

import io
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

_ONNX_AVAILABLE = False
try:
    import onnxruntime as ort   # noqa: F401
    import numpy as np          # noqa: F401
    _ONNX_AVAILABLE = True
except ImportError:
    pass

# ── Brand constants (simple strings only — no CSS, no HTML) ──────────────────
APP_NAME    = "MOSLock"
APP_VERSION = "3.0.0"
COMPANY     = "Mapped Out Solutions®"
STRAPLINE   = "We will always have a plan."

# Safety palette used only for PIL bbox drawing and st.metric delta colours.
COL_GREEN  = "#1a9e5c"
COL_AMBER  = "#d48500"
COL_RED    = "#cc2d20"
COL_STOP   = "#b30000"
COL_CYAN   = "#3ECFCF"

# ── Image search dirs ─────────────────────────────────────────────────────────
IMAGE_DIRS = [
    Path(__file__).parent / "demo_images",
    Path(__file__).parent,
    Path("/mnt/session/uploads/workspace/images"),
]

# ── 25 detection classes ──────────────────────────────────────────────────────
CLASS_NAMES = [
    "personal_lock", "permit_lock",
    "circuit_breaker_open", "circuit_breaker_closed",
    "isolator_switch_open", "isolator_switch_closed",
    "fuse_withdrawn", "fuse_inserted",
    "earth_switch_applied", "earth_switch_removed",
    "disconnect_open", "disconnect_closed",
    "compartment_1", "compartment_2", "compartment_3", "compartment_4",
    "compartment_5", "compartment_6", "compartment_7", "compartment_8",
    "compartment_9", "compartment_10",
    "indicator_lit", "indicator_off", "indicator_fault",
]

# colour used only when drawing PIL bounding boxes
BBOX_COLOUR = {
    "personal_lock":          "#FF6B35",
    "permit_lock":            "#F5A623",
    "circuit_breaker_open":   "#1a9e5c",
    "circuit_breaker_closed": "#cc2d20",
    "isolator_switch_open":   "#1a9e5c",
    "isolator_switch_closed": "#cc2d20",
    "fuse_withdrawn":         "#1a9e5c",
    "fuse_inserted":          "#cc2d20",
    "earth_switch_applied":   "#007BFF",
    "earth_switch_removed":   "#F5A623",
    "disconnect_open":        "#1a9e5c",
    "disconnect_closed":      "#cc2d20",
    "indicator_lit":          "#cc2d20",
    "indicator_off":          "#1a9e5c",
    "indicator_fault":        "#b30000",
}
for _c in [f"compartment_{i}" for i in range(1, 11)]:
    BBOX_COLOUR[_c] = COL_CYAN

COMPARTMENT_NAMES = {
    "compartment_1": "HV Incoming",    "compartment_2": "Transformer",
    "compartment_3": "LV Outgoing",    "compartment_4": "Control",
    "compartment_5": "Auxiliary",      "compartment_6": "Bus Section",
    "compartment_7": "Metering",       "compartment_8": "Protection Relay",
    "compartment_9": "Cable Termination", "compartment_10": "Earth Bar",
}

TWELVE_STEPS = [
    (1,  "Identify Energy Sources",      "isolation"),
    (2,  "Advise Relevant Parties",      "isolation"),
    (3,  "Isolate & Secure",             "isolation"),
    (4,  "Place Tags / Locks / Permits", "isolation"),
    (5,  "Verify Isolation (Test Dead)", "isolation"),
    (6,  "Commence Work",                "isolation"),
    (7,  "Complete Work",                "restoration"),
    (8,  "Check Work",                   "restoration"),
    (9,  "Clear Area",                   "restoration"),
    (10, "Remove Tags / Locks / Permits","restoration"),
    (11, "Restore Energy",               "restoration"),
    (12, "Check Operation",              "restoration"),
]

# ── Demo detection data ───────────────────────────────────────────────────────
DEMO_IMAGES = {
    "Substation — Base State": {
        "file": "Substation.jpeg",
        "label": "BST008 — No isolation (base state)",
        "detections": [
            {"class_name": "compartment_1", "confidence": 0.97, "bbox": (0.02,0.05,0.21,0.98), "label": "HV Incoming"},
            {"class_name": "compartment_2", "confidence": 0.95, "bbox": (0.22,0.05,0.52,0.98), "label": "Transformer"},
            {"class_name": "compartment_3", "confidence": 0.96, "bbox": (0.53,0.05,0.98,0.98), "label": "LV Outgoing"},
        ],
    },
    "Substation — Training Model": {
        "file": "Substation Trained.jpeg",
        "label": "BST008 — Training model annotations",
        "detections": [
            {"class_name": "compartment_1",        "confidence": 0.97, "bbox": (0.02,0.05,0.21,0.98), "label": "Compartment 1"},
            {"class_name": "compartment_2",        "confidence": 0.95, "bbox": (0.22,0.05,0.52,0.98), "label": "Compartment 2"},
            {"class_name": "compartment_3",        "confidence": 0.96, "bbox": (0.53,0.05,0.98,0.98), "label": "Compartment 3"},
            {"class_name": "isolator_switch_open", "confidence": 0.93, "bbox": (0.03,0.55,0.09,0.75), "label": "Isolation Pt 1"},
            {"class_name": "circuit_breaker_open", "confidence": 0.91, "bbox": (0.26,0.28,0.32,0.46), "label": "Isolation Pt 2"},
            {"class_name": "disconnect_open",      "confidence": 0.94, "bbox": (0.52,0.10,0.60,0.35), "label": "Isolation Pt 3"},
            {"class_name": "indicator_off",        "confidence": 0.96, "bbox": (0.72,0.47,0.79,0.60), "label": "Indicator 1"},
        ],
    },
    "Substation — Isolated LV": {
        "file": "Substation Isolated LV.jpeg",
        "label": "BST008 — LV compartment isolated",
        "detections": [
            {"class_name": "compartment_1",        "confidence": 0.97, "bbox": (0.02,0.05,0.21,0.98), "label": "C1 Energised"},
            {"class_name": "compartment_2",        "confidence": 0.95, "bbox": (0.22,0.05,0.52,0.98), "label": "C2 Energised"},
            {"class_name": "compartment_3",        "confidence": 0.96, "bbox": (0.53,0.05,0.98,0.98), "label": "C3 Isolated"},
            {"class_name": "isolator_switch_open", "confidence": 0.93, "bbox": (0.03,0.55,0.09,0.75), "label": "Isolation Pt 1"},
            {"class_name": "circuit_breaker_open", "confidence": 0.91, "bbox": (0.26,0.28,0.32,0.46), "label": "Isolation Pt 2"},
            {"class_name": "disconnect_open",      "confidence": 0.94, "bbox": (0.52,0.10,0.60,0.35), "label": "Isolation Pt 3"},
            {"class_name": "personal_lock",        "confidence": 0.96, "bbox": (0.54,0.22,0.59,0.30), "label": "Personal Lock"},
            {"class_name": "indicator_off",        "confidence": 0.96, "bbox": (0.72,0.47,0.79,0.60), "label": "Indicator Off"},
        ],
    },
    "Substation — Locked Out": {
        "file": "Sub Locked Out.jpg",
        "label": "BST008 — Lock at isolation point 3",
        "detections": [
            {"class_name": "compartment_1",  "confidence": 0.97, "bbox": (0.02,0.05,0.21,0.98), "label": "Compartment 1"},
            {"class_name": "compartment_2",  "confidence": 0.95, "bbox": (0.22,0.05,0.52,0.98), "label": "Compartment 2"},
            {"class_name": "compartment_3",  "confidence": 0.96, "bbox": (0.53,0.05,0.98,0.98), "label": "Compartment 3"},
            {"class_name": "disconnect_open","confidence": 0.94, "bbox": (0.52,0.10,0.60,0.35), "label": "Isolation Pt 3"},
            {"class_name": "personal_lock",  "confidence": 0.97, "bbox": (0.50,0.25,0.57,0.35), "label": "Lock at Pt 3"},
        ],
    },
}

DEFAULT_RULES = [
    {"id": "R001", "name": "HV Incoming — Circuit Breaker Open",
     "isolation_pt": "circuit_breaker_open",  "require_lock": True,  "indicator_only": False,
     "description": "HV incoming CB must be open with personal lock. Indicator must be OFF."},
    {"id": "R002", "name": "Transformer — Isolator Switch Open",
     "isolation_pt": "isolator_switch_open",  "require_lock": True,  "indicator_only": False,
     "description": "Transformer isolator must be open with personal lock."},
    {"id": "R003", "name": "LV Outgoing — Disconnect Open",
     "isolation_pt": "disconnect_open",        "require_lock": True,  "indicator_only": False,
     "description": "LV outgoing disconnect must be open with lock applied."},
    {"id": "R004", "name": "Working Earth Applied",
     "isolation_pt": "earth_switch_applied",   "require_lock": False, "indicator_only": False,
     "description": "Working earth must be applied to confirm dead circuit."},
    {"id": "R005", "name": "No Live Indicators",
     "isolation_pt": None,                     "require_lock": False, "indicator_only": True,
     "description": "All indicators must be OFF. indicator_fault triggers STOP."},
]


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def find_image(filename: str) -> Optional[Path]:
    for d in IMAGE_DIRS:
        p = d / filename
        if p.exists():
            return p
    return None


def draw_detections(img: "PILImage.Image", detections: List[Dict],
                    conf_thresh: float) -> "PILImage.Image":
    """Draw bounding boxes on a PIL image. Returns annotated copy."""
    if not _PIL:
        return img
    img = img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        if det["confidence"] < conf_thresh:
            continue
        cname = det["class_name"]
        bbox  = det["bbox"]
        label = f"{det.get('label', cname)}  {det['confidence']:.0%}"
        hex_c = BBOX_COLOUR.get(cname, "#888888")
        r, g, b = int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)
        col = (r, g, b)

        x1, y1, x2, y2 = (int(bbox[0]*w), int(bbox[1]*h),
                           int(bbox[2]*w), int(bbox[3]*h))
        lw = 4 if cname.startswith("compartment") else 2
        draw.rectangle([x1, y1, x2, y2], outline=col, width=lw)

        # label background
        try:
            tb = draw.textbbox((x1, max(0, y1-18)), label, font=font)
            draw.rectangle(tb, fill=col)
            draw.text((tb[0]+2, tb[1]), label,
                      fill=(0, 0, 0) if sum(col) > 400 else (255, 255, 255),
                      font=font)
        except Exception:
            draw.text((x1+2, max(0, y1-16)), label, fill=col, font=font)

    return img


# ═══════════════════════════════════════════════════════════════════════════════
# IFTTT RULE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_ifttt_engine(detections: List[Dict], rules: List[Dict],
                     conf_thresh: float, workers: int) -> Dict:
    detected     = {d["class_name"] for d in detections if d["confidence"] >= conf_thresh}
    below_thresh = {d["class_name"] for d in detections if d["confidence"] <  conf_thresh}

    locks      = sum(1 for d in detections if d["class_name"] in
                     {"personal_lock", "permit_lock"} and d["confidence"] >= conf_thresh)
    ind_lit    = "indicator_lit"   in detected
    ind_fault  = "indicator_fault" in detected
    ind_off    = "indicator_off"   in detected

    rule_results = []
    for rule in rules:
        v, reason = "PASS", ""
        if rule["indicator_only"]:
            if ind_fault:
                v, reason = "STOP",   "indicator_fault — mandatory manual verification"
            elif ind_lit:
                v, reason = "FAIL",   "indicator_lit — voltage present"
            elif not ind_off:
                v, reason = "REVIEW", "No indicator detected — cannot confirm safe state"
            else:
                reason = "All visible indicators OFF"
        else:
            iso = rule["isolation_pt"]
            if iso in detected:
                reason = f"{iso} detected at ≥{conf_thresh:.0%}"
                if rule["require_lock"] and locks == 0:
                    v, reason = "FAIL", "No locks detected — isolation not secured"
                elif rule["require_lock"] and locks < workers:
                    v, reason = "REVIEW", f"{locks} lock(s) but {workers} worker(s) signed on"
            elif iso in below_thresh:
                v, reason = "REVIEW", f"{iso} below {conf_thresh:.0%} threshold"
            else:
                danger = (iso or "").replace("_open","_closed").replace("_withdrawn","_inserted").replace("_applied","_removed")
                if danger in detected:
                    v, reason = "FAIL", f"{danger} detected — NOT in safe state"
                else:
                    v, reason = "REVIEW", f"{iso} not detected — state unknown"

        rule_results.append({"rule": rule, "verdict": v, "reason": reason})

    verdicts = [r["verdict"] for r in rule_results]
    if "STOP" in verdicts or ind_fault:
        overall, icon, msg = "STOP",            "🛑", "STOP — Manual verification MANDATORY before entry"
    elif "FAIL" in verdicts or ind_lit:
        overall, icon, msg = "DISCREPANCY",     "❌", "DISCREPANCY — Isolation incomplete. Do NOT enter."
    elif "REVIEW" in verdicts or below_thresh:
        overall, icon, msg = "REVIEW REQUIRED", "⚠️", "REVIEW REQUIRED — Manual verification needed"
    else:
        overall, icon, msg = "ALL CLEAR",       "✅", "ALL CLEAR — All isolation points verified"

    if overall == "ALL CLEAR" and below_thresh:
        overall, icon, msg = "REVIEW REQUIRED", "⚠️", \
            f"REVIEW REQUIRED — {len(below_thresh)} detection(s) below {conf_thresh:.0%}"

    return {
        "overall": overall, "icon": icon, "msg": msg,
        "rule_results": rule_results,
        "lock_count": locks, "workers": workers,
        "ind_lit": ind_lit, "ind_fault": ind_fault, "ind_off": ind_off,
        "detected": list(detected), "below": list(below_thresh),
    }


def infer_compartment_states(detections: List[Dict], conf_thresh: float) -> Dict[str, str]:
    detected = {d["class_name"] for d in detections if d["confidence"] >= conf_thresh}
    open_cls  = {"circuit_breaker_open","isolator_switch_open","fuse_withdrawn",
                 "earth_switch_applied","disconnect_open"}
    close_cls = {"circuit_breaker_closed","isolator_switch_closed","fuse_inserted",
                 "earth_switch_removed","disconnect_closed"}
    has_open  = bool(detected & open_cls)
    has_close = bool(detected & close_cls)
    states = {}
    for i in range(1, 11):
        k = f"compartment_{i}"
        if k not in detected:
            states[k] = "unknown"
        elif has_open and not has_close:
            states[k] = "isolated"
        elif has_close:
            states[k] = "energised"
        else:
            states[k] = "unknown"
    return states


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE ONNX ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_onnx_model(model_path: str):
    if not _ONNX_AVAILABLE:
        return None
    try:
        sys.path.insert(0, str(Path(__file__).parent / "yolo_training"))
        from inference import MOSLockInference  # noqa: F401
        return MOSLockInference(model_path, conf_thresh=0.90)
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None


def run_live_inference(engine, source, conf_thresh: float) -> List[Dict]:
    if engine is None:
        return []
    try:
        engine.set_confidence(conf_thresh)
        return [
            {"class_name": d.class_name, "confidence": d.confidence,
             "bbox": tuple(v / engine.imgsz for v in d.bbox),
             "label": d.class_name.replace("_", " ")}
            for d in engine.infer(source)
        ]
    except Exception as e:
        st.error(f"Inference error: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> Dict:
    with st.sidebar:
        st.title(f"{APP_NAME} V3")
        st.caption(f"{COMPANY} — {STRAPLINE}")
        st.divider()

        st.subheader("Mode")
        live_mode = st.toggle("Live Mode (requires ONNX model)", value=False)
        if live_mode and not _ONNX_AVAILABLE:
            st.warning("Install onnxruntime:\n```\npip install onnxruntime numpy\n```")
            live_mode = False

        st.divider()
        st.subheader("Permit")
        permit_id = st.text_input("Permit ID",
                                   value="HV-" + datetime.now().strftime("%y-%m-%d-%H-%M"))
        site      = st.text_input("Site", value="BST008 — Demo Site")
        workers   = st.number_input("Workers Signed On", min_value=0, max_value=20, value=1)
        step      = st.select_slider("12-Step Progress", options=list(range(1, 13)), value=5)

        st.divider()
        demo_choice = None
        if not live_mode:
            st.subheader("Demo Scenario")
            demo_choice = st.selectbox("Select scenario", list(DEMO_IMAGES.keys()), index=1)

        model_path = "./models/moslock_yolov8n.onnx"
        use_camera = "Camera"
        if live_mode:
            st.subheader("Live Mode")
            model_path = st.text_input("ONNX Model Path", value=model_path)
            use_camera = st.radio("Input source", ["Camera", "Upload file"])

        st.divider()
        st.subheader("Confidence")
        conf_pct = st.slider("Minimum confidence", 50, 99, 90, 1, format="%d%%")

        st.divider()
        st.subheader("Standards")
        st.caption("GCAA Fatal Hazard Protocol 7\nAS/NZS 4836:2023\nAS 4871.1")

    return {
        "live_mode":   live_mode,
        "demo_choice": demo_choice,
        "model_path":  model_path,
        "use_camera":  use_camera,
        "conf_thresh": conf_pct / 100,
        "workers":     int(workers),
        "step":        int(step),
        "permit_id":   permit_id,
        "site":        site,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

def tab_detection(ctrl: Dict, detections: List[Dict],
                  annotated_img: Optional["PILImage.Image"]):
    st.header("Detection Results")

    mode_str = "🟢 Live Mode" if ctrl["live_mode"] else "🔵 Demo Mode"
    st.caption(f"{mode_str}  ·  Confidence threshold: {ctrl['conf_thresh']:.0%}")

    if annotated_img is not None:
        demo_label = DEMO_IMAGES.get(ctrl.get("demo_choice", ""), {}).get("label", "")
        st.image(annotated_img, use_container_width=True, caption=demo_label)
    else:
        st.info("No image loaded. Select a demo scenario in the sidebar or enable Live Mode.")

    st.subheader("Detections")
    if not detections:
        st.info("No detections.")
        return

    if _PANDAS:
        rows = [{
            "Class":       d["class_name"],
            "Label":       d.get("label", d["class_name"]),
            "Confidence":  f"{d['confidence']:.1%}",
            "Above thresh": "✅" if d["confidence"] >= ctrl["conf_thresh"] else "⚠️ Below",
        } for d in detections]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        for d in detections:
            flag = "✅" if d["confidence"] >= ctrl["conf_thresh"] else "⚠️"
            st.text(f"{flag}  {d['class_name']}  —  {d['confidence']:.1%}")


def tab_verification(ctrl: Dict, detections: List[Dict], vd: Dict):
    st.header("Isolation Verification")

    # ── 12-step progress ──────────────────────────────────────────────────────
    st.subheader("12-Step Process (GCAA Fatal Hazard Protocol 7)")
    step = ctrl["step"]
    cols = st.columns(12)
    for i, (num, label, phase) in enumerate(TWELVE_STEPS):
        with cols[i]:
            if num < step:
                st.success(str(num), icon="✅")
            elif num == step:
                st.info(str(num))
            else:
                st.write(f"**{num}**")
    current_label = TWELVE_STEPS[step - 1][1]
    st.caption(f"Current step: **{step} — {current_label}**")
    st.divider()

    # ── Permit info ───────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Permit ID",   ctrl["permit_id"])
    c2.metric("Site",        ctrl["site"])
    c3.metric("Workers",     str(ctrl["workers"]))
    st.divider()

    # ── Overall verdict ───────────────────────────────────────────────────────
    overall = vd["overall"]
    icon    = vd["icon"]
    msg     = vd["msg"]

    if overall == "ALL CLEAR":
        st.success(f"{icon}  {msg}", icon="✅")
    elif overall == "STOP":
        st.error(f"{icon}  {msg}", icon="🛑")
    elif overall == "DISCREPANCY":
        st.error(f"{icon}  {msg}", icon="❌")
    else:
        st.warning(f"{icon}  {msg}", icon="⚠️")

    if vd["ind_fault"]:
        st.error(
            "🛑  STOP — indicator_fault detected.  "
            "Live line indicator is damaged or obscured.  "
            "MANDATORY manual verification with calibrated test instrument before entry.",
            icon="🛑"
        )

    st.divider()

    # ── KPI metrics ───────────────────────────────────────────────────────────
    st.subheader("Summary")
    k1, k2, k3, k4 = st.columns(4)
    n_det    = len(vd["detected"])
    locks    = vd["lock_count"]
    workers  = vd["workers"]
    n_below  = len(vd["below"])
    n_rules  = len(vd["rule_results"])
    n_pass   = sum(1 for r in vd["rule_results"] if r["verdict"] == "PASS")

    k1.metric("Classes Detected",           n_det)
    k2.metric("Locks / Workers Signed On",  f"{locks} / {workers}",
              delta="OK" if locks >= workers > 0 else ("Missing locks" if locks < workers else None))
    k3.metric(f"Below {ctrl['conf_thresh']:.0%} Threshold", n_below,
              delta="OK" if n_below == 0 else f"{n_below} need review")
    k4.metric("Rules PASS",                 f"{n_pass} / {n_rules}",
              delta="All passed" if n_pass == n_rules else f"{n_rules - n_pass} failed/review")

    st.divider()

    # ── Lock balance ──────────────────────────────────────────────────────────
    st.subheader("Lock Count")
    diff = locks - workers
    if workers == 0:
        st.info(f"🔒  {locks} lock(s) detected — no workers signed on yet.")
    elif diff == 0:
        st.success(f"🔒  {locks} lock(s) present — matches {workers} worker(s) signed on.")
    elif diff < 0:
        st.error(f"🔒  {locks} lock(s) detected but {workers} worker(s) signed on — {abs(diff)} lock(s) missing.")
    else:
        st.warning(f"🔒  {locks} lock(s) detected, {workers} worker(s) signed on — {diff} extra lock(s).")

    st.divider()

    # ── Rule results table ────────────────────────────────────────────────────
    st.subheader("Rule Verification")
    if _PANDAS:
        rows = []
        for rr in vd["rule_results"]:
            v = rr["verdict"]
            emoji = {"PASS": "✅ PASS", "FAIL": "❌ FAIL",
                     "REVIEW": "⚠️ REVIEW", "STOP": "🛑 STOP"}.get(v, v)
            rows.append({
                "Rule":        rr["rule"]["id"],
                "Name":        rr["rule"]["name"],
                "Verdict":     emoji,
                "Detail":      rr["reason"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        for rr in vd["rule_results"]:
            v = rr["verdict"]
            fn = {"PASS": st.success, "FAIL": st.error,
                  "REVIEW": st.warning, "STOP": st.error}.get(v, st.info)
            fn(f"**{rr['rule']['id']} — {rr['rule']['name']}**  |  {rr['reason']}")


def tab_compartments(detections: List[Dict], conf_thresh: float):
    st.header("Compartment Status")
    st.caption("States inferred from detected isolation points. "
               "Green = Isolated, Red = Energised, Amber = Unknown.")

    states = infer_compartment_states(detections, conf_thresh)
    detected = {d["class_name"] for d in detections}

    # Display as 2-column pairs so it's easy to read on tablet
    items = [(f"compartment_{i}", COMPARTMENT_NAMES[f"compartment_{i}"]) for i in range(1, 11)]
    for i in range(0, 10, 2):
        col_a, col_b = st.columns(2)
        for col, (key, name) in zip([col_a, col_b], items[i:i+2]):
            state = states.get(key, "unknown")
            seen  = "Detected" if key in detected else "Not detected"
            label = f"C{key.split('_')[1]}  —  {name}"
            if state == "isolated":
                col.success(f"✅  {label}\n\n*Isolated*  ·  {seen}")
            elif state == "energised":
                col.error(f"⚡  {label}\n\n*Energised*  ·  {seen}")
            else:
                col.warning(f"?  {label}\n\n*Unknown*  ·  {seen}")

    st.divider()
    if _PANDAS:
        st.subheader("Detail Table")
        rows = [{"ID": f"C{i}", "Zone": COMPARTMENT_NAMES[f"compartment_{i}"],
                 "State": states[f"compartment_{i}"].capitalize(),
                 "Detected": "Yes" if f"compartment_{i}" in detected else "—"}
                for i in range(1, 11)]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def tab_training_model():
    st.header("Training Model — IFTTT Rules")
    st.info(
        "The Training Model defines what a correctly isolated substation looks like. "
        "Each rule maps an isolation point to a required state. "
        "Configured by the authorised electrical engineer."
    )

    st.subheader("Active Rules")
    for rule in DEFAULT_RULES:
        with st.expander(f"{rule['id']}  —  {rule['name']}"):
            st.write(rule["description"])
            c1, c2 = st.columns(2)
            c1.write(f"**Isolation point:** `{rule.get('isolation_pt', 'N/A')}`")
            c2.write(f"**Requires lock:** {'Yes' if rule.get('require_lock') else 'No'}")

    st.divider()
    st.subheader("Add Custom Rule (this session)")
    with st.form("add_rule_form", clear_on_submit=True):
        open_pts = [n for n in CLASS_NAMES
                    if any(s in n for s in ("_open","_withdrawn","_applied"))]
        col1, col2 = st.columns(2)
        new_iso  = col1.selectbox("Isolation state required", open_pts)
        new_comp = col2.selectbox("Compartment", [f"compartment_{i}" for i in range(1,11)])
        new_lock = col1.checkbox("Require lock", value=True)
        new_name = col2.text_input("Rule name", value="Custom Rule")
        if st.form_submit_button("Add Rule"):
            st.success(f"Rule added: {new_name} — requires {new_iso} on {new_comp}")


def tab_permit(ctrl: Dict, vd: Dict):
    st.header("High Voltage Access Permit")

    # Permit summary
    st.subheader(f"Permit: {ctrl['permit_id']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Site",     ctrl["site"])
    c2.metric("Generated", datetime.now().strftime("%d %b %Y %H:%M"))
    c3.metric("Verdict",  f"{vd['icon']}  {vd['overall']}")

    st.divider()

    # 12-step reference
    with st.expander("GCAA 12-Step Isolation Reference"):
        for num, label, phase in TWELVE_STEPS:
            done = num < ctrl["step"]
            curr = num == ctrl["step"]
            mark = "✅" if done else ("▶️" if curr else "⏳")
            st.write(f"{mark}  **Step {num}** — {label}  *(_{phase}_)*")

    st.divider()

    # Worker sign-on register
    st.subheader("Worker Sign-On Register")
    st.caption("Each worker places their personal lock on the lockout station when signing on.")

    if "workers_register" not in st.session_state:
        st.session_state.workers_register = []

    with st.form("sign_on_form", clear_on_submit=True):
        col1, col2 = st.columns([3, 2])
        name    = col1.text_input("Worker name (printed)")
        contact = col2.text_input("Contact No.")
        if st.form_submit_button("Sign On 🔒") and name:
            st.session_state.workers_register.append({
                "Name": name, "Contact": contact,
                "Signed On": datetime.now().strftime("%H:%M"),
                "Signed Off": "—", "Status": "Signed On",
            })
            st.success(f"✅  {name} signed on at {datetime.now().strftime('%H:%M')}")

    reg = st.session_state.workers_register
    if reg:
        if _PANDAS:
            st.dataframe(pd.DataFrame(reg), use_container_width=True, hide_index=True)
        signed_on = sum(1 for r in reg if r["Status"] == "Signed On")
        if signed_on != vd["lock_count"] and vd["lock_count"] > 0:
            st.warning(
                f"⚠️  Register shows {signed_on} worker(s) signed on "
                f"but {vd['lock_count']} lock(s) detected. Reconcile before proceeding."
            )

        st.write("**Sign Off:**")
        for i, w in enumerate(reg):
            if w["Status"] == "Signed On":
                if st.button(f"Sign Off: {w['Name']}", key=f"so_{i}"):
                    st.session_state.workers_register[i]["Signed Off"] = \
                        datetime.now().strftime("%H:%M")
                    st.session_state.workers_register[i]["Status"] = "Signed Off"
                    st.rerun()
    else:
        st.info("No workers signed on yet.")

    st.divider()

    # Isolation drawing
    with st.expander("Single Line Diagram Reference"):
        p = find_image("Isolation drawing.jpg")
        if p:
            st.image(str(p), use_container_width=True,
                     caption="BST008 HV Single Line Diagram")
        else:
            st.info("Place 'Isolation drawing.jpg' in the demo_images/ folder.")


def tab_about():
    st.header(f"About {APP_NAME} V3")
    st.subheader(COMPANY)
    st.write(f"*{STRAPLINE}*")
    st.divider()

    st.markdown("""
**MOSLock** is an AI-assisted isolation verification tool for high-voltage
mobile substations. It uses a YOLOv8n computer vision model to detect and
classify equipment states, cross-referencing detections against a Training Model
rule set (IFTTT engine) to verify that isolation has been completed correctly per
the GCAA 12-step isolation process (Fatal Hazard Protocol 7).

### How It Works
1. **Scan** — point the tablet camera at the substation (or select a demo scenario)
2. **Detect** — YOLOv8n identifies locks, isolation points, compartments, and indicators
3. **Verify** — IFTTT engine checks detections against Training Model rules
4. **Verdict** — ALL CLEAR / DISCREPANCY / REVIEW REQUIRED / STOP
5. **Record** — Permit register logs who signed on, when, and what was verified

### ⚠️ Safety
MOSLock is a verification **aid**, not a replacement for the physical isolation
process. The GCAA 12-step procedure must always be completed in full.
If `indicator_fault` is detected, mandatory manual verification with a calibrated
test instrument is required before entry.

### Standards
- Glencore GCAA Fatal Hazard Protocol 7
- AS/NZS 4836:2023 Safe Working on LV Electrical Installations
- AS 4871.1 Electrical Equipment for Coal Mines
""")
    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("Version",       APP_VERSION)
    c1.metric("Detection Classes", "25")
    c2.metric("Target Device", "Samsung Galaxy Tab S10")
    c2.metric("Model",         "YOLOv8n (ONNX)")

    with st.expander("Demo Mode explained"):
        st.write("""
Demo Mode shows pre-loaded detection results from the V2 trial site photos
(BST008 substation, Glencore Coal Assets Australia).
No ONNX model is required.

Switch to **Live Mode** in the sidebar to use the camera and run real-time
inference (requires `./models/moslock_yolov8n.onnx`).

Train the model:
```bash
pip install ultralytics
cd yolo_training
python generate_synthetic_samples.py
python train.py
python export.py
```
""")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title=f"{APP_NAME} V3 — {COMPANY}",
        page_icon="🔒",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Session state init
    if "workers_register" not in st.session_state:
        st.session_state.workers_register = []

    ctrl = render_sidebar()

    # ── App header ────────────────────────────────────────────────────────────
    st.title(f"🔒 {APP_NAME} V3")
    mode_str = "🟢 Live Mode" if ctrl["live_mode"] else "🔵 Demo Mode"
    st.caption(f"{mode_str}  ·  {ctrl['site']}  ·  {COMPANY}  ·  {STRAPLINE}")
    st.divider()

    # ── Acquire detections ────────────────────────────────────────────────────
    detections:    List[Dict]              = []
    annotated_img: Optional[PILImage.Image] = None

    if ctrl["live_mode"]:
        engine = load_onnx_model(ctrl["model_path"])
        if engine is None:
            if not _ONNX_AVAILABLE:
                st.warning("**onnxruntime not installed** — showing Demo Mode.\n\n"
                           "Install: `pip install onnxruntime numpy`")
            else:
                st.warning(f"**Model not found:** `{ctrl['model_path']}`  — showing Demo Mode.\n\n"
                           "Run `yolo_training/train.py` then `export.py` to generate the model.")
            ctrl["live_mode"] = False

        if ctrl["live_mode"]:
            st.info("📷  Point the tablet at the substation, then capture.")
            src = None
            if ctrl["use_camera"] == "Camera":
                cam = st.camera_input("Capture substation image")
                if cam:
                    src = PILImage.open(cam).convert("RGB")
            else:
                up = st.file_uploader("Upload photo",
                                       type=["jpg","jpeg","png","webp"])
                if up:
                    src = PILImage.open(up).convert("RGB")

            if src is not None:
                detections    = run_live_inference(engine, src, ctrl["conf_thresh"])
                annotated_img = draw_detections(src, detections, ctrl["conf_thresh"])

    if not ctrl["live_mode"]:
        key  = ctrl.get("demo_choice") or list(DEMO_IMAGES.keys())[1]
        data = DEMO_IMAGES.get(key, {})
        detections = data.get("detections", [])
        img_path   = find_image(data.get("file", ""))
        if img_path and _PIL:
            try:
                src = PILImage.open(str(img_path)).convert("RGB")
                annotated_img = draw_detections(src, detections, ctrl["conf_thresh"])
            except Exception:
                annotated_img = None
        else:
            annotated_img = None

    # ── IFTTT engine ──────────────────────────────────────────────────────────
    vd = run_ifttt_engine(detections, DEFAULT_RULES,
                          ctrl["conf_thresh"], ctrl["workers"])

    # ── STOP banner — always visible at top ───────────────────────────────────
    if vd["ind_fault"]:
        st.error(
            "🛑  **STOP — indicator_fault detected.**  "
            "Live line indicator damaged or obscured.  "
            "MANDATORY manual verification with calibrated instrument before entry.",
            icon="🛑"
        )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📷 Detection",
        "✅ Verification",
        "🏭 Compartments",
        "📐 Training Model",
        "📋 Permit",
        "ℹ️ About",
    ])

    with t1: tab_detection(ctrl, detections, annotated_img)
    with t2: tab_verification(ctrl, detections, vd)
    with t3: tab_compartments(detections, ctrl["conf_thresh"])
    with t4: tab_training_model()
    with t5: tab_permit(ctrl, vd)
    with t6: tab_about()

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        f"{APP_NAME} V3  ·  {COMPANY}  ·  v{APP_VERSION}  ·  "
        f"Permit: {ctrl['permit_id']}  ·  "
        "GCAA Fatal Hazard Protocol 7  ·  AS/NZS 4836:2023  ·  AS 4871.1  ·  "
        f"*{STRAPLINE}*"
    )


if __name__ == "__main__":
    main()
