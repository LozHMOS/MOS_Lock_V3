"""
MOSLock V3 — Training, Permit & AI Verification
Mapped Out Solutions® | mos | Lock®  |  "We will always have a plan."

V2 base kept intact. V3 adds: YOLO class mapping in Training Model (Step 2/3),
AI/AR Verification Engine with Demo + Live ONNX camera mode (Step 5).
Standards: STD0930 | FRM1277 | Glencore Fatal Hazard Protocol 7 | AS/NZS 4836
"""

import streamlit as st
from PIL import Image
import datetime
import pandas as pd
import sys
from pathlib import Path

# ── Lazy imports — only needed for Live ONNX mode ────────────────────────────
_ONNX_AVAILABLE = False
try:
    import onnxruntime as _ort   # noqa: F401
    import numpy as _np          # noqa: F401
    _ONNX_AVAILABLE = True
except ImportError:
    pass

_PIL_DRAW = False
try:
    from PIL import ImageDraw, ImageFont
    _PIL_DRAW = True
except ImportError:
    pass

_INFERENCE_CLASS = None
try:
    sys.path.insert(0, str(Path(__file__).parent / "yolo_training"))
    from inference import MOSLockInference as _MI
    _INFERENCE_CLASS = _MI
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
MOS_CYAN = "#3ECFCF"
MOS_NAVY = "#1e3a5f"

DEVICE_CLASS_MAP = {
    "Circuit Breaker": ("circuit_breaker_open",  "circuit_breaker_closed"),
    "Isolator Switch": ("isolator_switch_open",   "isolator_switch_closed"),
    "Fuse":            ("fuse_withdrawn",          "fuse_inserted"),
    "Disconnect":      ("disconnect_open",         "disconnect_closed"),
    "Earth Switch":    ("earth_switch_applied",    "earth_switch_removed"),
}

BBOX_COLOUR = {
    "personal_lock":          "#FF6B35",
    "permit_lock":            "#F5A623",
    "circuit_breaker_open":   "#28a745",
    "circuit_breaker_closed": "#dc3545",
    "isolator_switch_open":   "#28a745",
    "isolator_switch_closed": "#dc3545",
    "fuse_withdrawn":         "#28a745",
    "fuse_inserted":          "#dc3545",
    "earth_switch_applied":   "#007bff",
    "earth_switch_removed":   "#ffc107",
    "disconnect_open":        "#28a745",
    "disconnect_closed":      "#dc3545",
    "indicator_lit":          "#dc3545",
    "indicator_off":          "#28a745",
    "indicator_fault":        "#6f42c1",
}
for _c in [f"compartment_{i}" for i in range(1, 11)]:
    BBOX_COLOUR[_c] = MOS_CYAN

IMAGE_DIRS = [
    Path("."),
    Path("demo_images"),
    Path("/mnt/session/uploads/workspace/images"),
    Path(__file__).parent / "demo_images",
    Path(__file__).parent,
]

DEMO_SCENARIOS = {
    "Substation — Base State": {
        "file": "Substation.jpeg",
        "label": "BST008 — No isolation (base state)",
        "detections": [
            {"class_name": "compartment_1", "confidence": 0.97, "bbox": (0.02, 0.05, 0.21, 0.98), "label": "HV Incoming"},
            {"class_name": "compartment_2", "confidence": 0.95, "bbox": (0.22, 0.05, 0.52, 0.98), "label": "Transformer"},
            {"class_name": "compartment_3", "confidence": 0.96, "bbox": (0.53, 0.05, 0.98, 0.98), "label": "LV Outgoing"},
        ],
    },
    "Substation — Training Model": {
        "file": "Substation Trained.jpeg",
        "label": "BST008 — Training model annotations",
        "detections": [
            {"class_name": "compartment_1",        "confidence": 0.97, "bbox": (0.02, 0.05, 0.21, 0.98), "label": "Compartment 1"},
            {"class_name": "compartment_2",        "confidence": 0.95, "bbox": (0.22, 0.05, 0.52, 0.98), "label": "Compartment 2"},
            {"class_name": "compartment_3",        "confidence": 0.96, "bbox": (0.53, 0.05, 0.98, 0.98), "label": "Compartment 3"},
            {"class_name": "isolator_switch_open", "confidence": 0.93, "bbox": (0.03, 0.55, 0.09, 0.75), "label": "Isolation Pt 1"},
            {"class_name": "circuit_breaker_open", "confidence": 0.91, "bbox": (0.26, 0.28, 0.32, 0.46), "label": "Isolation Pt 2"},
            {"class_name": "disconnect_open",      "confidence": 0.94, "bbox": (0.52, 0.10, 0.60, 0.35), "label": "Isolation Pt 3"},
            {"class_name": "indicator_off",        "confidence": 0.96, "bbox": (0.72, 0.47, 0.79, 0.60), "label": "Indicator 1"},
        ],
    },
    "Substation — Isolated LV": {
        "file": "Substation Isolated LV.jpeg",
        "label": "BST008 — LV compartment isolated",
        "detections": [
            {"class_name": "compartment_1",        "confidence": 0.97, "bbox": (0.02, 0.05, 0.21, 0.98), "label": "C1 Energised"},
            {"class_name": "compartment_2",        "confidence": 0.95, "bbox": (0.22, 0.05, 0.52, 0.98), "label": "C2 Energised"},
            {"class_name": "compartment_3",        "confidence": 0.96, "bbox": (0.53, 0.05, 0.98, 0.98), "label": "C3 Isolated"},
            {"class_name": "isolator_switch_open", "confidence": 0.93, "bbox": (0.03, 0.55, 0.09, 0.75), "label": "Isolation Pt 1"},
            {"class_name": "circuit_breaker_open", "confidence": 0.91, "bbox": (0.26, 0.28, 0.32, 0.46), "label": "Isolation Pt 2"},
            {"class_name": "disconnect_open",      "confidence": 0.94, "bbox": (0.52, 0.10, 0.60, 0.35), "label": "Isolation Pt 3"},
            {"class_name": "personal_lock",        "confidence": 0.96, "bbox": (0.54, 0.22, 0.59, 0.30), "label": "Personal Lock"},
            {"class_name": "indicator_off",        "confidence": 0.96, "bbox": (0.72, 0.47, 0.79, 0.60), "label": "Indicator Off"},
        ],
    },
    "Substation — Locked Out": {
        "file": "Sub Locked Out.jpg",
        "label": "BST008 — Lock at isolation point 3",
        "detections": [
            {"class_name": "compartment_1",   "confidence": 0.97, "bbox": (0.02, 0.05, 0.21, 0.98), "label": "Compartment 1"},
            {"class_name": "compartment_2",   "confidence": 0.95, "bbox": (0.22, 0.05, 0.52, 0.98), "label": "Compartment 2"},
            {"class_name": "compartment_3",   "confidence": 0.96, "bbox": (0.53, 0.05, 0.98, 0.98), "label": "Compartment 3"},
            {"class_name": "disconnect_open", "confidence": 0.94, "bbox": (0.52, 0.10, 0.60, 0.35), "label": "Isolation Pt 3"},
            {"class_name": "personal_lock",   "confidence": 0.97, "bbox": (0.50, 0.25, 0.57, 0.35), "label": "Lock at Pt 3"},
        ],
    },
}

_MOS_LOGO_SIDEBAR = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 185 42" width="185" height="42">'
    '<circle cx="21" cy="21" r="19" fill="#2d2926"/>'
    '<polyline points="11,29 11,15 17,15 17,29" fill="none" stroke="white" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>'
    '<polyline points="25,29 25,15 31,15 31,29" fill="none" stroke="white" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>'
    '<circle cx="11" cy="15" r="2.5" fill="white"/>'
    '<circle cx="31" cy="15" r="2.5" fill="white"/>'
    '<text x="48" y="16" font-family="Arial,sans-serif" font-size="8.5" font-weight="300" fill="#888">mapped out solutions&#174;</text>'
    '<text x="48" y="33" font-family="Arial,sans-serif" font-size="14" font-weight="300" fill="#2d2926">mos &#124; '
    '<tspan fill="#3ECFCF" font-weight="700">Lock</tspan>'
    '<tspan font-size="8" dy="-4">&#174;</tspan></text>'
    '</svg>'
)

_MOS_LOGO_MAIN = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 230 52" width="230" height="52">'
    '<circle cx="26" cy="26" r="24" fill="#2d2926"/>'
    '<polyline points="14,36 14,18 21,18 21,36" fill="none" stroke="white" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round"/>'
    '<polyline points="31,36 31,18 38,18 38,36" fill="none" stroke="white" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round"/>'
    '<circle cx="14" cy="18" r="3" fill="white"/>'
    '<circle cx="38" cy="18" r="3" fill="white"/>'
    '<text x="60" y="20" font-family="Arial,sans-serif" font-size="10" font-weight="300" fill="#555">mapped out solutions&#174;</text>'
    '<text x="60" y="40" font-family="Arial,sans-serif" font-size="17" font-weight="300" fill="#2d2926">mos &#124; '
    '<tspan fill="#3ECFCF" font-weight="700">Lock</tspan>'
    '<tspan font-size="10" dy="-4">&#174;</tspan></text>'
    '</svg>'
)

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MOSLock V3 – Training & Permit",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS  (V2 verbatim + V3 cyan/MOS additions appended) ────────────────────────
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        height: 48px; padding: 0 16px;
        background-color: #e8ecf0;
        border-radius: 6px 6px 0 0;
        font-weight: 600; font-size: 13px; color: #212529;
    }
    div[data-testid="metric-container"] {
        background: #f0f4f8; border-left: 4px solid #1e3a5f;
        border-radius: 6px; padding: 12px; color: #212529;
    }
    .section-header {
        background: linear-gradient(90deg, #1e3a5f, #2d6096);
        color: white; padding: 8px 16px; border-radius: 6px;
        margin: 12px 0 8px 0; font-weight: 700; font-size: 15px;
    }
    .risk-low  { background:#d4edda; border-left:4px solid #28a745; padding:8px 14px; border-radius:4px; margin:4px 0; color:#212529 !important; }
    .risk-medium { background:#fff3cd; border-left:4px solid #ffc107; padding:8px 14px; border-radius:4px; margin:4px 0; color:#212529 !important; }
    .risk-high { background:#f8d7da; border-left:4px solid #dc3545; padding:8px 14px; border-radius:4px; margin:4px 0; color:#212529 !important; }
    .lock-counter { background:#1e3a5f; color:white; border-radius:8px; padding:12px; text-align:center; font-size:20px; font-weight:700; }
    .step-complete { color:#28a745; font-weight:700; }
    .step-active   { color:#007bff; font-weight:700; }
    .step-pending  { color:#adb5bd; }
    /* V3 additions */
    .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color:#3ECFCF !important; color:#1e3a5f !important; }
    div[data-testid="metric-container"] { border-left-color:#3ECFCF; }
    .mos-header-bar  { border-bottom:2px solid #3ECFCF; padding-bottom:10px; margin-bottom:10px; }
    .mos-sidebar-bar { border-bottom:2px solid #3ECFCF; padding-bottom:8px;  margin-bottom:10px; }
    .yolo-badge        { display:inline-block; background:#e6fafa; border:1px solid #3ECFCF; border-radius:4px; padding:2px 8px; font-size:11px; font-family:monospace; color:#1e3a5f; margin:2px; }
    .yolo-badge-unsafe { background:#fde8e6; border-color:#dc3545; color:#7a1a14; }
    .verify-allclear  { background:linear-gradient(90deg,#28a745,#20c65a); border-radius:8px; padding:14px 20px; color:white; font-size:18px; font-weight:700; text-align:center; margin:10px 0; }
    .verify-overall-review { background:#ffc107; border-radius:8px; padding:14px 20px; color:#212529; font-size:18px; font-weight:700; text-align:center; margin:10px 0; }
    .verify-overall-fail   { background:#dc3545; border-radius:8px; padding:14px 20px; color:white; font-size:18px; font-weight:700; text-align:center; margin:10px 0; }
    .verify-overall-stop   { background:#1e3a5f; border:3px solid #3ECFCF; border-radius:8px; padding:14px 20px; color:white; font-size:18px; font-weight:700; text-align:center; margin:10px 0; }
    .verify-pass  { background:#d4edda; border-left:4px solid #28a745; padding:8px 14px; border-radius:4px; margin:4px 0; color:#212529; }
    .verify-fail  { background:#f8d7da; border-left:4px solid #dc3545; padding:8px 14px; border-radius:4px; margin:4px 0; color:#212529; }
    .verify-review { background:#fff3cd; border-left:4px solid #ffc107; padding:8px 14px; border-radius:4px; margin:4px 0; color:#212529; }
    .verify-stop  { background:#1e3a5f; border-left:5px solid #3ECFCF; padding:10px 16px; border-radius:4px; margin:4px 0; color:white; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ───────────────────────────────────────────────────────────────
defaults = {
    'training_complete': False, 'rules': [], 'lock_photo': None,
    'show_permit_preview': None, 'permit_status': 'Draft',
    'parts_complete': set(), 'signed_on_workers': [],
    'lock_count': 0, 'active_permits': 2,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── HELPERS ────────────────────────────────────────────────────────────────────
def safe_image(path, caption="", **kwargs):
    try:
        img = Image.open(path)
        st.image(img, caption=caption, **kwargs)
        return True
    except FileNotFoundError:
        st.info(f"📷 Image placeholder: *{path}* (place this file in the app directory)")
        return False


def find_image(filename):
    for d in IMAGE_DIRS:
        p = Path(d) / filename
        if p.exists():
            return p
    return None


def draw_detections_pil(img, detections, conf_thresh):
    if not _PIL_DRAW:
        return img
    img = img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    font = None
    for fp in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
               "/System/Library/Fonts/Helvetica.ttc", "C:/Windows/Fonts/arialbd.ttf"]:
        if Path(fp).exists():
            try:
                font = ImageFont.truetype(fp, 14); break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()
    for det in detections:
        if det["confidence"] < conf_thresh:
            continue
        cname = det["class_name"]
        bbox  = det["bbox"]
        label = f"{det.get('label', cname)}  {det['confidence']:.0%}"
        hex_c = BBOX_COLOUR.get(cname, "#888888")
        col   = (int(hex_c[1:3],16), int(hex_c[3:5],16), int(hex_c[5:7],16))
        x1,y1,x2,y2 = int(bbox[0]*w),int(bbox[1]*h),int(bbox[2]*w),int(bbox[3]*h)
        lw = 4 if cname.startswith("compartment") else (5 if cname=="indicator_fault" else 2)
        draw.rectangle([x1,y1,x2,y2], outline=col, width=lw)
        try:
            tb = draw.textbbox((x1, max(0,y1-18)), label, font=font)
            draw.rectangle(tb, fill=col)
            lum = col[0]*0.299 + col[1]*0.587 + col[2]*0.114
            draw.text((tb[0]+2,tb[1]), label, fill=(0,0,0) if lum>140 else (255,255,255), font=font)
        except Exception:
            draw.text((x1+2,max(0,y1-16)), label, fill=col, font=font)
    return img


def build_ip_class_map(num_iso):
    result = {}
    for i in range(num_iso):
        dt = st.session_state.get(f"ip_type_{i}", "Circuit Breaker")
        result[i+1] = DEVICE_CLASS_MAP.get(dt, ("circuit_breaker_open","circuit_breaker_closed"))
    return result


def run_verification_engine(detections, rules, ip_class_map, conf_thresh):
    detected     = {d["class_name"] for d in detections if d["confidence"] >= conf_thresh}
    below_thresh = {d["class_name"] for d in detections if d["confidence"] <  conf_thresh}
    locks     = sum(1 for d in detections if d["class_name"] in {"personal_lock","permit_lock"} and d["confidence"] >= conf_thresh)
    ind_fault = "indicator_fault" in detected
    ind_lit   = "indicator_lit"   in detected
    ind_off   = "indicator_off"   in detected
    rule_results = []
    for rule in rules:
        iso_num = rule.get("iso", 1)
        cond    = rule.get("cond", "Isolated")
        risk    = rule.get("risk", "Safe to Access")
        safe_c, unsafe_c = ip_class_map.get(iso_num, ("circuit_breaker_open","circuit_breaker_closed"))
        expected = safe_c   if cond == "Isolated" else unsafe_c
        danger   = unsafe_c if cond == "Isolated" else safe_c
        ind_exp  = rule.get("ind_status", "Off")
        if ind_fault:
            v, reason = "STOP",   "indicator_fault — mandatory manual verification required"
        elif expected in detected:
            if risk == "Safe to Access" and locks == 0:
                v, reason = "REVIEW", f"{expected} detected but no personal locks on box"
            elif ind_exp == "Off" and ind_lit:
                v, reason = "FAIL",   f"{expected} detected but indicator shows LIVE"
            else:
                v, reason = "PASS",   f"{expected} confirmed ≥{conf_thresh:.0%}"
        elif danger in detected:
            v, reason = "FAIL",   f"{danger} detected — NOT in safe state"
        elif expected in below_thresh:
            v, reason = "REVIEW", f"{expected} below {conf_thresh:.0%} threshold"
        else:
            v, reason = "REVIEW", f"{expected} not detected — state unknown"
        rule_results.append({
            "Rule": f"R{len(rule_results)+1:02d}",
            "IF  Isolation Pt": f"IP{iso_num} ({cond})",
            "Expected Class": expected,
            "Access Risk": risk,
            "Verdict": v,
            "Detail": reason,
        })
    verdicts = [r["Verdict"] for r in rule_results]
    if   "STOP"   in verdicts or ind_fault: overall = "STOP"
    elif "FAIL"   in verdicts or ind_lit:   overall = "DISCREPANCY"
    elif "REVIEW" in verdicts or below_thresh: overall = "REVIEW REQUIRED"
    elif not rule_results:                  overall = "NO RULES DEFINED"
    else:                                   overall = "ALL CLEAR"
    return {"overall": overall, "rule_results": rule_results,
            "lock_count": locks, "ind_fault": ind_fault,
            "ind_lit": ind_lit, "ind_off": ind_off, "detected": list(detected)}


def render_verification_results(vd):
    overall = vd["overall"]
    if overall == "ALL CLEAR":
        st.markdown('<div class="verify-allclear">✅  ALL CLEAR — All isolation points verified. Safe to proceed.</div>', unsafe_allow_html=True)
    elif overall == "STOP":
        st.markdown('<div class="verify-overall-stop">🛑  STOP — indicator_fault detected. MANDATORY manual verification with calibrated instrument before entry.</div>', unsafe_allow_html=True)
    elif overall == "DISCREPANCY":
        st.markdown('<div class="verify-overall-fail">❌  DISCREPANCY — Isolation incomplete. Do NOT enter.</div>', unsafe_allow_html=True)
    elif overall == "REVIEW REQUIRED":
        st.markdown('<div class="verify-overall-review">⚠️  REVIEW REQUIRED — Manual verification needed.</div>', unsafe_allow_html=True)
    else:
        st.info(f"ℹ️  {overall}")
    if vd["rule_results"]:
        st.markdown("**Rule-by-Rule Verification:**")
        st.dataframe(pd.DataFrame(vd["rule_results"]), use_container_width=True, hide_index=True)
    lc = vd["lock_count"]
    if lc > 0:
        st.markdown(f'<div class="risk-low">🔒 <strong>{lc} personal lock(s)</strong> detected on lock box.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="risk-medium">🔒 No personal locks detected — confirm lock box manually.</div>', unsafe_allow_html=True)
    if vd["ind_fault"]:
        st.markdown('<div class="verify-stop">⚠️ indicator_fault — live line indicator damaged/obscured. Manual test instrument check MANDATORY.</div>', unsafe_allow_html=True)
    elif vd["ind_lit"]:
        st.markdown('<div class="verify-fail">⚡ indicator_lit — voltage present.</div>', unsafe_allow_html=True)
    elif vd["ind_off"]:
        st.markdown('<div class="verify-pass">✅ indicator_off — no voltage on indicators.</div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_onnx_model(model_path):
    if not _ONNX_AVAILABLE:
        return None
    if _INFERENCE_CLASS is not None:
        try:
            return _INFERENCE_CLASS(model_path, conf_thresh=0.90)
        except Exception:
            pass
    try:
        import onnxruntime as ort
        return {"session": ort.InferenceSession(model_path, providers=["CPUExecutionProvider"]), "type": "raw"}
    except Exception:
        return None


def run_live_inference(engine, pil_image, conf_thresh):
    if engine is None:
        return []
    try:
        if hasattr(engine, "infer"):
            engine.set_confidence(conf_thresh)
            return [{"class_name": d.class_name, "confidence": d.confidence,
                     "bbox": (d.bbox[0]/engine.imgsz, d.bbox[1]/engine.imgsz,
                              d.bbox[2]/engine.imgsz, d.bbox[3]/engine.imgsz),
                     "label": d.class_name.replace("_"," ")} for d in engine.infer(pil_image)]
    except Exception as e:
        st.error(f"Inference error: {e}")
    return []

# ─── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<div class="mos-sidebar-bar">{_MOS_LOGO_SIDEBAR}</div>', unsafe_allow_html=True)
    st.markdown("*Safety-Critical Permit System*")
    st.markdown("---")
    st.markdown("### 👤 Current User")
    user_name = st.text_input("Name", value="J. Smith", key="sidebar_user")
    user_role = st.selectbox("Role", ["HV Permit Issuer","HV Permit Holder","HV Authorised Isolator","Electrical Engineer","Supervisor"], key="sidebar_role")
    st.markdown("---")
    st.markdown("### 🏭 Site & Shift")
    site  = st.selectbox("Mine Site",  ["Bulga Underground","Ulan Underground","Mandalong"], key="sidebar_site")
    shift = st.selectbox("Current Shift", ["Day (06:00–18:00)","Night (18:00–06:00)"], key="sidebar_shift")
    st.markdown("---")
    st.markdown("### 📊 Live Stats")
    col_s1, col_s2 = st.columns(2)
    with col_s1: st.metric("Active Permits", st.session_state.active_permits)
    with col_s2: st.metric("Locks On Box",   st.session_state.lock_count)
    if st.session_state.signed_on_workers:
        st.markdown(f"**Signed On:** {len(st.session_state.signed_on_workers)} worker(s)")
        for w in st.session_state.signed_on_workers:
            st.markdown(f"  ✅ {w}")
    st.markdown("---")
    st.markdown("### 🚨 Emergency Contacts")
    st.markdown("**Mine Rescue:** 000\n**Shift Supervisor:** Ch. 1\n**Control Room:** Ch. 3\n**First Aid:** Level 2 South")
    st.markdown("---")
    st.caption("MOSLock V3 | Mapped Out Solutions® | STD0930 / FHP-07")

# ─── MAIN HEADER ────────────────────────────────────────────────────────────────
hdr_logo, hdr_title = st.columns([1, 3])
with hdr_logo:
    st.markdown(f'<div class="mos-header-bar" style="padding-top:4px;">{_MOS_LOGO_MAIN}</div>', unsafe_allow_html=True)
with hdr_title:
    st.title("🔒 MOSLock V3 – Training & Permit System")

if not st.session_state.training_complete:
    st.warning("⚠️  **Setup Required:** Complete Equipment Training (Tab 1 – Training Model) before issuing HV Permits.")
else:
    status_icons = {"Draft":"🔵","Submitted":"🟡","Issuer Approved":"🟠","Engineer Approved":"🟠","Active":"🟢","Completed":"⚫"}
    icon = status_icons.get(st.session_state.permit_status, "⚪")
    st.success(f"✅ Equipment trained and ready.  Current Permit Status: {icon} **{st.session_state.permit_status}**")

kpi1,kpi2,kpi3,kpi4,kpi5 = st.columns(5)
with kpi1: st.metric("Active HV Permits", "2")
with kpi2: st.metric("Permits Today", "4", delta="+1")
with kpi3: st.metric("Equipment in Library", "4" if st.session_state.training_complete else "3", delta="+1" if st.session_state.training_complete else "0")
with kpi4: st.metric("TVL Observations (Week)", "3")
with kpi5: st.metric("Compliance Score", "94%" if st.session_state.training_complete else "87%", delta="+2%")
st.caption("**Demo – Underground Coal Operations** | STD0930 | Glencore Fatal Hazard Protocol 7 | AS/NZS 4836")
st.markdown("---")

# ─── TABS ────────────────────────────────────────────────────────────────────────
tab0,tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    "📊 Dashboard","🎓 Training Model","⚡ HV Access Permit",
    "👁️ TVL Observation","📋 12 Step Reference","🗄️ Equipment Library","📁 Permit History"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 0 – DASHBOARD  (V2 unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
with tab0:
    st.subheader("📊 Operational Dashboard")
    st.markdown('<div class="risk-high">🚨 <strong>SAFETY ALERT:</strong> Permit HV-2026-001 expires in 2 hours — Permit Holder must sign off or request extension.</div>', unsafe_allow_html=True)
    st.markdown("")
    st.markdown('<div class="risk-medium">⚠️ <strong>NOTICE:</strong> TVL Observation due today for Zone B Substation.</div>', unsafe_allow_html=True)
    st.markdown("")
    col_left, col_right = st.columns([2,1])
    with col_left:
        st.markdown("### 📋 Active Permits")
        for p in [
            {"id":"HV-2026-001","equipment":"Mobile Substation","holder":"J. Smith","issuer":"R. Jones","status":"Active","expires":"Today 18:00","workers":3,"locks":4},
            {"id":"HV-2026-004","equipment":"Feeder Breaker","holder":"M. Brown","issuer":"R. Jones","status":"Approved","expires":"Tomorrow 06:00","workers":2,"locks":2},
        ]:
            dot = "🟢" if p["status"]=="Active" else "🔵"
            with st.expander(f"{dot} **{p['id']}** — {p['equipment']} | Status: {p['status']} | Expires: {p['expires']}"):
                c1,c2,c3 = st.columns(3)
                with c1:
                    st.write(f"**Permit Holder:** {p['holder']}")
                    st.write(f"**Issuer:** {p['issuer']}")
                with c2:
                    st.write(f"**Workers Signed On:** {p['workers']}")
                    st.write(f"**Locks on Box:** {p['locks']}")
                with c3:
                    if p['locks']==p['workers']+1:
                        st.markdown('<div class="risk-low">✅ Lock count verified</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="risk-high">🚨 Lock count mismatch – investigate immediately</div>', unsafe_allow_html=True)
                    if st.button("View Full Permit", key=f"dash_view_{p['id']}"):
                        st.info("Full permit opens in HV Access Permit tab.")
        st.markdown("### 📈 Weekly Permit Activity")
        st.bar_chart(pd.DataFrame({"Permits Issued":[2,3,1,4,2,1,0],"Completed":[2,2,1,3,2,1,0]},index=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]))
    with col_right:
        st.markdown("### 🔒 Lock Box Status")
        st.markdown(f'<div class="lock-counter">🔒<br>{st.session_state.lock_count}<br><small>Locks on Box</small></div>', unsafe_allow_html=True)
        st.markdown("")
        st.markdown("**Quick Sign-On (Permit HV-2026-001)**")
        new_worker = st.text_input("Worker Name", key="dash_new_worker")
        if st.button("✅ Sign On", key="dash_sign_on"):
            if new_worker and new_worker not in st.session_state.signed_on_workers:
                st.session_state.signed_on_workers.append(new_worker)
                st.session_state.lock_count += 1
                st.rerun()
            elif new_worker in st.session_state.signed_on_workers:
                st.warning(f"{new_worker} is already signed on.")
        if st.session_state.signed_on_workers:
            st.markdown("**Currently Signed On:**")
            for i,w in enumerate(st.session_state.signed_on_workers):
                wc1,wc2 = st.columns([3,1])
                with wc1: st.write(f"✅ {w}")
                with wc2:
                    if st.button("Off", key=f"dash_signoff_{i}"):
                        st.session_state.signed_on_workers.remove(w)
                        if st.session_state.lock_count>0: st.session_state.lock_count -= 1
                        st.rerun()
        st.markdown("---")
        st.markdown("### ⚡ Zone Isolation Status")
        st.markdown("| Zone | Status |\n|------|--------|\n| Substation A | 🔴 Isolated |\n| Conveyor 3 | 🟢 Live |\n| HV Panel B3 | 🔴 Isolated |\n| Feeder Breaker | 🟡 Pending |")
        st.markdown("---")
        st.progress(75, text="3/4 Permits Completed Today")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – TRAINING MODEL  (V2 base + V3 YOLO class mapping + AI Verification)
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("🎓 New Equipment Training Model")
    st.markdown("Train the AI/AR model on new equipment by defining isolation points, compartments, indicators, and the IFTTT logic the model will learn. Each isolation point is mapped to a specific YOLO detection class pair.")

    current_step = 5 if st.session_state.training_complete else (3 if st.session_state.rules else 2)
    def step_icon(n):
        return "✅" if current_step>n else ("🔵" if current_step==n else "⭕")

    st.markdown(f"""
    <div style="display:flex;gap:16px;margin-bottom:16px;padding:12px;background:#f0f4f8;border-radius:8px;border-left:4px solid #3ECFCF;flex-wrap:wrap;">
        <span class="{'step-complete' if current_step>1 else 'step-active' if current_step==1 else 'step-pending'}">{step_icon(1)} Step 1: Equipment Photo</span> →
        <span class="{'step-complete' if current_step>2 else 'step-active' if current_step==2 else 'step-pending'}">{step_icon(2)} Step 2: Define Points</span> →
        <span class="{'step-complete' if current_step>3 else 'step-active' if current_step==3 else 'step-pending'}">{step_icon(3)} Step 3: Logic Rules</span> →
        <span class="{'step-complete' if current_step>4 else 'step-pending'}">{step_icon(4)} Step 4: Submit</span> →
        <span class="{'step-complete' if st.session_state.training_complete else 'step-pending'}">{'✅' if st.session_state.training_complete else '⭕'} Step 5: AI Verification</span>
    </div>
    """, unsafe_allow_html=True)

    # Step 1
    st.markdown('<div class="section-header">Step 1 – Equipment Photo & Details</div>', unsafe_allow_html=True)
    col_photo, col_meta = st.columns([2,1])
    with col_photo:
        safe_image("Substation.jpeg", caption="Equipment – Mobile Substation (clean photo for AI training)", use_container_width=True)
        custom_photo = st.file_uploader("Upload a custom equipment photo", type=["jpg","jpeg","png"], key="custom_photo")
        if custom_photo:
            st.image(custom_photo, caption="Custom Equipment Photo", use_container_width=True)
            st.success("✅ Custom photo loaded for training.")
    with col_meta:
        st.markdown("**Equipment Details**")
        eq_name     = st.text_input("Equipment Name",   value="Mobile Substation",    key="eq_name")
        eq_type     = st.selectbox("Voltage Class",     ["HV (>1000 V)","LV (≤1000 V)","Control Circuit"], key="eq_type")
        eq_voltage  = st.text_input("System Voltage",   value="11 kV",                key="eq_voltage")
        eq_location = st.text_input("Typical Location", value="Zone B – Underground", key="eq_location")
        eq_make     = st.text_input("Make / Model",                                   key="eq_make")
        eq_serial   = st.text_input("Serial / Asset No.",                             key="eq_serial")

    # Step 2
    st.markdown('<div class="section-header">Step 2 – Define Isolation Points, Compartments & Indicators</div>', unsafe_allow_html=True)
    st.info("💡 **Full app:** Tap directly on the equipment photo to pin each point. Each isolation point is assigned a YOLO detection class pair — the AI uses these to recognise safe vs unsafe states in real-time.")
    c1,c2,c3,c4 = st.columns(4)
    with c1: num_iso   = st.selectbox("Isolation Points",[1,2,3,4,5],index=2,key="num_iso")
    with c2: num_comp  = st.selectbox("Compartments",   [1,2,3,4,5],index=2,key="num_comp")
    with c3: num_ind   = st.selectbox("Indicators",     [1,2,3],    index=1,key="num_ind")
    with c4: num_earth = st.selectbox("Earth Points",   [0,1,2,3,4],index=1,key="num_earth")

    st.markdown("**Label each Isolation Point and assign its YOLO detection class pair:**")
    iso_cols = st.columns(num_iso)
    for i,col in enumerate(iso_cols):
        with col:
            st.markdown(f"**IP {i+1}**")
            dt = st.selectbox("Device Type", list(DEVICE_CLASS_MAP.keys()), key=f"ip_type_{i}")
            st.text_input("Label / ID", value=f"ISO-{i+1:02d}", key=f"ip_label_{i}")
            safe_c, unsafe_c = DEVICE_CLASS_MAP[dt]
            st.markdown(f"**YOLO Classes:**<br><span class='yolo-badge'>✅ {safe_c}</span><br><span class='yolo-badge yolo-badge-unsafe'>⚠️ {unsafe_c}</span>", unsafe_allow_html=True)

    st.markdown("**Compartments:**")
    comp_opts = ["HV Incoming","Transformer","LV Outgoing","Control","Auxiliary","Bus Section","Metering","Protection Relay","Cable Termination","Earth Bar"]
    comp_cols = st.columns(min(num_comp,5))
    for i,col in enumerate(comp_cols):
        with col:
            st.selectbox(f"Comp {i+1}", comp_opts, index=i%len(comp_opts), key=f"comp_name_{i}")
            st.markdown(f"<span class='yolo-badge'>compartment_{i+1}</span>", unsafe_allow_html=True)

    st.markdown("**Live Line Indicators:**")
    ind_cols = st.columns(num_ind)
    for i,col in enumerate(ind_cols):
        with col:
            st.text_input(f"Indicator {i+1} Label", value=f"LLI-{i+1:02d}", key=f"ind_label_{i}")
            st.markdown("<span class='yolo-badge'>indicator_off</span> <span class='yolo-badge yolo-badge-unsafe'>indicator_lit</span> <span class='yolo-badge yolo-badge-unsafe'>indicator_fault</span>", unsafe_allow_html=True)

    # Step 3
    st.markdown('<div class="section-header">Step 3 – If This Then That (IFTTT) Logic Rules</div>', unsafe_allow_html=True)
    st.markdown("Define what the AI model expects when each isolation point is operated. YOLO detection classes are shown automatically from your Step 2 selections.")
    btn_add,btn_clear = st.columns([1,5])
    with btn_add:
        if st.button("➕ Add Rule", key="add_rule"):
            st.session_state.rules.append({"iso":1,"cond":"Isolated","comp":1,"comp_status":"Isolated","ind":1,"ind_status":"Off","risk":"Safe to Access"})
    with btn_clear:
        if st.session_state.rules and st.button("🗑️ Clear All Rules", key="clear_rules"):
            st.session_state.rules = []; st.rerun()

    for i,rule in enumerate(st.session_state.rules):
        st.markdown("---")
        hdr_col,del_col = st.columns([10,1])
        with hdr_col: st.markdown(f"**Rule {i+1}**")
        with del_col:
            if st.button("❌", key=f"del_rule_{i}"):
                st.session_state.rules.pop(i); st.rerun()
        r1,r2,r3,r4 = st.columns(4)
        with r1:
            st.markdown("**🔌 IF** Isolation Point")
            rule["iso"]  = st.selectbox("Point #",list(range(1,num_iso+1)),index=min(rule["iso"]-1,num_iso-1),key=f"iso_{i}")
            rule["cond"] = st.selectbox("State",["Isolated","Not Isolated"],key=f"cond_{i}")
            ip_dt = st.session_state.get(f"ip_type_{rule['iso']-1}","Circuit Breaker")
            sc,uc = DEVICE_CLASS_MAP.get(ip_dt,("circuit_breaker_open","circuit_breaker_closed"))
            exp_c = sc if rule["cond"]=="Isolated" else uc
            st.markdown(f"<small>Expects YOLO class:</small><br><span class='yolo-badge'>🎯 {exp_c}</span>", unsafe_allow_html=True)
        with r2:
            st.markdown("**📦 THEN** Compartment")
            rule["comp"]        = st.selectbox("Comp #",list(range(1,num_comp+1)),index=min(rule["comp"]-1,num_comp-1),key=f"comp_{i}")
            rule["comp_status"] = st.selectbox("Status",["Isolated","Energised"],key=f"comp_status_{i}")
            st.markdown(f"<span class='yolo-badge'>📦 compartment_{rule['comp']}</span>", unsafe_allow_html=True)
        with r3:
            st.markdown("**💡 AND** Indicator")
            rule["ind"]        = st.selectbox("Indicator #",list(range(1,num_ind+1)),index=min(rule["ind"]-1,num_ind-1),key=f"ind_{i}")
            rule["ind_status"] = st.selectbox("State",["Off","Lit"],key=f"ind_status_{i}")
            ind_c = "indicator_off" if rule["ind_status"]=="Off" else "indicator_lit"
            st.markdown(f"<span class='yolo-badge'>💡 {ind_c}</span>", unsafe_allow_html=True)
        with r4:
            st.markdown("**⚠️ Access Risk**")
            rule["risk"] = st.selectbox("Result",["Safe to Access","Unsafe – Do Not Access","Requires Additional Verification"],key=f"risk_{i}")
            if rule["risk"]=="Safe to Access":
                st.markdown('<div class="risk-low">🟢 SAFE</div>', unsafe_allow_html=True)
            elif rule["risk"]=="Unsafe – Do Not Access":
                st.markdown('<div class="risk-high">🔴 UNSAFE</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="risk-medium">🟡 VERIFY</div>', unsafe_allow_html=True)

    # Step 4
    st.markdown('<div class="section-header">Step 4 – Submit for AI Library Approval</div>', unsafe_allow_html=True)
    can_submit = len(st.session_state.rules) > 0
    if not can_submit:
        st.warning("⚠️ Add at least one logic rule (Step 3) before submitting.")
    col_sub,col_trained = st.columns(2)
    with col_sub:
        approved_by = st.text_input("Submitted by (Electrical Engineer or authorised delegate)", key="approved_by")
        if st.button("📤 Submit for Approval to AI Library", key="submit_ai_library", disabled=not can_submit):
            if approved_by.strip():
                ip_map = build_ip_class_map(num_iso)
                summary = " | ".join(f"IP{pt}: {sc}" for pt,(sc,_) in ip_map.items())
                st.success(f"✅ Equipment submitted. AI model training initiated.\n\n**YOLO classes registered:** {summary}")
                st.session_state.training_complete = True
                st.session_state.permit_status = "Draft"
            else:
                st.error("Enter the name of the approving Electrical Engineer before submitting.")
    with col_trained:
        if st.session_state.training_complete:
            st.markdown("**✅ Training Complete — AR Annotation Preview:**")
            safe_image("Substation Trained.jpeg", caption="Trained equipment with AI/AR visual annotations", use_container_width=True)
            st.success(f"Model trained on {num_iso} isolation point(s), {num_comp} compartment(s), {num_ind} indicator(s), {len(st.session_state.rules)} logic rule(s).")

    # Step 5 – AI / AR Verification Engine (V3 new)
    if st.session_state.training_complete:
        st.markdown("---")
        st.markdown('<div class="section-header">Step 5 – AI / AR Verification Engine</div>', unsafe_allow_html=True)
        st.markdown("Verify that the physical isolation matches the trained model rules. **Demo Mode** tests the rule engine against pre-loaded substation images. **Live Mode** uses the tablet camera with the trained ONNX model.")

        verify_mode = st.radio("Verification Mode", ["Demo Mode","Live Mode (camera + ONNX model)"], horizontal=True, key="verify_mode_radio")
        conf_pct    = st.slider("Confidence Threshold", 50, 99, 90, 1, format="%d%%", key="verify_conf",
                                help="Detections below this threshold trigger REVIEW regardless of class.")
        conf_thresh = conf_pct / 100.0
        ip_class_map = build_ip_class_map(st.session_state.get("num_iso", num_iso))
        st.markdown("**YOLO class map for this equipment:**  " +
                    "  ".join(f"<span class='yolo-badge'>IP{pt}: {sc}</span>" for pt,(sc,_) in ip_class_map.items()),
                    unsafe_allow_html=True)
        st.markdown("")

        if verify_mode == "Demo Mode":
            scenario_key = st.selectbox("Select demo scenario", list(DEMO_SCENARIOS.keys()), index=2, key="demo_scenario_select")
            scenario     = DEMO_SCENARIOS[scenario_key]
            demo_dets    = scenario["detections"]
            img_path     = find_image(scenario["file"])
            if img_path:
                try:
                    pil_img   = Image.open(str(img_path)).convert("RGB")
                    annotated = draw_detections_pil(pil_img, demo_dets, conf_thresh)
                    st.image(annotated, caption=scenario["label"], use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not load image: {e}")
            else:
                st.info(f"📷 Demo image not found: **{scenario['file']}** — place demo images in the `demo_images/` folder.")
            with st.expander("Raw detections from this scenario"):
                st.dataframe(pd.DataFrame([{"Class":d["class_name"],"Label":d["label"],"Confidence":f"{d['confidence']:.1%}","Above Threshold":"✅" if d["confidence"]>=conf_thresh else "⚠️ Below"} for d in demo_dets]), use_container_width=True, hide_index=True)
            if st.session_state.rules:
                vd = run_verification_engine(demo_dets, st.session_state.rules, ip_class_map, conf_thresh)
                st.markdown("### Verification Result")
                render_verification_results(vd)
            else:
                st.warning("⚠️ No rules defined. Add rules in Step 3 to run verification.")

        else:  # Live Mode
            model_path = st.text_input("ONNX Model Path", value="./models/moslock_yolov8n.onnx", key="live_model_path")
            if not _ONNX_AVAILABLE:
                st.warning("**onnxruntime not installed** — camera capture available but inference disabled.\nInstall: `pip install onnxruntime numpy`")
            model_exists = Path(model_path).exists()
            if _ONNX_AVAILABLE and not model_exists:
                st.warning(f"Model not found at `{model_path}`.\nTrain first:\n```\ncd yolo_training\npython train.py\npython export.py\n```")
            st.info("📷 Point the tablet camera at the substation — ensure all isolation points and indicators are visible — then press **Take Photo**.")
            cam_img = st.camera_input("Capture substation for AI verification", key="live_camera_verify")
            if cam_img is not None:
                pil_cam = Image.open(cam_img).convert("RGB")
                if _ONNX_AVAILABLE and model_exists:
                    with st.spinner("Running AI detection…"):
                        engine       = load_onnx_model(model_path)
                        live_dets    = run_live_inference(engine, pil_cam, conf_thresh)
                    if live_dets:
                        st.image(draw_detections_pil(pil_cam, live_dets, conf_thresh), caption="Live AI Detection Result", use_container_width=True)
                        with st.expander("Detection details"):
                            st.dataframe(pd.DataFrame([{"Class":d["class_name"],"Confidence":f"{d['confidence']:.1%}","Above Threshold":"✅" if d["confidence"]>=conf_thresh else "⚠️ Below"} for d in live_dets]), use_container_width=True, hide_index=True)
                    else:
                        st.image(pil_cam, caption="Captured image (no detections above threshold)", use_container_width=True)
                        live_dets = []
                        st.warning("No detections above threshold. Try moving closer or lowering the confidence threshold.")
                    if st.session_state.rules:
                        vd = run_verification_engine(live_dets, st.session_state.rules, ip_class_map, conf_thresh)
                        st.markdown("### Live Verification Result")
                        render_verification_results(vd)
                    else:
                        st.warning("⚠️ No rules defined. Add rules in Step 3 to run verification.")
                else:
                    st.image(pil_cam, caption="Captured image (AI inference not available)", use_container_width=True)
                    st.info("📷 Photo captured. Install onnxruntime and provide a trained model to enable AI verification.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – HV ACCESS PERMIT  (V2 unchanged — all 18 parts)
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("⚡ High Voltage Access Permit (STD0930)")
    if not st.session_state.training_complete:
        st.error("🚫 **Cannot issue permit:** Equipment must be trained and approved first. Complete Tab 1 – Training Model.")
        st.stop()

    statuses = ["Draft","Submitted","Issuer Approved","Engineer Approved","Active","Completed"]
    current_idx = statuses.index(st.session_state.permit_status) if st.session_state.permit_status in statuses else 0
    st.markdown("**Permit Workflow:**")
    s_cols = st.columns(len(statuses))
    for i,(scol,s) in enumerate(zip(s_cols,statuses)):
        with scol:
            if i<current_idx: st.markdown(f"✅ ~~{s}~~")
            elif i==current_idx: st.markdown(f"🔵 **{s}**")
            else: st.markdown(f"⭕ {s}")
    pct = int((current_idx/(len(statuses)-1))*100)
    st.progress(pct, text=f"Permit Progress: {pct}%")
    st.caption(f"Parts completed: {len(st.session_state.parts_complete)} / 18")
    st.markdown("---")

    def mark_done(n): st.session_state.parts_complete.add(n)

    with st.expander("Part 1 – Permit Details", expanded=False):
        mark_done(1)
        c1,c2 = st.columns(2)
        with c1:
            st.text_input("Site", value="Bulga Underground Operations", key="permit_site")
            st.text_input("Permit ID Number", value="HV-2026-005", key="permit_id")
            st.date_input("Start Date", value=datetime.date.today(), key="start_date")
            st.date_input("End Date",   value=datetime.date.today(), key="end_date")
        with c2:
            st.text_input("Duration", key="duration")
            st.text_input("HV Permit Issuer", key="issuer")
            st.text_input("HV Permit Holder", key="holder")
            st.text_input("Lock Box Number",  key="lock_box")
        st.text_area("Task Description", placeholder="Describe the work to be performed under this permit…", key="task_desc")
        st.checkbox("Post-work verification — dates and duration confirmed", key="post_work_verify")

    with st.expander("Part 2 – Permitted Work", expanded=False):
        mark_done(2)
        st.caption("Permit Issuer to complete")
        st.text_area("Description of Work", key="perm_work_desc")
        st.text_input("Work Authorisation Reference", key="work_auth_ref")
        st.text_input("Company", key="company")

    with st.expander("Part 3 – Mains and Apparatus to be Accessed", expanded=False):
        mark_done(3)
        c1,c2 = st.columns(2)
        with c1: st.text_input("Location", key="apparatus_location")
        with c2: st.text_input("System Voltage", value="11 kV", key="system_voltage")
        st.multiselect("Equipment to be accessed", ["Mobile Substation (HV) — 3 Isolation Points","Conveyor Drive Motor (LV) — 2 Isolation Points","HV Switchgear Panel B3 — 4 Isolation Points","Feeder Breaker (HV) — 1 Isolation Point"], key="selected_equipment")

    with st.expander("Part 4 – Attachments", expanded=False):
        mark_done(4)
        st.file_uploader("Upload attachments (SLD, JSA, risk assessment, procedures, etc.)", type=["pdf","docx","jpg","png"], accept_multiple_files=True, key="attachments")

    with st.expander("Part 5 – Conditions and Requirements", expanded=False):
        mark_done(5)
        st.caption("Permit Issuer to complete")
        for i,c in enumerate(["Wear appropriate arc-flash PPE at all times within the exclusion zone","Test before touch — always verify dead before accessing conductors","Two-person rule applies — no one works alone on HV equipment","Working earths must be applied before access is permitted","Maintain radio contact with control room throughout task"]):
            st.checkbox(c, value=True, key=f"std_cond_{i}")
        st.text_area("Additional site-specific conditions", key="additional_conds")

    with st.expander("Part 6 – Permit Approval", expanded=False):
        mark_done(6)
        c1,c2 = st.columns(2)
        with c1:
            st.write("**HV Permit Issuer Signature**")
            st.text_input("Name (printed) – Issuer", key="issuer_sig_name")
            if st.button("✍️ Submit for Issuer Approval", key="submit_issuer"):
                st.session_state.permit_status = "Submitted"; st.success("✅ Approval request sent to HV Permit Issuer.")
        with c2:
            st.write("**Electrical Engineer (or delegate) Signature**")
            st.text_input("Name (printed) – Engineer", key="engineer_sig_name")
            if st.button("✍️ Submit for Engineer Approval", key="submit_engineer"):
                st.session_state.permit_status = "Engineer Approved"; st.success("✅ Approval request sent to Electrical Engineer.")

    with st.expander("Part 7 – Pre-Isolation Tasks", expanded=False):
        mark_done(7)
        st.caption("High Voltage Authorised Isolator to complete")
        checks = [st.checkbox(f"{i+1}. {c}", key=f"pre_iso_{i}") for i,c in enumerate(["All required equipment is available for the job","High voltage test equipment checked and calibrated","Intention to isolate communicated to all affected parties","Working earths available and visually inspected","Nearest live point(s) identified and barriers erected","Latest HV Single Line Diagram attached and reviewed","HV Isolation Verifier confirmed available on site"])]
        remaining = checks.count(False)
        if remaining==0: st.success("✅ All pre-isolation tasks complete.")
        else: st.warning(f"⚠️ {remaining} pre-isolation task(s) still to complete.")

    with st.expander("Part 8 – Switching Instructions (12 Step Process)", expanded=False):
        mark_done(8)
        st.warning("⚠️ Fatal Hazard Protocol 7 requires strict adherence to the 12-step isolation process.")
        proc_avail = st.radio("Is an approved documented procedure available?", ["Yes – attach copy and use in place of these instructions","No – complete switching instructions below"], index=1, key="proc_radio")
        if "No" in proc_avail:
            st.write("**Complete the 12 step switching instructions:**")
            st.dataframe(pd.DataFrame({"Step":list(range(1,13)),"Apparatus":[""]*12,"Action":[""]*12,"Permit Lock No.":[""]*12,"Time":[""]*12,"HV Isolator Initials":[""]*12,"HV Verifier Initials":[""]*12}), use_container_width=True)
            st.caption("Table is editable in the full application.")

    with st.expander("Isolation Verification – Photo Capture", expanded=False):
        mark_done(8)
        st.markdown('<div class="risk-high">🚨 Mandatory: Capture or upload a verification photo before proceeding to Part 9.</div>', unsafe_allow_html=True)
        st.markdown("")
        cam_col,up_col = st.columns(2)
        with cam_col:
            st.write("**📷 Tablet / Mobile Camera**")
            picture = st.camera_input("Take photo of isolation point", key="isolation_camera")
            if picture: st.success("✅ Photo captured."); st.image(picture, use_container_width=True)
        with up_col:
            st.write("**📁 Upload from Device**")
            uploaded = st.file_uploader("Upload isolation photo", type=["jpg","png","jpeg"], key="isolation_uploader")
            if uploaded: st.success("✅ Photo uploaded."); st.image(uploaded, use_container_width=True)

    with st.expander("Part 9 – HV Isolation Verification (AI/AR)", expanded=False):
        mark_done(9)
        st.markdown('<div class="section-header">AI / AR Verification Engine</div>', unsafe_allow_html=True)
        col_lock_img,col_iso_confirm = st.columns(2)
        with col_lock_img:
            st.write("**🔒 Verification Photo – Lock Attached**")
            safe_image("Sub Locked Out.jpg", caption="Verification Photo – Lock Attached", use_container_width=True)
            st.markdown('<div class="risk-low">✅ <strong>AR Recognition Result</strong><br>Lock detected at <strong>Isolation Point 3</strong><br>Lock Serial: <strong>PL-00342</strong><br>Confidence: <strong>98.2%</strong></div>', unsafe_allow_html=True)
        with col_iso_confirm:
            st.write("**⚡ Confirmed Isolation Status**")
            safe_image("Substation Isolated LV.jpeg", caption="Confirmed Isolation – Compartments Verified by AR", use_container_width=True)
            st.markdown("**Compartment Status:**")
            st.table(pd.DataFrame({"Compartment":["Comp 1 – HV Incoming","Comp 2 – LV Outgoing","Comp 3 – Control"],"Status":["🔴 Isolated","🔴 Isolated","🟢 Energised"],"AR Verified":["✅ Yes","✅ Yes","✅ Yes"]}))
        st.write("**📐 Single Line Diagram / Isolation Drawing**")
        p = find_image("Isolation drawing.jpg")
        if p:
            try:
                drawing = Image.open(str(p)); drawing = drawing.resize((int(drawing.width*0.5),int(drawing.height*0.5)))
                st.image(drawing, caption="Isolation Drawing — AR Overlay Check")
            except Exception:
                st.info("📐 Isolation Drawing placeholder")
        else:
            st.info("📐 Isolation Drawing / SLD placeholder (file not found in demo)")
        c1,c2 = st.columns(2)
        with c1:
            st.write("**HV Authorised Isolator**")
            st.text_input("Name (printed) – Isolator", key="isolator_sig_name")
            st.text_input("Date / Time / Contact",      key="isolator_sig_dt")
        with c2:
            st.write("**Isolation Verifier**")
            st.text_input("Name (printed) – Verifier",  key="verifier_sig_name")
            st.text_input("Date / Time / Contact",      key="verifier_sig_dt")

    with st.expander("Part 10 – Permit Activation", expanded=False):
        mark_done(10)
        st.caption("HV Permit Holder verifies isolation and attaches Permit Holder's lock")
        st.text_input("Name (printed) – Permit Holder", key="activation_name")
        if st.checkbox("I have personally verified the isolation is complete and I am attaching my personal lock to the lock box", key="activation_ack"):
            st.session_state.permit_status = "Active"
            if st.session_state.lock_count==0: st.session_state.lock_count=1
            st.success("✅ Permit Activated — Equipment is isolated and locked out. Work may now commence.")

    with st.expander("Part 11 – Sign-on / Sign-off", expanded=False):
        mark_done(11)
        st.caption("HV Permit Holder and Workers Sign-on / Sign-off / Handover")
        new_so = st.text_input("Worker Name to Sign On", key="new_signon")
        if st.button("✅ Sign On", key="part11_sign_on"):
            if new_so and new_so not in st.session_state.signed_on_workers:
                st.session_state.signed_on_workers.append(new_so)
                st.session_state.lock_count += 1
                st.success(f"'{new_so}' signed on. Total locks on box: {st.session_state.lock_count}.")
        if st.session_state.signed_on_workers:
            st.markdown("**Currently Signed On:**")
            for w in st.session_state.signed_on_workers: st.write(f"✅ {w}")
        st.caption("Full sign-on / sign-off roster table available in production app.")

    with st.expander("Part 12 – Working Earth Locations", expanded=False):
        mark_done(12)
        st.markdown('<div class="risk-high">🚨 All working earths MUST be recorded before work commences.</div>', unsafe_allow_html=True)
        st.markdown("")
        st.dataframe(pd.DataFrame({"Location and Details":[""]*4,"Placed by (printed name)":[""]*4,"PH Initials (Place)":[""]*4,"Removed by (printed name)":[""]*4,"PH Initials (Remove)":[""]*4}), use_container_width=True)

    with st.expander("Part 13 – Testing (Where Applicable)", expanded=False):
        mark_done(13)
        if st.radio("Testing Required?",["Required","Not Required"],key="testing_radio")=="Required":
            for i,item in enumerate(["All personnel instructed to sign-off and remove personal locks","HV Authorised Isolator instructed to remove working earths","Permit Holder has witnessed and countersigned isolator's removal"]):
                st.checkbox(item, key=f"test_{i}")

    with st.expander("Part 14 – Task Monitoring and Inspection", expanded=False):
        mark_done(14)
        st.caption("Supervisor, safety representatives, and other monitoring personnel")
        st.text_area("Inspection Notes / Observations", key="inspection_notes")

    with st.expander("Part 15 – Permit Cancellation Pre-Restoration of Power", expanded=False):
        mark_done(15)
        st.markdown('<div class="risk-high">⛔ ALL checks below must be complete before requesting restoration of power.</div>', unsafe_allow_html=True)
        st.markdown("")
        all_done = all(st.checkbox(f"{i+1}. {c}", key=f"pre_restore_{i}") for i,c in enumerate(["Visual examination of work area completed","All tests completed and documented","All earth bonds in place and secure","All disconnected cables capped / insulated","All identification labels and warning signs updated","All barrier tape removed","All personal locks removed and all personnel signed off","Work crews confirm equipment is safe to operate"]))
        if all_done: st.success("✅ All pre-restoration checks complete. Power restoration may be requested.")
        st.text_input("Person notified of intention to restore power", key="notified_person")
        st.checkbox("I confirm all work is complete (or cancelled) and equipment is safe to re-energise", key="pre_restore_confirm")

    with st.expander("Part 16 – Restoration of Power", expanded=False):
        mark_done(16)
        st.warning("⚠️ All energy source restorations must include the 12-step process (Glencore Fatal Hazard Protocol 7).")
        if "No" in st.radio("Is an approved documented procedure available?",["Yes – attach copy","No – complete below"],index=1,key="proc_restore_radio"):
            st.dataframe(pd.DataFrame({"Step":list(range(1,13)),"Apparatus":[""]*12,"Action":[""]*12,"Permit Lock No.":[""]*12,"Time":[""]*12,"HV Isolator Initials":[""]*12,"HV Verifier Initials":[""]*12}), use_container_width=True)

    with st.expander("Part 17 – Permit Completion (HV Permit Holder)", expanded=False):
        mark_done(17)
        st.warning("⚠️ Only return plant or equipment to service when this permit is complete or cancelled and all signatures are obtained.")
        st.checkbox("Permit activities COMPLETE", key="part17_complete")
        st.checkbox("Permit activities INCOMPLETE (state reason in comments)", key="part17_incomplete")
        st.text_area("Comments / Reason for Cancellation", key="part17_comments")
        if st.session_state.get("part17_complete"):
            if st.button("🔴 Mark Permit as Complete", key="cancel_permit"):
                st.session_state.permit_status = "Completed"
                st.success("Permit marked COMPLETED. Equipment may be returned to service.")

    with st.expander("Part 18 – Permit Review", expanded=False):
        mark_done(18)
        st.caption("To be completed by an authorised Electrical Engineer or delegate")
        st.text_area("Review Comments and Follow-up Actions", key="review_comments")
        st.text_input("Reviewer Name (printed)", key="reviewer_name")
        st.checkbox("This permit has been reviewed and complies with all applicable standards", key="review_confirm")

    st.markdown("---")
    btn1,btn2,btn3 = st.columns(3)
    with btn1:
        if st.button("💾 Save Draft", key="save_draft"): st.info("Draft saved. (Full app: encrypted cloud sync.)")
    with btn2:
        if st.button("🖨️ Generate PDF Preview", key="gen_pdf"): st.info("PDF generation available in full application.")
    with btn3:
        if st.button("✅ Finalise & Submit Permit", key="finish_permit", type="primary"):
            st.success("✅ HV isolation permit finalised and submitted to records.")
            st.session_state.permit_status = "Completed"

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – TVL OBSERVATION  (V2 unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("👁️ TVL – Targeted Visible Leadership Observation (FRM1277)")
    tvl_all_keys = ["tvl_sup_1","tvl_sup_2","tvl_lock_1","tvl_lock_2","tvl_sign_1","tvl_sign_2","tvl_sign_3","tvl_sign_4","tvl_new_1","tvl_new_2","tvl_new_3","tvl_new_4","tvl_access_1","tvl_access_2","tvl_access_3","tvl_maint_1","tvl_maint_2","tvl_train_1","tvl_arc_1","tvl_arc_2","tvl_test_1"]
    checked = sum(1 for k in tvl_all_keys if st.session_state.get(k,False))
    total   = len(tvl_all_keys)
    score   = int((checked/total)*100)
    score_col,bar_col = st.columns([1,3])
    with score_col:
        st.metric("Compliance Score", f"{score}%", delta=f"{score-87}% vs last TVL")
        if score>=80: st.markdown('<div class="risk-low">🟢 COMPLIANT</div>', unsafe_allow_html=True)
        elif score>=60: st.markdown('<div class="risk-medium">🟡 MONITOR</div>', unsafe_allow_html=True)
        else: st.markdown('<div class="risk-high">🔴 NON-COMPLIANT — escalate immediately</div>', unsafe_allow_html=True)
    with bar_col:
        st.progress(score, text=f"{checked} / {total} items checked — score updates live")
    st.markdown("---")
    dc1,dc2 = st.columns(2)
    with dc1:
        st.markdown("**Observation Details**")
        st.date_input("Date", value=datetime.date.today(), key="tvl_date")
        st.time_input("Time", key="tvl_time")
        st.text_input("Location", key="tvl_location")
        st.selectbox("Shift", ["Day (06:00–18:00)","Night (18:00–06:00)"], key="tvl_shift")
        st.text_input("Observation Team Leader", key="tvl_leader")
    with dc2:
        st.markdown("**Work Context**")
        st.text_input("Description of Work Being Observed", key="tvl_description")
        linked_job = st.selectbox("Linked Permit / Job", ["HV-2026-001 (Mobile Substation)","HV-2026-002 (Conveyor Drive)","HV-2026-004 (Feeder Breaker)"], key="linked_job")
        if linked_job:
            safe_image("Sub Locked Out.jpg", caption=f"Isolation lock photo from permit: {linked_job}", use_container_width=True)
            st.success("✅ Linked isolation lock photo pulled from permit record.")
    st.markdown("---")

    def tvl_section(title, items, key_prefix):
        st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
        results = [st.checkbox(item, key=f"{key_prefix}_{i+1}") for i,item in enumerate(items)]
        st.caption(f"Section: {sum(results)}/{len(items)} ✅")
        return results

    tvl_section("Electrical Supervisor Credentials",["Check of supervisor appointment documentation","Check for presence of No Plan No Work booklet"],"tvl_sup")
    tvl_section("Isolation & Personal Locks",["Audit work area to ensure compliance with isolation procedure","Challenge test a worker — verify they carry personal locks on site"],"tvl_lock")
    tvl_section("Electrical Area Signage",["Signs are relevant and easily understandable","Signage covers all entry points with correct qualification details","Clearly defines all qualification requirements","Legible and compliant with Australian Standards"],"tvl_sign")
    tvl_section("New Starters – Electrical Workers",["Verify Electrical Licence currency","Verify EEHA currency","Verify LVR/CPR competency","Verify EEHA Challenge Test completion"],"tvl_new")

    st.markdown('<div class="section-header">Access to Exposed Conductors</div>', unsafe_allow_html=True)
    for label,key,preview_label in [("High Voltage Access Permits","tvl_access_1","High Voltage Access Permit"),("Group Isolation Permits","tvl_access_2","Group Isolation Permit"),("System Impairment Permits – Electrical","tvl_access_3","System Impairment Permit")]:
        ac1,ac2 = st.columns([4,1])
        with ac1: st.checkbox(label, key=key)
        with ac2:
            if st.button("👁️ View Live", key=f"view_{key}"): st.session_state.show_permit_preview = preview_label

    if st.session_state.show_permit_preview:
        with st.expander(f"🔍 Live Read-Only Permit: {st.session_state.show_permit_preview}", expanded=True):
            st.info(f"📋 Read-only view of **{st.session_state.show_permit_preview}** | Permit: HV-2026-001")
            lc1,lc2 = st.columns(2)
            with lc1:
                st.text_input("Permit ID",     value="HV-2026-001",      disabled=True)
                st.text_input("Permit Holder", value="J. Smith",          disabled=True)
                st.text_input("Status",        value="Active",            disabled=True)
            with lc2:
                st.text_input("Equipment",     value="Mobile Substation", disabled=True)
                st.text_input("Issued By",     value="R. Jones",          disabled=True)
                st.text_input("Expires",       value="Today 18:00",       disabled=True)
            st.text_area("Task Description", value="Isolation of mobile substation for scheduled maintenance", disabled=True)
            st.success("✅ Permit is active and being followed correctly.")
            if st.button("✖️ Close Permit View", key="close_preview"):
                st.session_state.show_permit_preview = None; st.rerun()

    tvl_section("Maintenance & Compliance of Electrical Equipment",["Inspections completed when due and records sighted","Locking mechanisms maintained and functional"],"tvl_maint")
    tvl_section("Training",["All coal mine workers have been trained in the isolation procedure"],"tvl_train")
    tvl_section("Arc Flash Protection",["Wearing appropriate arc-flash clothing (verified by observation)","Carrying Trolex Non-Contact Voltage Tester (verified)"],"tvl_arc")
    tvl_section("Testing and Tagging",["Regular insulation and continuity tests completed and audited"],"tvl_test")

    st.markdown("---")
    obs1,obs2,obs3 = st.columns(3)
    with obs1: st.text_area("✅ What was done well?",           key="tvl_well")
    with obs2: st.text_area("🔧 Opportunities for Improvement", key="tvl_improve")
    with obs3: st.text_area("💬 Other Comments",                key="tvl_comments")
    if st.button("📤 Submit TVL Observation", key="submit_tvl", type="primary"):
        st.success(f"✅ TVL Observation submitted. Compliance score: **{score}%**. Linked to permit: {linked_job}.")
        if score<80: st.warning("⚠️ Compliance score below 80% — automatic notification sent to Electrical Engineer and Site Manager for follow-up.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – 12 STEP REFERENCE  (V2 unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📋 12 Step Isolation Reference")
    st.caption("Always-available reference for Permit Holders. Derived from Glencore Fatal Hazard Protocol 7 / AS/NZS 4836.")
    ref_img_col,steps_col = st.columns([1,1])
    with ref_img_col:
        p12 = find_image("12stepiso.png")
        if p12:
            try:
                img = Image.open(str(p12)); img = img.resize((int(img.width*0.5),int(img.height*0.5)))
                st.image(img, caption="12 Step Isolation Process Reference Card", use_container_width=True)
            except Exception:
                st.info("📋 Place '12stepiso.png' in the app directory.")
        else:
            st.info("📋 12 Step process diagram — place '12stepiso.png' in the app directory.")
    with steps_col:
        st.markdown("### The 12 Steps")
        for step,detail in [
            ("1. Identify Energy Sources","List ALL energy sources: electrical, hydraulic, pneumatic, gravitational, stored mechanical. Do not begin until all are identified."),
            ("2. Advise Relevant Parties","Notify all affected personnel, supervisors, and control room of the planned isolation. Record who was notified."),
            ("3. Stop Equipment Safely","Operate equipment through a normal stop cycle where possible. Do not isolate under load unless necessary."),
            ("4. Isolate Energy Sources","Operate all identified isolation devices to the OFF / OPEN / ISOLATED position."),
            ("5. Lock and Tag","Apply your personal lock AND a danger tag to each isolation device. No one else may remove your lock."),
            ("6. Verify Isolation","Use approved test equipment to confirm zero energy state at all isolation points. Test the tester first and last."),
            ("7. Commence Work","Work may ONLY commence after Steps 1–6 are complete and verified by the Isolation Verifier."),
            ("8. Complete Work","Ensure all tools, materials, and personnel are clear of the equipment before proceeding."),
            ("9. Check Work Area","Conduct a final visual inspection. Confirm no tools or materials are left inside equipment."),
            ("10. Clear Area","Ensure all workers are accounted for and physically clear of the equipment."),
            ("11. Remove Locks and Tags","Each worker removes ONLY their own personal lock. Nobody removes another person's lock — ever."),
            ("12. Restore Energy","Re-energise in the reverse order of isolation following approved switching instructions. Record all switching actions."),
        ]:
            with st.expander(f"**{step}**"): st.write(detail)
    st.markdown("---")
    st.markdown('<div class="risk-high">🚨 <strong>CRITICAL REMINDERS:</strong> Never remove another person\'s lock. Never work on equipment without your own lock applied. Treat ALL equipment as live until you have personally tested it and proven it dead. When in doubt — STOP and seek guidance.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 – EQUIPMENT LIBRARY  (V2 unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🗄️ Equipment Library")
    st.caption("Catalogued plant with AI/AR verification data, isolation history, and permit linkage.")
    sc1,sc2,sc3,sc4 = st.columns([3,1,1,1])
    with sc1: search_term = st.text_input("🔍 Search", placeholder="Name, type, location, serial…", key="eq_search")
    with sc2: ar_filter   = st.selectbox("AR Status",["All","Ready","Pending","Training Required"],key="ar_filter")
    with sc3: type_filter = st.selectbox("Voltage Class",["All","HV","LV"],key="type_filter")
    with sc4:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➕ Add New Equipment", key="add_eq"): st.info("Opens Training Model tab with new equipment form.")
    equipment_data = [
        {"name":"Mobile Substation",     "type":"HV","voltage":"11 kV","iso_points":3,"compartments":3,"earth_points":2,"last_trained":"2026-03-15","ar_status":"Ready","location":"Zone B – Underground","permits_issued":5},
        {"name":"Conveyor Drive Motor",  "type":"LV","voltage":"415 V","iso_points":2,"compartments":2,"earth_points":1,"last_trained":"2026-03-20","ar_status":"Ready","location":"Conveyor 3","permits_issued":3},
        {"name":"HV Switchgear Panel B3","type":"HV","voltage":"11 kV","iso_points":4,"compartments":4,"earth_points":3,"last_trained":"2026-03-25","ar_status":"Ready","location":"Substation B","permits_issued":7},
        {"name":"Feeder Breaker",        "type":"HV","voltage":"3.3 kV","iso_points":1,"compartments":2,"earth_points":1,"last_trained":"2026-03-10","ar_status":"Pending","location":"Zone A","permits_issued":2},
    ]
    filtered_eq = [e for e in equipment_data if (not search_term or search_term.lower() in str(e).lower()) and (ar_filter=="All" or e["ar_status"]==ar_filter) and (type_filter=="All" or e["type"]==type_filter)]
    st.markdown(f"**{len(filtered_eq)} of {len(equipment_data)} items shown**")
    st.markdown("---")
    for eq in filtered_eq:
        icon = "⚡" if eq["type"]=="HV" else "🔌"
        ar_icon = "✅" if eq["ar_status"]=="Ready" else "⏳"
        eqc1,eqc2,eqc3,eqc4 = st.columns([1,3,2,2])
        with eqc1: st.markdown(f"<div style='font-size:40px;text-align:center;'>{icon}</div><div style='text-align:center;'>{ar_icon} {eq['ar_status']}</div>", unsafe_allow_html=True)
        with eqc2:
            st.markdown(f"**{eq['name']}**")
            st.write(f"🏭 {eq['location']}  |  ⚡ {eq['voltage']}")
            st.write(f"🔌 {eq['iso_points']} Isolation Points  |  📦 {eq['compartments']} Compartments  |  🌍 {eq['earth_points']} Earth Points")
            st.caption(f"Last trained: {eq['last_trained']}")
        with eqc3: st.metric("Permits Issued", eq["permits_issued"])
        with eqc4:
            st.button("👁️ View Details", key=f"view_eq_{eq['name']}")
            st.button("⚡ Start Permit", key=f"start_permit_{eq['name']}")
            st.button("🔄 Re-train",     key=f"retrain_{eq['name']}")
        st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 – PERMIT HISTORY  (V2 unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("📁 Permit History")
    st.caption("Searchable, auditable record of all HV permits. Immutable once closed.")
    hc1,hc2,hc3 = st.columns([3,1,1])
    with hc1: h_search = st.text_input("🔍 Search Permits", placeholder="Permit ID, equipment, person, date…", key="permit_search")
    with hc2: h_status = st.selectbox("Status",["All","Active","Approved","Completed","Cancelled"],key="hist_status_filter")
    with hc3: h_date   = st.selectbox("Date Range",["All Time","Today","This Week","This Month"],key="hist_date_filter")
    permit_history = [
        {"id":"HV-2026-001","date":"2026-03-15","equipment":"Mobile Substation","holder":"J. Smith", "issuer":"R. Jones","status":"Active",    "parts":18,"tvl":"Linked"},
        {"id":"HV-2026-002","date":"2026-03-20","equipment":"Conveyor Drive",    "holder":"M. Brown","issuer":"R. Jones","status":"Completed","parts":18,"tvl":"Linked"},
        {"id":"HV-2026-003","date":"2026-03-25","equipment":"HV Switchgear Panel","holder":"T. Davis","issuer":"R. Jones","status":"Completed","parts":18,"tvl":"Not Linked"},
        {"id":"HV-2026-004","date":"2026-04-01","equipment":"Feeder Breaker",    "holder":"J. Smith", "issuer":"R. Jones","status":"Approved", "parts":6, "tvl":"Pending"},
    ]
    filtered_hist = [p for p in permit_history if (not h_search or h_search.lower() in str(p).lower()) and (h_status=="All" or p["status"]==h_status)]
    hm1,hm2,hm3,hm4 = st.columns(4)
    with hm1: st.metric("Total Permits", len(permit_history))
    with hm2: st.metric("Active",        sum(1 for p in permit_history if p["status"]=="Active"))
    with hm3: st.metric("Completed",     sum(1 for p in permit_history if p["status"]=="Completed"))
    with hm4: st.metric("TVL Linked",    sum(1 for p in permit_history if p["tvl"]=="Linked"))
    st.markdown(f"**Showing {len(filtered_hist)} of {len(permit_history)} permit(s)**")
    st.markdown("---")
    status_icons = {"Active":"🟢","Completed":"⚫","Approved":"🔵","Cancelled":"🔴","Pending":"🟡"}
    for p in filtered_hist:
        dot = status_icons.get(p["status"],"⚪")
        with st.expander(f"{dot} **{p['id']}**  |  {p['equipment']}  |  Holder: {p['holder']}  |  {p['date']}  |  {p['status']}"):
            hpc1,hpc2,hpc3 = st.columns(3)
            with hpc1:
                st.write(f"**Permit ID:** {p['id']}")
                st.write(f"**Date Issued:** {p['date']}")
                st.write(f"**Equipment:** {p['equipment']}")
            with hpc2:
                st.write(f"**Permit Holder:** {p['holder']}")
                st.write(f"**Issuer:** {p['issuer']}")
                st.write(f"**Parts Complete:** {p['parts']}/18")
            with hpc3:
                st.write(f"**Status:** {dot} {p['status']}")
                st.write(f"**TVL Observation:** {p['tvl']}")
                hb1,hb2,hb3 = st.columns(3)
                with hb1: st.button("👁️ View",  key=f"hist_view_{p['id']}")
                with hb2: st.button("🖨️ PDF",   key=f"hist_pdf_{p['id']}")
                with hb3: st.button("📊 Audit", key=f"hist_audit_{p['id']}")

# ─── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("---")
fc1,fc2,fc3 = st.columns(3)
with fc1: st.caption("**MOSLock V3** – Mapped Out Solutions® | mos | Lock®")
with fc2: st.caption("Standards: STD0930 | FRM1277 | Glencore FHP-07 | AS/NZS 4836")
with fc3: st.caption("Underground Coal Operations – *We will always have a plan.*")
