# MOSLock PRD Delta — V1.1
## Changes from V1.0 | Mapped Out Solutions® | mos | Lock®

**Document Date:** December 2025  
**Version:** V1.1  
**Status:** Delivered — Trial Ready  

---

## 1. What Changed from V1.0

### 1.1 Branding (NEW in V1.1)

| Item | V1.0 | V1.1 |
|------|------|------|
| Visual identity | Generic Streamlit default | Full MOS Brand Guidelines V2 applied |
| Primary colour | Streamlit red | MOS Cyan `#3ECFCF` |
| Sidebar | Default white | Dark navy `#071539` per brand guide |
| Logo | Text only | MOS M-Disc + wordmark (SVG, base64-embedded, portable) |
| Typography | System default | Helvetica Neue / Inter, brand-specified weights |
| Buttons | Default | Min 48px touch target, cyan fill, bold, tablet-optimised |
| Tab strip | Default | Dark navy strip, cyan active tab |
| Footer | None | Branded footer with company, version, standards references |
| CSS architecture | Inline Streamlit config | Centralised `moslock_branding/theme.css` with CSS custom properties |
| Brand config | None | `moslock_branding/brand_config.py` — single source of truth |
| High contrast mode | None | Toggle in sidebar for underground/low-light conditions |

### 1.2 Detection Classes (EXPANDED in V1.1)

| Item | V1.0 | V1.1 |
|------|------|------|
| Total classes | 10 | **25** |
| Lock types | 1 | 2 (personal_lock, permit_lock) |
| Isolation point types | 3 types × 2 states | **5 types × 2 states = 10** |
| Compartment zones | 3 | **10** (full BST008 coverage) |
| Live line indicators | 1 | **3 states** (lit, off, fault) |

New isolation point types added:
- `fuse_withdrawn` / `fuse_inserted` (fuse isolation)
- `earth_switch_applied` / `earth_switch_removed` (working earth)
- `disconnect_open` / `disconnect_closed` (disconnector)

New compartments added:
- compartment_4: Control
- compartment_5: Auxiliary
- compartment_6: Bus Section
- compartment_7: Metering
- compartment_8: Protection Relay
- compartment_9: Cable Termination
- compartment_10: Earth Bar

### 1.3 App Architecture (MAJOR CHANGE in V1.1)

| Item | V1.0 | V1.1 |
|------|------|------|
| Deployment | Local only | Local + Streamlit Community Cloud |
| Graceful degradation | None | 3 scenarios with full graceful fallback |
| Demo Mode | Static screenshots | Pre-loaded detections with bbox overlays on real photos |
| Live Mode | Camera capture only | Camera + file upload + ONNX inference |
| IFTTT engine | Basic | Full rule evaluation: PASS/FAIL/REVIEW/STOP per rule |
| Verdict system | Binary pass/fail | 4-state verdict: ALL CLEAR / DISCREPANCY / REVIEW / STOP |
| Confidence threshold | Fixed 80% | Configurable slider 50–99% (default 90%) |
| Tab structure | 3 tabs | **6 tabs**: Detection, Verification, Compartments, Training Model, Permit, About |
| Permit register | None | Worker sign-on/sign-off with timestamps |
| 12-step process | Reference only | Visual progress indicator linked to permit state |
| Compartment grid | None | 10-cell status grid: isolated/energised/unknown |
| Lock count | Detected only | Detected vs. workers signed on (balance check) |
| indicator_fault handling | Not implemented | **STOP alert with pulse animation** — mandatory manual check |

### 1.4 Tablet Optimisation (NEW in V1.1)

| Item | V1.0 | V1.1 |
|------|------|------|
| Target device | Generic | Samsung Galaxy Tab S10 (10.9-inch, 2560×1600) |
| Min touch target | Not specified | 48px enforced via CSS |
| Body font size | Default (14px) | 16px enforced |
| Header size | Default | 22–28px |
| Button size | Default | 17px font, 48px min-height |
| Sidebar default | Open | Collapsed (saves screen space on tablet) |
| Tables | Fixed | Horizontally scrollable |
| Hover interactions | Present | Removed (touch-first design) |
| Camera input | Not primary | PRIMARY input for Live Mode |
| Offline support | Partial | Full: Google Fonts fallback, no external dependencies |

### 1.5 YOLO Training Pipeline (NEW in V1.1)

| Item | V1.0 | V1.1 |
|------|------|------|
| Model | None | YOLOv8n — fine-tunable, ONNX-exportable |
| Training data | None | 750 synthetic images (30 × 25 classes) |
| Synthetic generator | None | `generate_synthetic_samples.py` — shapes, labels, augmentation |
| Augmentation | None | Brightness ±55%, rotation ±15°, scale ±40%, noise, blur |
| Bootstrap annotation | None | `bootstrap_annotations.py` — Claude Vision pre-annotation |
| Export | None | `export.py` — ONNX (primary) + TFLite (optional) |
| Inference module | None | `inference.py` — standalone ONNX runner, NMS, bbox post-processing |
| Model metadata | None | `moslock_yolov8n_metadata.json` |
| Training README | None | Full workflow from synthetic → real data → trained model → tablet |

---

## 2. What Is Delivered in V1.1

### Delivered — Production Ready

- ✅ `moslock_v3.py` — full Streamlit app, 3 deployment scenarios
- ✅ `moslock_branding/brand_config.py` — Python brand constants
- ✅ `moslock_branding/theme.css` — complete CSS stylesheet
- ✅ `moslock_branding/README.md` — brand customisation guide
- ✅ `yolo_training/data.yaml` — 25-class dataset config
- ✅ `yolo_training/train.py` — YOLOv8n fine-tuning script
- ✅ `yolo_training/export.py` — ONNX + TFLite export
- ✅ `yolo_training/inference.py` — ONNX inference module (used by app)
- ✅ `yolo_training/generate_synthetic_samples.py` — 750-image synthetic dataset generator
- ✅ `yolo_training/bootstrap_annotations.py` — Claude Vision pre-annotation tool
- ✅ `yolo_training/requirements.txt` — pinned dependencies
- ✅ `yolo_training/README.md` — complete training workflow guide
- ✅ `DEPLOYMENT_TAB_S10.md` — Tab S10 deployment guide (3 options)
- ✅ `DEMO_INSTRUCTIONS_V3.md` — trial site walk-through + FAQ
- ✅ `PRD_DELTA_V1.1.md` — this document

### Delivered — Requires Real Site Data to Reach Production Accuracy

- ⚙ Trained ONNX model — pipeline is complete; accuracy improves with real annotated photos
- ⚙ Per-site Training Model rules — default rules provided; site EE configures for specific equipment

---

## 3. Not In V1.1 — Future Roadmap

### V3.1 — Data Persistence & Audit Trail

| Feature | Description | Priority |
|---------|-------------|----------|
| Local SQLite database | Persist permit registers, scan history, detections between sessions | High |
| PDF audit report export | GCAA-format verification record exportable as PDF | High |
| Scan history viewer | Timeline of all verifications performed with thumbnails | Medium |
| Session restore | Resume permit after tablet sleep/battery event | Medium |

### V3.2 — Security & Access Control

| Feature | Description | Priority |
|---------|-------------|----------|
| Authorised engineer sign-in | PIN/biometric lock on Training Model changes | High |
| Rule version control | Hash-signed rule set; changes logged with name/time | High |
| Permit digital signature | Worker sign-on with biometric or PIN confirmation | Medium |
| Audit log tamper protection | Append-only log with checksums | Medium |

### V3.3 — Integration

| Feature | Description | Priority |
|---------|-------------|----------|
| CMMS integration | Push verified isolation records to SAP/Maximo/etc. | High |
| GCAA permit system integration | Sync with existing HV permit management system | High |
| Email/SMS notification | Alert permit issuer when verification complete | Medium |
| Multi-substation mode | Handle multiple substations on one permit | Medium |

### V3.4 — Model & Detection Improvements

| Feature | Description | Priority |
|---------|-------------|----------|
| Real annotated dataset | ≥200 BST008 photos, reviewed via CVAT | Critical for production |
| INT8 quantisation | Reduce model size to ~2 MB, increase inference speed | Medium |
| Thermal image support | Detect hot spots / heat signatures on compartment doors | Low |
| Lock serial number OCR | Read permit lock numbers for cross-check with permit record | Medium |
| Spatial reasoning | Confirm isolation point is on the correct compartment via bbox overlap | High |

### V4.0 — Platform Expansion

| Feature | Description | Priority |
|---------|-------------|----------|
| Android native app | React Native / Flutter wrapping for app store distribution | Low |
| AR overlay | Overlay detections on camera feed using WebXR | Low |
| Multiple substation types | Models for Ampcontrol, Becker, Joy, etc. | High |

---

## 4. Known Limitations in V1.1

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| Trained on synthetic data only | Lower detection accuracy on real equipment | bootstrap_annotations.py + real photo training before site trial |
| No spatial reasoning (bbox overlap for compartment mapping) | Compartment state is inferred, not spatially verified | V3.4 roadmap; current rule engine is adequate for demo |
| Session state lost on battery death | Permit register not persisted | V3.1 SQLite persistence; advise to stay on charge underground |
| indicator_fault requires real equipment | Fault state hard to demo without a covered indicator | Use demo mode to demonstrate the STOP alert |
| Confidence threshold applies uniformly | indicator_fault might be under-detected at high threshold | Roadmap: per-class threshold override |
| No authentication on Training Model | Rules could be changed by non-engineer | V3.2 access control; acceptable for V3 demo |

---

## 5. Acceptance Criteria for Trial Site

The following criteria define a successful V3.0 trial demo:

- [ ] App launches on Tab S10 within 5 seconds
- [ ] Demo Mode loads all 4 scenarios with annotated images
- [ ] IFTTT engine returns correct verdict for each demo scenario
- [ ] Verification tab shows per-rule PASS/FAIL/REVIEW correctly
- [ ] Compartment grid shows correct states
- [ ] Worker sign-on register accepts entries and persists during session
- [ ] Confidence threshold slider changes verdict in real time
- [ ] High Contrast Mode toggles and improves legibility visibly
- [ ] All 6 tabs navigate without errors on the tablet
- [ ] App runs for 2 hours without crash
- [ ] Streamlit Community Cloud deployment accessible from stakeholder laptops
- [ ] If Live Mode with ONNX model: detections appear within 3 seconds of capture

---

## 6. Dependency Versions

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | ≥1.40.0 | Web framework |
| pillow | ≥11.0.0 | Image processing, bbox drawing |
| pandas | ≥2.0.0 | Data tables |
| onnxruntime | ≥1.20.0 | ONNX inference (Live Mode) |
| numpy | ≥1.26.0 | Array ops for inference |
| ultralytics | 8.3.56 | YOLO training (training only, not app) |
| anthropic | 0.40.0 | Claude Vision for bootstrap annotation |

---

*Mapped Out Solutions® | mos | Lock® V3.0 | PRD Delta V1.1*  
*"We will always have a plan."*
