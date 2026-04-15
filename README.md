# MOSLock V3
## Mapped Out Solutions® | mos | Lock®
**"We will always have a plan."**

AI-assisted high-voltage isolation verification for mobile substations.  
Built for Samsung Galaxy Tab S10 + Streamlit Community Cloud.

---

## Quick Start

### Remote Demo (desktop/laptop — 2 commands)
```bash
pip install streamlit pillow pandas
streamlit run moslock_v3.py
```

### Tablet Demo (Samsung Galaxy Tab S10 — Demo Mode)
```bash
# In Termux on the tablet:
pip install streamlit pillow pandas
streamlit run moslock_v3.py
# Open Chrome → http://localhost:8501
```

### Tablet Live (with trained ONNX model)
```bash
pip install streamlit pillow pandas onnxruntime numpy
# Place trained model at ./models/moslock_yolov8n.onnx
streamlit run moslock_v3.py
# Toggle Live Mode in sidebar
```

---

## Files

```
moslock_v3.py                     ← Main Streamlit app
requirements.txt                  ← App dependencies
moslock_branding/
├── brand_config.py               ← Colours, fonts, logo constants
├── theme.css                     ← Complete CSS stylesheet
└── README.md                     ← How to change colours/logo/fonts
yolo_training/
├── data.yaml                     ← 25-class dataset config
├── train.py                      ← YOLOv8n fine-tuning
├── export.py                     ← ONNX + TFLite export
├── inference.py                  ← ONNX inference module
├── generate_synthetic_samples.py ← Synthetic training data generator
├── bootstrap_annotations.py     ← Claude Vision pre-annotation
├── requirements.txt              ← Training-specific dependencies
└── README.md                     ← Full training workflow
demo_images/                      ← Place V2 demo JPEGs here
models/                           ← Place moslock_yolov8n.onnx here
DEPLOYMENT_TAB_S10.md             ← Tab S10 deployment guide
DEMO_INSTRUCTIONS_V3.md          ← Trial site walk-through
PRD_DELTA_V1.1.md                ← Changes from V1.0
```

---

## Detection Classes (25)

- **Locks (2):** personal_lock, permit_lock
- **Isolation Points (10):** circuit_breaker, isolator_switch, fuse, earth_switch, disconnect — each with open/closed states
- **Compartments (10):** HV Incoming → Earth Bar
- **Indicators (3):** indicator_lit, indicator_off, indicator_fault

---

## Safety

MOSLock is a verification **aid**. The GCAA 12-step isolation process (Fatal Hazard Protocol 7) must always be followed. A STOP verdict (indicator_fault) requires mandatory physical verification with a calibrated test instrument before entry.

**Standards:** GCAA Fatal Hazard Protocol 7 | AS/NZS 4836:2023 | AS 4871.1

---

*Mapped Out Solutions® | v3.0.0 | December 2025*
