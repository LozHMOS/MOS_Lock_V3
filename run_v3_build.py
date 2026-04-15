import anthropic
import os
import time

client = anthropic.Anthropic()

DOCS_FOLDER = "/Users/lozhemmings/MOSLock_V3_docs"
BETA_HEADER = "managed-agents-2026-04-01"
FILES_HEADER = "files-api-2025-04-14"
AGENT_ID = "agent_011CZtocfpSLLACtBTknn93u"
ENV_ID = "env_01GgjkYLBcwjqABdV8Bgpw8a"

print("=" * 60)
print("STEP 1: Uploading files from", DOCS_FOLDER)
print("=" * 60)

resources = []
supported_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".svg", ".docx", ".txt", ".md"}

for filename in sorted(os.listdir(DOCS_FOLDER)):
    ext = os.path.splitext(filename)[1].lower()
    if ext not in supported_extensions:
        print(f"  Skipping {filename} (unsupported type)")
        continue
    filepath = os.path.join(DOCS_FOLDER, filename)
    if not os.path.isfile(filepath):
        continue
    print(f"  Uploading: {filename}...")
    with open(filepath, "rb") as f:
        uploaded = client.beta.files.upload(
            file=(filename, f),
            betas=[FILES_HEADER],
        )
    print(f"     Done: {uploaded.id}")
    if "brand" in filename.lower() or "style" in filename.lower() or "logo" in filename.lower():
        mount_dir = "/workspace/branding"
    else:
        mount_dir = "/workspace/images"
    resources.append({
        "type": "file",
        "file_id": uploaded.id,
        "mount_path": f"{mount_dir}/{filename}",
    })

print(f"\nUploaded {len(resources)} file(s)\n")

print("=" * 60)
print("STEP 2: Creating session")
print("=" * 60)

session = client.beta.sessions.create(
    agent=AGENT_ID,
    environment_id=ENV_ID,
    title="MOSLock V3 - Branded YOLO + Tablet Trial Build",
    resources=resources,
    betas=[BETA_HEADER],
)

print(f"  Session created: {session.id}")
print(f"  Status: {session.status}\n")

with open("session_info.txt", "w") as sf:
    sf.write(f"SESSION_ID={session.id}\n")
    sf.write(f"AGENT_ID={AGENT_ID}\n")
    sf.write(f"ENV_ID={ENV_ID}\n")

print("=" * 60)
print("STEP 3: Sending V3 build prompt")
print("=" * 60)

V3_PROMPT = """
I've uploaded our brand styling guide and logo to /workspace/branding/ and all our equipment photos from the V2 demo to /workspace/images/. Read ALL uploaded files first before writing any code.

Build MOSLock V3 — a branded, tablet-ready Streamlit web app with an integrated YOLO training pipeline. This is for a trial site demo in 2 weeks on a Samsung Galaxy Tab S10 (Android). The Streamlit app will also be deployed via GitHub + Streamlit Community Cloud so stakeholders can view the branded demo from any desktop/laptop browser.

DETECTION CLASSES (25 effective classes):

Locks (2 types):
- personal_lock (individual worker padlock — colour-coded, typically red/blue/yellow)
- permit_lock (permit holder's lock — typically larger, numbered, attached to lock box)

Isolation Points (5 types, each with 2 states = 10 state classes):
- circuit_breaker_open / circuit_breaker_closed
- isolator_switch_open / isolator_switch_closed
- fuse_withdrawn / fuse_inserted
- earth_switch_applied / earth_switch_removed
- disconnect_open / disconnect_closed

Compartments (10):
- compartment_1 through compartment_10 (distinct zones on the mobile substation — HV incoming, transformer, LV outgoing, control, auxiliary, bus section, metering, protection relay, cable termination, earth bar)

Live Line Indicators (3 states):
- indicator_lit (voltage present)
- indicator_off (no voltage)
- indicator_fault (indicator damaged/obscured — triggers mandatory manual verification)

BUILD ORDER — write ALL files to /mnt/session/outputs/:

1. YOLO TRAINING PIPELINE (yolo_training/)
   - data.yaml with all 25 classes
   - train.py — YOLOv8n fine-tuning (nano model for tablet inference speed)
   - export.py — export to ONNX + TFLite
   - inference.py — single-image inference returning class, confidence, bbox
   - generate_synthetic_samples.py — creates 30 synthetic training images per class with augmentation (rotation, brightness, noise, scale). Uses coloured shapes, labels, and patterns as stand-ins until real site photos replace them.
   - bootstrap_annotations.py — uses Claude vision API to pre-annotate real equipment photos in YOLO .txt format for human review in CVAT/LabelImg
   - requirements.txt (pinned: ultralytics, opencv-python-headless, onnxruntime, anthropic, Pillow, numpy)
   - README.md — full workflow from synthetic data to trained model to tablet deployment

2. STREAMLIT V3 APP (moslock_v3.py)
   Take the V2 app structure as the base. Apply ALL changes:

   a) BRANDING — read /workspace/branding/ files and apply:
      - All colours from the brand guide to buttons, headers, sections, sidebar, KPI cards, status indicators, risk badges
      - Logo at top of sidebar + top-left of main area (embed as base64 for portability)
      - Typography from brand guide via CSS injection with Google Fonts (system font fallback for offline)
      - Risk indicators KEEP safety meaning (red=danger, amber=caution, green=safe) but use brand-consistent shades
      - Footer with company branding, version, standards references
      - Output moslock_branding/theme.css (all styles in one swappable file)
      - Output moslock_branding/brand_config.py (colours, fonts, logo as Python constants)
      - Output moslock_branding/README.md (how to change colours/logo/fonts)

   b) DEMO MODE (default — no model, no ONNX, no special packages required):
      - Pre-loaded detection results for the uploaded substation images
      - Draws bounding boxes on equipment photos with class labels + confidence
      - IFTTT rule engine compares detections against Training Model rules
      - Pass/fail/manual-review verdict per isolation point
      - All 10 compartments in status table
      - Lock count derived from detected locks vs signed-on workers
      - indicator_fault triggers STOP alert

   c) LIVE MODE (toggle — requires trained .onnx model on device):
      - Camera capture then ONNX inference then overlay detections then IFTTT engine then verdict
      - Model path configurable (default: ./models/moslock_yolov8n.onnx)
      - Confidence threshold slider (default 90%, range 50-99%)

   d) TABLET OPTIMISATION (Samsung Tab S10, 10.1 inch, 1920x1200):
      - Min touch target 48px
      - Body font 16px, headers 20px+, buttons 18px
      - High-contrast mode toggle for underground lighting
      - Camera input PRIMARY, file upload secondary
      - Sidebar collapsed by default on narrow screens
      - Tables horizontally scrollable
      - No hover-dependent interactions

   e) IFTTT RULE ENGINE:
      - Training Model tab defines rules mapping isolation point states to expected compartment states and indicator states
      - Verification engine checks YOLO detections against rules
      - Per-rule: PASS (green) / FAIL (red) / REVIEW (amber)
      - Overall: ALL CLEAR / DISCREPANCY / REVIEW REQUIRED
      - Verdict blocked from ALL CLEAR if ANY detection below threshold

   f) GRACEFUL DEGRADATION — app MUST run in three scenarios without crashes:
      Scenario 1 REMOTE DEMO (desktop/laptop): pip install streamlit pillow pandas — all YOLO/ONNX imports lazy with try/except, missing images show placeholder, Live Mode shows info message, all tabs functional
      Scenario 2 TABLET DEMO (Tab S10, no model): same as 1 plus camera, Demo Mode overlays pre-loaded detections
      Scenario 3 TABLET LIVE (Tab S10, trained model): full pipeline with real ONNX inference

3. DEPLOYMENT GUIDE (DEPLOYMENT_TAB_S10.md)
   - Samsung Galaxy Tab S10 specific
   - Option A: Termux + Python + Streamlit (local, no root)
   - Option B: Raspberry Pi as local WiFi AP underground
   - Option C: Streamlit Community Cloud via GitHub
   - Step-by-step for each, ONNX model loading, battery expectations, offline confirmation

4. DEMO INSTRUCTIONS (DEMO_INSTRUCTIONS_V3.md)
   - Trial site walk-through script with talking points
   - FAQ for electrical engineers and safety managers

5. PRD DELTA (PRD_DELTA_V1.1.md)
   - Changes from v1.0, what is delivered vs future, updated timeline

Write ALL files to /mnt/session/outputs/. Working code throughout — no placeholders, no TODO comments. This is the build.
"""

client.beta.sessions.events.send(
    session_id=session.id,
    events=[{
        "type": "user.message",
        "content": [{"type": "text", "text": V3_PROMPT}],
    }],
    betas=[BETA_HEADER],
)

print("  Prompt sent\n")

print("=" * 60)
print("STEP 4: Watching the agent build (20-40 minutes)")
print("=" * 60)
print()

try:
    with client.beta.sessions.events.stream(
        session_id=session.id,
        betas=[BETA_HEADER],
    ) as stream:
        for event in stream:
            if event.type == "agent.message":
                if hasattr(event, "content"):
                    for block in event.content:
                        if hasattr(block, "text"):
                            preview = block.text[:200]
                            print(f"\nAGENT: {preview}{'...' if len(block.text) > 200 else ''}")

            elif event.type == "agent.tool_use":
                name = getattr(event, "name", "?")
                inp = getattr(event, "input", {})
                if name == "write":
                    print(f"\nWRITING: {inp.get('file_path', '?')}")
                elif name == "bash":
                    cmd = str(inp.get("command", "?"))[:100]
                    print(f"\nBASH: {cmd}")
                elif name == "read":
                    print(f"\nREADING: {inp.get('file_path', '?')}")
                else:
                    print(f"\nTOOL: {name}")

            elif event.type == "agent.tool_result":
                is_err = getattr(event, "is_error", False)
                print(f"  {'ERROR' if is_err else 'Done'}")

            elif event.type == "session.status_idle":
                print("\n" + "=" * 60)
                print("BUILD COMPLETE")
                print("=" * 60)
                break

            elif event.type == "session.status_terminated":
                print("\nSession terminated")
                break

            elif event.type == "session.error":
                print(f"\nERROR: {event}")

except KeyboardInterrupt:
    print(f"\n\nInterrupted. Session still running: {session.id}")

print(f"\n{'=' * 60}")
print("STEP 5: Listing output files")
print("=" * 60)

time.sleep(5)

try:
    files = client.beta.files.list(
        session_id=session.id,
        betas=[FILES_HEADER],
    )
    file_list = list(files.data) if hasattr(files, "data") else []
    if not file_list:
        print("  Files still indexing. Run download_outputs.py in a minute.")
    for f in file_list:
        size = getattr(f, "size_bytes", 0)
        name = getattr(f, "filename", "unknown")
        print(f"  {name} ({size:,} bytes) - {f.id}")
except Exception as e:
    print(f"  Could not list files yet: {e}")

print(f"\nSession ID: {session.id}")
print("Run: python3 download_outputs.py")