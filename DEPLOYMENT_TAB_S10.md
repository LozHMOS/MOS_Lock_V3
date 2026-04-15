# MOSLock V3 — Deployment Guide
## Samsung Galaxy Tab S10 + Streamlit Community Cloud
**Mapped Out Solutions® | mos | Lock® v3.0**

---

## Target Hardware

| Device | Samsung Galaxy Tab S10 |
|--------|------------------------|
| Screen | 10.9-inch, 2560×1600, Dynamic AMOLED |
| SoC | Snapdragon 8 Gen 2 |
| RAM | 8 GB |
| OS | Android 14 |
| Storage | 256 GB (min 2 GB free for model + app) |

The ONNX model (~6 MB) runs at approximately **12–20 FPS** on CPU via onnxruntime.
GPU inference is not supported on Android without additional bridge tooling.

---

## Option A — Termux + Python + Streamlit (Recommended — Local, No Root)

This is the primary deployment method for site use. The app runs entirely on the
tablet — no internet connection required after initial setup.

### Prerequisites
- Samsung Galaxy Tab S10 with Android 14
- F-Droid app store installed (or Termux APK from termux.dev)
- Developer options enabled (for USB debugging during setup)

### Step 1: Install Termux

1. Download **Termux** from [f-droid.org](https://f-droid.org/en/packages/com.termux/) or [termux.dev](https://termux.dev)
   - **Do NOT use Google Play Store version** — it is outdated and breaks package installs
2. Open Termux and grant storage permission:
   ```bash
   termux-setup-storage
   ```

### Step 2: Update Termux & Install Python

```bash
pkg update -y && pkg upgrade -y
pkg install -y python python-pip git clang libffi openssl
pip install --upgrade pip
```

> ⏱ Allow 5–10 minutes on first run. Use WiFi.

### Step 3: Install MOSLock Dependencies

```bash
pip install streamlit pillow pandas
pip install onnxruntime numpy
```

> If `onnxruntime` fails to install via pip on ARM64 (Tab S10), try:
> ```bash
> pip install onnxruntime-aarch64  # ARM-specific wheel (if available)
> # OR install from source:
> pkg install cmake ninja && pip install onnxruntime --no-binary :all:
> ```

### Step 4: Copy MOSLock App Files

**Via USB:**
1. Connect Tab S10 to laptop/PC with USB cable
2. Set USB mode to **File Transfer (MTP)** on tablet
3. Copy files to `/sdcard/moslock/`:
   ```
   moslock_v3.py
   moslock_branding/
   demo_images/
   models/moslock_yolov8n.onnx   ← trained model (optional for Demo Mode)
   ```

**Via WiFi/Cloud:**
```bash
# In Termux
mkdir -p ~/moslock
cd ~/moslock

# Copy from internal storage (after USB transfer to /sdcard)
cp -r /sdcard/moslock/* ~/moslock/
```

### Step 5: Copy Demo Images

The demo images from the V2 trial site must be in the `demo_images/` folder:
```bash
mkdir -p ~/moslock/demo_images
# Copy from /sdcard/
cp /sdcard/moslock/demo_images/*.jpg ~/moslock/demo_images/
cp /sdcard/moslock/demo_images/*.jpeg ~/moslock/demo_images/
```

### Step 6: Configure Streamlit for Tablet

Create Streamlit config:
```bash
mkdir -p ~/.streamlit
cat > ~/.streamlit/config.toml << 'EOF'
[server]
headless = true
address = "0.0.0.0"
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
base = "light"
EOF
```

### Step 7: Launch MOSLock

```bash
cd ~/moslock
streamlit run moslock_v3.py
```

The terminal will show:
```
  You can now view your Streamlit app in your browser.
  Network URL: http://192.168.x.x:8501
  External URL: http://xxx.xxx.xxx.xxx:8501
```

### Step 8: Open in Browser

1. Open **Chrome** or **Samsung Internet** on the Tab S10
2. Navigate to: `http://localhost:8501`
3. For full-screen: tap the three dots menu → **Add to Home Screen**

> **Tip:** Create a home screen shortcut so operators can launch MOSLock
> with a single tap, without needing to know about Termux.

### Step 9: Load ONNX Model (Live Mode)

1. Copy the trained model:
   ```bash
   mkdir -p ~/moslock/models
   cp /sdcard/moslock/models/moslock_yolov8n.onnx ~/moslock/models/
   ```
2. In MOSLock sidebar → toggle **Live Mode**
3. Model path should read: `./models/moslock_yolov8n.onnx`
4. Point camera at substation and verify detections

### Battery Expectations

| Activity | Battery Drain |
|----------|--------------|
| Demo Mode (no inference) | ~4% / hour |
| Live Mode — periodic capture (every 30s) | ~8% / hour |
| Live Mode — continuous real-time inference | ~18% / hour |
| Tablet at 100% charge | ~13 hours Demo / ~6 hours Live continuous |

**Recommendation:** Keep tablet on a power bank or charging cable when in Live Mode underground.

### Offline Confirmation

After initial setup, confirm the app works without internet:
1. Enable **Airplane Mode** on tablet
2. Launch Streamlit: `streamlit run moslock_v3.py`
3. Open `http://localhost:8501` in browser
4. Verify: all tabs functional, Demo Mode images load, IFTTT engine runs
5. Live Mode: load ONNX model and test inference with a local photo

All functionality works offline. Google Fonts fallback to system Helvetica/Arial automatically.

---

## Option B — Raspberry Pi as Local WiFi Access Point (Underground / No Internet)

Use this when multiple tablets or laptops need to access MOSLock simultaneously
at an underground site with no internet connectivity.

### Hardware Required
- Raspberry Pi 4 (4 GB RAM minimum) or Pi 5
- USB-C power supply (or large power bank)
- 32 GB+ microSD card
- Optional: rugged case for underground environment

### Step 1: Set Up Raspberry Pi OS

1. Flash Raspberry Pi OS Lite (64-bit) to SD card using Raspberry Pi Imager
2. Enable SSH and configure WiFi in Imager before flashing
3. Boot Pi and connect via SSH:
   ```bash
   ssh pi@raspberrypi.local
   ```

### Step 2: Install Python & Dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git

python3 -m venv ~/moslock_env
source ~/moslock_env/bin/activate

pip install streamlit pillow pandas onnxruntime numpy
```

### Step 3: Configure WiFi Access Point

```bash
sudo apt install -y hostapd dnsmasq

# Create AP config
sudo tee /etc/hostapd/hostapd.conf << 'EOF'
interface=wlan0
driver=nl80211
ssid=MOSLock-AP
hw_mode=g
channel=7
wmm_enabled=0
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0
wpa=2
wpa_passphrase=MOSLockSafe2025
wpa_key_mgmt=WPA-PSK
wpa_pairwise=TKIP
rsn_pairwise=CCMP
EOF

# Configure DHCP
sudo tee /etc/dnsmasq.conf << 'EOF'
interface=wlan0
dhcp-range=192.168.99.10,192.168.99.50,255.255.255.0,24h
EOF

# Set static IP for Pi
sudo tee -a /etc/dhcpcd.conf << 'EOF'
interface wlan0
static ip_address=192.168.99.1/24
nohook wpa_supplicant
EOF

sudo systemctl enable hostapd dnsmasq
sudo systemctl restart hostapd dnsmasq
```

### Step 4: Deploy MOSLock on Pi

```bash
mkdir ~/moslock
# SCP files from your laptop:
scp -r moslock_v3.py moslock_branding/ demo_images/ models/ pi@raspberrypi.local:~/moslock/

# On the Pi:
cd ~/moslock
source ~/moslock_env/bin/activate
streamlit run moslock_v3.py --server.address 0.0.0.0 --server.port 8501 &
```

### Step 5: Auto-Start on Boot

```bash
sudo tee /etc/systemd/system/moslock.service << 'EOF'
[Unit]
Description=MOSLock V3 Streamlit App
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/moslock
ExecStart=/home/pi/moslock_env/bin/streamlit run moslock_v3.py --server.address 0.0.0.0 --server.port 8501
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable moslock
sudo systemctl start moslock
```

### Step 6: Connect Tablets

1. On each tablet: **Settings → WiFi → MOSLock-AP**
2. Password: `MOSLockSafe2025`
3. Open browser: `http://192.168.99.1:8501`
4. Bookmark/add to home screen

### Underground Range
- WiFi range in underground environments: **15–30 metres** (depending on rock type)
- For longer runs, use a second Pi as a WiFi repeater or run ethernet to additional APs

---

## Option C — Streamlit Community Cloud via GitHub

Use this for stakeholder demos from any browser (desktop/laptop/tablet with internet).

### Prerequisites
- GitHub account
- Streamlit Community Cloud account (free at share.streamlit.io)

### Step 1: Create GitHub Repository

```bash
# In your project directory
git init
git add moslock_v3.py moslock_branding/ requirements_streamlit.txt
git commit -m "MOSLock V3 initial deployment"
git remote add origin https://github.com/YOUR_ORG/moslock-v3.git
git push -u origin main
```

### Step 2: Create requirements_streamlit.txt

```
streamlit>=1.40.0
pillow>=11.0.0
pandas>=2.0.0
```

> Note: `onnxruntime` and `numpy` are NOT in the Cloud requirements.
> The app handles their absence gracefully — Live Mode shows an info message.

### Step 3: What NOT to commit

Add to `.gitignore`:
```
# Demo images (too large for GitHub — see Step 5)
demo_images/
# Trained model (too large)
models/
# Termux / local environment
__pycache__/
*.pyc
.streamlit/secrets.toml
```

### Step 4: Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Connect your GitHub repository
4. Set **Main file path**: `moslock_v3.py`
5. Click **Deploy**

Deployment takes 2–3 minutes. Your app is live at:
`https://YOUR_APP.streamlit.app`

### Step 5: Demo Images for Cloud Deployment

Since demo images can't be committed (large binary files), the app generates
a branded placeholder automatically when images are not found.

For the full demo with real images, one of two approaches:
1. **Use Git LFS** (GitHub Large File Storage):
   ```bash
   git lfs install
   git lfs track "demo_images/*.jpg" "demo_images/*.jpeg"
   git add .gitattributes demo_images/
   git commit -m "Add demo images via LFS"
   git push
   ```

2. **Host images on a CDN** and update `IMAGE_SEARCH_DIRS` in `moslock_v3.py`
   to include the URL (requires `requests` library).

### Step 6: Streamlit Secrets (Optional)

For future API integrations, use Streamlit secrets:
```toml
# .streamlit/secrets.toml (local only — never commit this)
[anthropic]
api_key = "your_key_here"
```

---

## ONNX Model File Management

### File size
- `moslock_yolov8n.onnx`: approximately **6.2 MB**
- Loads in ~1–2 seconds on Tab S10

### Model versioning
Keep multiple versions:
```
models/
├── moslock_yolov8n_v1_synthetic.onnx    ← trained on synthetic data only
├── moslock_yolov8n_v2_real.onnx         ← trained with real site photos
└── moslock_yolov8n.onnx                 ← symlink or copy of current best
```

Update the model path in the sidebar without restarting the app.

### Model refresh process (field update)
1. Train new model on laptop: `python train.py && python export.py`
2. Transfer via USB or WiFi to tablet: `adb push models/moslock_yolov8n.onnx /sdcard/moslock/models/`
3. Copy from internal storage: `cp /sdcard/moslock/models/*.onnx ~/moslock/models/`
4. Reload page — the app caches the model per session, so a browser refresh loads the new model

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Streamlit not found in Termux | `pkg install python && pip install streamlit` |
| onnxruntime install fails on ARM | Try `pip install onnxruntime --only-binary :all:` |
| Camera not working in browser | Use HTTPS (required for camera API) — set up nginx + SSL, or use http on localhost |
| App crashes on startup | Check `streamlit run moslock_v3.py 2>&1 | head -50` for error |
| Demo images not showing | Check `demo_images/` folder exists and contains the JPEG files |
| Sidebar not collapsing | Normal — tap ← arrow or swipe left edge to collapse |
| Font looks wrong | Helvetica not on Android — Arial fallback is used, appearance is near-identical |
| Model path error | Ensure ONNX file is at `./models/moslock_yolov8n.onnx` relative to `moslock_v3.py` |

---

*Mapped Out Solutions® | mos | Lock® V3.0 | "We will always have a plan."*
