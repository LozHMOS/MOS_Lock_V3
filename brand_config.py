"""
moslock_branding/brand_config.py
Mapped Out Solutions® — MOSLock V3 Brand Constants
Source of truth for all colours, typography and logo assets.

Brand Guide: MOS Brand Guidelines V2, December 2025
Primary colour (screen/digital): Cyan/Turquoise #3ECFCF
Primary colour (print): Dark Navy #071539
Logo disc: Near-black #2d2926
Secondary Purple: #D04EFF  |  Secondary Blue: #007BFF
Typography: Helvetica (Light / Regular / Black), Arial fallback
"""

import base64

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE — sourced directly from MOS Brand Guidelines V2 Dec 2025
# ─────────────────────────────────────────────────────────────────────────────

# Primary
MOS_CYAN         = "#3ECFCF"   # Screen/digital primary — turquoise/teal
MOS_DARK_NAVY    = "#071539"   # Print primary — very dark navy (Pantone 533C, RAL 5026)
MOS_NEAR_BLACK   = "#2d2926"   # Logo disc colour (RGB 45/41/38)

# Secondary
MOS_PURPLE       = "#D04EFF"   # Secondary — vivid purple (CMYK 56/73/0/0)
MOS_BLUE         = "#007BFF"   # Secondary — strong blue (CMYK 83/57/0/0)

# Neutrals (black tints per brand guide)
MOS_BLACK_100    = "#1A1A1A"   # 100%
MOS_BLACK_90     = "#2E2E2E"   # 90%
MOS_BLACK_80     = "#404040"   # 80%
MOS_BLACK_60     = "#666666"   # 60%
MOS_BLACK_40     = "#999999"   # 40%
MOS_BLACK_20     = "#CCCCCC"   # 20%
MOS_BLACK_10     = "#E5E5E5"   # 10%
MOS_BLACK_05     = "#F2F2F2"   # 5%
MOS_WHITE        = "#FFFFFF"

# UI Surface colours
MOS_SIDEBAR_BG   = "#071539"   # Sidebar: dark navy canvas
MOS_CARD_BG      = "#FFFFFF"   # Card surfaces
MOS_PAGE_BG      = "#F4F6F9"   # Main page background
MOS_BORDER       = "#E2E6EA"   # Subtle borders

# ─────────────────────────────────────────────────────────────────────────────
# SAFETY STATUS COLOURS
# Preserve safety meaning: red = danger, amber = caution, green = safe.
# Shades are brand-consistent (using MOS palette relatives).
# ─────────────────────────────────────────────────────────────────────────────

SAFE_GREEN       = "#1ABF74"   # ALL CLEAR / Isolated — vivid safe green
CAUTION_AMBER    = "#F5A623"   # REVIEW REQUIRED — warm amber
DANGER_RED       = "#E8392D"   # FAIL / Energised — strong safety red
STOP_RED         = "#CC0000"   # STOP (indicator_fault) — deep red
INFO_TEAL        = MOS_CYAN    # Informational — uses brand primary

# Compartment state colours
COMPARTMENT_ISOLATED  = "#1ABF74"   # green
COMPARTMENT_ENERGISED = "#E8392D"   # red
COMPARTMENT_UNKNOWN   = "#F5A623"   # amber

# Verdict colours
VERDICT_ALL_CLEAR    = "#1ABF74"
VERDICT_DISCREPANCY  = "#E8392D"
VERDICT_REVIEW       = "#F5A623"
VERDICT_STOP         = "#CC0000"

# ─────────────────────────────────────────────────────────────────────────────
# TYPOGRAPHY — Helvetica family per brand guide; Google Fonts fallback for web
# ─────────────────────────────────────────────────────────────────────────────

FONT_STACK       = "'Helvetica Neue', 'Helvetica', 'Arial', sans-serif"
GOOGLE_FONT_URL  = "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700;900&display=swap"
# Inter is the closest freely-available analogue to Helvetica for web.
# The system Helvetica Neue will take priority on macOS/iOS.

FONT_WEIGHT_LIGHT   = 300
FONT_WEIGHT_REGULAR = 400
FONT_WEIGHT_BOLD    = 700
FONT_WEIGHT_BLACK   = 900

# Font sizes — tablet-optimised (Tab S10 minimum 48px touch targets)
FONT_SIZE_BODY     = "16px"
FONT_SIZE_SMALL    = "14px"
FONT_SIZE_LARGE    = "18px"
FONT_SIZE_H1       = "28px"
FONT_SIZE_H2       = "22px"
FONT_SIZE_H3       = "18px"
FONT_SIZE_BUTTON   = "18px"
FONT_SIZE_LABEL    = "13px"

# ─────────────────────────────────────────────────────────────────────────────
# APP IDENTITY
# ─────────────────────────────────────────────────────────────────────────────

APP_NAME         = "MOSLock"
APP_VERSION      = "3.0.0"
PRODUCT_NAME     = "mos | Lock\u00ae"          # mos | Lock®
COMPANY_NAME     = "Mapped Out Solutions\u00ae"
STRAPLINE        = "We will always have a plan."
BUILD_DATE       = "2025-12"

# Standards referenced
STANDARDS = [
    "Glencore GCAA Fatal Hazard Protocol 7",
    "AS/NZS 4836:2023 Safe Working on LV Electrical Installations",
    "AS 4871.1 Electrical Equipment for Coal Mines",
    "Queensland Mines Safety Legislation",
]

# ─────────────────────────────────────────────────────────────────────────────
# LOGO SVG — MOS M-Disc + Wordmark (two variants)
# Based on MOS Brand Guidelines V2 Dec 2025
# The M-disc is a dark charcoal circle (#2d2926) with a white geometric M
# formed by two mirrored bracket-connector shapes — "digital and code feel"
# ─────────────────────────────────────────────────────────────────────────────

def _make_logo_svg(dark_bg: bool = False) -> str:
    """
    Generate MOS logo SVG.
    dark_bg=True  → white-out version (white disc, dark M, white wordmark)
    dark_bg=False → standard version (dark disc, white M, dark wordmark)
    Per brand guide: white-out MUST be used on dark/photo backgrounds.
    """
    if dark_bg:
        disc_fill   = MOS_WHITE
        mark_stroke = MOS_NEAR_BLACK
        text_fill   = MOS_WHITE
        reg_fill    = MOS_WHITE
    else:
        disc_fill   = MOS_NEAR_BLACK
        mark_stroke = MOS_WHITE
        text_fill   = MOS_NEAR_BLACK
        reg_fill    = MOS_NEAR_BLACK

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 68" width="240" height="68">
  <!-- MOS M-Disc graphic -->
  <circle cx="34" cy="34" r="32" fill="{disc_fill}"/>
  <!-- M mark: two mirrored connector/bracket shapes (circuit-board aesthetic) -->
  <g fill="none" stroke="{mark_stroke}" stroke-width="5.5" stroke-linecap="round" stroke-linejoin="round">
    <!-- Left bracket: arms pointing left, closed on right -->
    <path d="M 18 48 L 18 26 L 28 26 L 28 48"/>
    <!-- Right bracket: arms pointing right, closed on left -->
    <path d="M 40 48 L 40 26 L 50 26 L 50 48"/>
  </g>
  <!-- Terminal dots — rounded caps at outer top corners (circuit-board terminals) -->
  <circle cx="18" cy="26" r="3.5" fill="{mark_stroke}"/>
  <circle cx="50" cy="26" r="3.5" fill="{mark_stroke}"/>
  <!-- Wordmark: "mapped out solutions" + registered mark -->
  <text x="80" y="30"
        font-family="Helvetica Neue, Helvetica, Arial, sans-serif"
        font-size="14" font-weight="300" fill="{text_fill}" letter-spacing="0.3">mapped out solutions</text>
  <text x="229" y="28"
        font-family="Helvetica Neue, Helvetica, Arial, sans-serif"
        font-size="9" fill="{reg_fill}">\u00ae</text>
  <!-- Product sub-brand: mos | Lock -->
  <text x="80" y="52"
        font-family="Helvetica Neue, Helvetica, Arial, sans-serif"
        font-size="18" font-weight="300" fill="{MOS_CYAN}" letter-spacing="0.5">mos</text>
  <text x="111" y="52"
        font-family="Helvetica Neue, Helvetica, Arial, sans-serif"
        font-size="18" font-weight="300" fill="{text_fill}" letter-spacing="0.5"> | Lock</text>
  <text x="178" y="50"
        font-family="Helvetica Neue, Helvetica, Arial, sans-serif"
        font-size="10" fill="{text_fill}">\u00ae</text>
</svg>"""
    return svg


def _svg_to_b64(svg_str: str) -> str:
    return base64.b64encode(svg_str.encode("utf-8")).decode("utf-8")


# Pre-built base64 variants — use these throughout the app
LOGO_SVG_DARK_BG   = _make_logo_svg(dark_bg=True)    # for sidebar (dark navy bg)
LOGO_SVG_LIGHT_BG  = _make_logo_svg(dark_bg=False)   # for main area (white bg)
LOGO_B64_DARK_BG   = _svg_to_b64(LOGO_SVG_DARK_BG)
LOGO_B64_LIGHT_BG  = _svg_to_b64(LOGO_SVG_LIGHT_BG)


def logo_img_tag(dark_bg: bool = True, width: int = 200) -> str:
    """Return an <img> tag with the embedded logo for use in st.markdown."""
    b64 = LOGO_B64_DARK_BG if dark_bg else LOGO_B64_LIGHT_BG
    return (
        f'<img src="data:image/svg+xml;base64,{b64}" '
        f'width="{width}" alt="Mapped Out Solutions — mos | Lock®" '
        f'style="display:block; margin-bottom:8px;"/>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION CLASS METADATA — 25 effective classes
# ─────────────────────────────────────────────────────────────────────────────

CLASS_METADATA = {
    # Locks
    "personal_lock":          {"label": "Personal Lock",          "color": "#FF6B35", "group": "Lock",             "id": 0},
    "permit_lock":            {"label": "Permit Lock",            "color": "#F5A623", "group": "Lock",             "id": 1},
    # Isolation Points — Open states (safe)
    "circuit_breaker_open":   {"label": "CB Open",                "color": "#1ABF74", "group": "Isolation Point",  "id": 2},
    "circuit_breaker_closed": {"label": "CB Closed",              "color": "#E8392D", "group": "Isolation Point",  "id": 3},
    "isolator_switch_open":   {"label": "Isolator Open",          "color": "#1ABF74", "group": "Isolation Point",  "id": 4},
    "isolator_switch_closed": {"label": "Isolator Closed",        "color": "#E8392D", "group": "Isolation Point",  "id": 5},
    "fuse_withdrawn":         {"label": "Fuse Withdrawn",         "color": "#1ABF74", "group": "Isolation Point",  "id": 6},
    "fuse_inserted":          {"label": "Fuse Inserted",          "color": "#E8392D", "group": "Isolation Point",  "id": 7},
    "earth_switch_applied":   {"label": "Earth Applied",          "color": "#007BFF", "group": "Isolation Point",  "id": 8},
    "earth_switch_removed":   {"label": "Earth Removed",          "color": "#F5A623", "group": "Isolation Point",  "id": 9},
    "disconnect_open":        {"label": "Disconnect Open",        "color": "#1ABF74", "group": "Isolation Point",  "id": 10},
    "disconnect_closed":      {"label": "Disconnect Closed",      "color": "#E8392D", "group": "Isolation Point",  "id": 11},
    # Compartments
    "compartment_1":          {"label": "HV Incoming",            "color": "#3ECFCF", "group": "Compartment",      "id": 12},
    "compartment_2":          {"label": "Transformer",            "color": "#3ECFCF", "group": "Compartment",      "id": 13},
    "compartment_3":          {"label": "LV Outgoing",            "color": "#3ECFCF", "group": "Compartment",      "id": 14},
    "compartment_4":          {"label": "Control",                "color": "#3ECFCF", "group": "Compartment",      "id": 15},
    "compartment_5":          {"label": "Auxiliary",              "color": "#3ECFCF", "group": "Compartment",      "id": 16},
    "compartment_6":          {"label": "Bus Section",            "color": "#3ECFCF", "group": "Compartment",      "id": 17},
    "compartment_7":          {"label": "Metering",               "color": "#3ECFCF", "group": "Compartment",      "id": 18},
    "compartment_8":          {"label": "Protection Relay",       "color": "#3ECFCF", "group": "Compartment",      "id": 19},
    "compartment_9":          {"label": "Cable Termination",      "color": "#3ECFCF", "group": "Compartment",      "id": 20},
    "compartment_10":         {"label": "Earth Bar",              "color": "#3ECFCF", "group": "Compartment",      "id": 21},
    # Live Line Indicators
    "indicator_lit":          {"label": "Indicator LIT ⚡",        "color": "#E8392D", "group": "Live Indicator",   "id": 22},
    "indicator_off":          {"label": "Indicator Off ✓",        "color": "#1ABF74", "group": "Live Indicator",   "id": 23},
    "indicator_fault":        {"label": "Indicator FAULT ⚠",      "color": "#CC0000", "group": "Live Indicator",   "id": 24},
}

CLASS_NAMES = list(CLASS_METADATA.keys())   # preserves order, id matches index
