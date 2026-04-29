import os
import io
import tempfile

import numpy as np
import requests
import soundfile as sf
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import torchaudio
import torch

from features import extract_features


# ─────────────────────────────────────────────────────────────────
# HELPERS  (ML / audio logic — unchanged)
# ─────────────────────────────────────────────────────────────────

def preview_rms_from_bytes(audio_bytes: bytes) -> float:
    try:
        buf = io.BytesIO(audio_bytes)
        audio_np, _ = sf.read(buf)
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        return float(np.sqrt(np.mean(audio_np ** 2)))
    except Exception:
        return 0.0


def get_openweather_temp(city: str, country: str | None = None) -> float | None:
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return None
    q = city if not country else f"{city},{country}"
    try:
        resp = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": q, "appid": api_key, "units": "metric"},
            timeout=5,
        )
        return float(resp.json()["main"]["temp"]) if resp.status_code == 200 else None
    except Exception:
        return None


@st.cache_resource
def load_model():
    for path in ["models/sonabee_rf.pkl", "models/model.pkl", "models/bee_model.pkl"]:
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception:
                continue
    return None


def prepare_feature_vector(audio_path: str, temp_c: float, model) -> np.ndarray | None:
    try:
        feats = np.asarray(extract_features(audio_path)).flatten()
    except Exception as e:
        st.error(f"Feature extraction error: {e}")
        return None

    model_dim = getattr(model, "n_features_in_", None)
    if model_dim is None:
        return np.append(feats, temp_c).reshape(1, -1)

    if model_dim == len(feats) + 1:
        fv = np.append(feats, temp_c)
    elif model_dim == len(feats):
        fv = feats
    elif len(feats) < model_dim:
        fv = np.pad(feats, (0, model_dim - len(feats)))
    else:
        fv = feats[:model_dim]
    return fv.reshape(1, -1)


def analyze_acoustics(waveform: torch.Tensor, sr: int) -> dict:
    if waveform.ndim > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    rms = torch.sqrt(torch.mean(waveform ** 2)).item()
    spec = torch.fft.rfft(waveform)
    mag = torch.abs(spec).squeeze(0)
    freqs = torch.fft.rfftfreq(waveform.shape[-1], d=1.0 / sr)
    centroid = ((freqs * mag).sum() / (mag.sum() + 1e-9)).item()

    messages = []
    if rms < 0.01:
        messages.append("Low overall hive activity — quiet buzz amplitude.")
    elif rms > 0.06:
        messages.append("Elevated buzz intensity, suggesting agitation or heavy activity.")
    else:
        messages.append("Moderate buzz level consistent with typical colony activity.")

    if 200 <= centroid <= 320:
        messages.append("Dominant frequency within expected brood/worker buzz range.")
    elif centroid > 320:
        messages.append("Higher-than-usual dominant frequency — may reflect agitation, ventilation, or disturbed bees.")
    else:
        messages.append("Lower dominant frequency — could indicate reduced activity or cold, clustered bees.")

    return {"rms": rms, "centroid": centroid, "messages": messages}


def plot_spectrogram(waveform: torch.Tensor, sr: int):
    if waveform.ndim > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    mel_db = torchaudio.transforms.AmplitudeToDB()(
        torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=2048, hop_length=512, n_mels=64)(waveform)
    )
    fig, ax = plt.subplots(figsize=(7, 3), facecolor="#FAF7F2")
    im = ax.imshow(mel_db.squeeze(0).numpy(), origin="lower", aspect="auto",
                   extent=[0, waveform.shape[-1] / sr, 0, sr / 2], cmap="magma")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Mel band"); ax.set_title("Mel Spectrogram")
    fig.colorbar(im, ax=ax, label="dB"); fig.tight_layout()
    return fig


def plot_mfcc(waveform: torch.Tensor, sr: int):
    if waveform.ndim > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=sr, n_mfcc=13, melkwargs={"n_fft": 2048, "hop_length": 512}
    )(waveform).squeeze(0).numpy()
    fig, ax = plt.subplots(figsize=(7, 3), facecolor="#FAF7F2")
    im = ax.imshow(mfcc, origin="lower", aspect="auto",
                   extent=[0, waveform.shape[-1] / sr, 1, mfcc.shape[0]], cmap="viridis")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("MFCC Coefficient"); ax.set_title("MFCCs Over Time")
    fig.colorbar(im, ax=ax, label="Coefficient value"); fig.tight_layout()
    return fig


def interpret_prediction(label: int, prob_risk: float, acoustics: dict, temp_c: float):
    if label == 0:
        status, emoji = "Healthy", "🟢"
    elif prob_risk > 0.9:
        status, emoji = "At risk", "🔴"
    else:
        status, emoji = "Watch", "🟡"

    why = list(acoustics.get("messages", []))
    if temp_c < 15:
        why.append("Ambient temperature is low — bees may be clustering to maintain brood warmth.")
    elif temp_c > 35:
        why.append("Ambient temperature is high — increased fanning may alter the acoustic profile.")

    actions = (
        [
            "Continue regular inspections every 1–2 weeks.",
            "Monitor for seasonal changes in activity and hive weight.",
            "Use Sonabee periodically to build a healthy acoustic baseline.",
        ] if label == 0 else [
            "Inspect for queen presence and brood pattern within 24–48 hours.",
            "Check for signs of Varroa or other parasites.",
            "Verify food stores and recent weather stress.",
            "Compare with past audio from the same hive, if available.",
        ]
    )
    return status, emoji, why, actions


# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sonabee – Acoustic Hive Intelligence",
    page_icon="🐝",
    layout="centered",
)

# ─────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Base ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
[data-testid="stMain"] {
    background-color: #F5F0E8 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    color: #1C1C1E;
}
[data-testid="stMainBlockContainer"] { padding-top: 1rem !important; }

/* ── Hide chrome ── */
#MainMenu, footer, header, [data-testid="stDecoration"] { display: none !important; }

/* ── Top divider accent ── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    display: block;
    height: 4px;
    background: linear-gradient(90deg, #F5B942, #F5D98B);
    position: fixed; top: 0; left: 0; right: 0; z-index: 999;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent;
    gap: 0.15rem;
    border-bottom: 1.5px solid #E8E2D8;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-size: 0.82rem;
    font-weight: 500;
    color: #9A9080;
    padding: 0.5rem 0.9rem;
    border-radius: 8px 8px 0 0;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #C8860A !important;
    border-bottom: 2.5px solid #F5B942 !important;
    background: rgba(245,185,66,0.07) !important;
}

/* ── Primary button → amber, pill-shaped ── */
[data-testid="stButton"] > button[kind="primary"] {
    background: #E8A020 !important;
    background: linear-gradient(180deg, #F5B942 0%, #E8A020 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.75rem 1.5rem !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 4px 14px rgba(232,160,32,0.38) !important;
    transition: opacity 0.15s ease, box-shadow 0.15s ease !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    opacity: 0.92 !important;
    box-shadow: 0 6px 20px rgba(232,160,32,0.48) !important;
}

/* ── Secondary buttons ── */
[data-testid="stButton"] > button[kind="secondary"] {
    border-radius: 10px !important;
    border: 1.5px solid #DDD8CF !important;
    color: #555 !important;
    background: #fff !important;
}

/* ── Radio → pill toggles ── */
div[data-testid="stRadio"] > div {
    display: flex !important;
    flex-direction: row !important;
    gap: 0.5rem !important;
    flex-wrap: nowrap !important;
}
div[data-testid="stRadio"] label {
    flex: 1 !important;
    background: #FFFFFF !important;
    border: 1.5px solid #DDD8CF !important;
    border-radius: 999px !important;
    padding: 0.45rem 1rem !important;
    text-align: center !important;
    cursor: pointer !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #555 !important;
    transition: all 0.15s ease !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
div[data-testid="stRadio"] label:has(input:checked) {
    background: #F5B942 !important;
    border-color: #F5B942 !important;
    color: #fff !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 8px rgba(245,185,66,0.35) !important;
}
div[data-testid="stRadio"] label input { display: none !important; }
div[data-testid="stRadio"] label p {
    margin: 0 !important;
    font-size: 0.85rem !important;
    font-weight: inherit !important;
    color: inherit !important;
}

/* ── Slider → amber ── */
[data-testid="stSlider"] [class*="TrackFill"],
[data-testid="stSlider"] [class*="track"] > div:first-child {
    background-color: #F5B942 !important;
}
[data-testid="stSlider"] [class*="thumb"] {
    background-color: #4A8FD4 !important;
    border: 2px solid #fff !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.2) !important;
    width: 18px !important; height: 18px !important;
}

/* ── File uploader → dashed card ── */
[data-testid="stFileUploader"] section {
    border: 2px dashed #CEBFA0 !important;
    border-radius: 14px !important;
    background: #FDFAF5 !important;
    padding: 1.5rem 1rem !important;
    text-align: center !important;
    transition: border-color 0.15s ease !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #F5B942 !important;
    background: #FFFDF7 !important;
}
[data-testid="stFileUploader"] section [data-testid="stMarkdownContainer"] p {
    font-size: 0.84rem !important;
    color: #9A9080 !important;
}

/* ── Bordered containers ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 18px !important;
    border: 1.5px solid #EDE8DF !important;
    background: #FFFFFF !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05) !important;
    padding: 0.1rem 0.1rem !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    border-radius: 14px !important;
    border: 1.5px solid #EDE8DF !important;
    background: #FFFFFF !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04) !important;
}
[data-testid="stExpander"] summary {
    font-weight: 600; font-size: 0.88rem; color: #1C1C1E;
    padding: 0.75rem 1rem !important;
}

/* ── Captions ── */
[data-testid="stCaptionContainer"] p {
    color: #9A9080 !important;
    font-size: 0.78rem !important;
}

/* ── Audio player ── */
audio { width: 100% !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# REUSABLE HTML COMPONENTS
# ─────────────────────────────────────────────────────────────────

def _section_card(content_html: str, padding: str = "1.25rem 1.4rem") -> str:
    """White rounded card wrapper — used for input sections."""
    return f"""
    <div style="
        background:#FFFFFF;
        border-radius:18px;
        border:1.5px solid #EDE8DF;
        box-shadow:0 2px 12px rgba(0,0,0,0.05);
        padding:{padding};
        margin-bottom:0.9rem;
    ">{content_html}</div>"""


def _metric_card(label: str, value: str, sub: str, detail: str, bg: str, accent: str) -> str:
    """Single metric result card with distinct pastel background."""
    return f"""
    <div style="
        background:{bg};
        border-radius:16px;
        border:1px solid rgba(0,0,0,0.05);
        padding:1rem 1.1rem;
        margin-bottom:0.65rem;
    ">
      <div style="
          font-size:0.68rem;
          font-weight:700;
          letter-spacing:0.1em;
          text-transform:uppercase;
          color:#8A8078;
          margin-bottom:0.45rem;
      ">{label}</div>
      <div style="margin-bottom:0.4rem;">
        <span style="font-size:1.3rem;font-weight:800;color:#1C1C1E;">{value}</span>
        <span style="font-size:0.85rem;font-weight:500;color:{accent};margin-left:0.4rem;">{sub}</span>
      </div>
      <div style="font-size:0.82rem;color:#6A6058;line-height:1.45;">{detail}</div>
    </div>"""


def _hive_card(name: str, status: str, last_check: str, trend: str,
               note: str, bg: str, border: str, badge_bg: str,
               badge_color: str, trend_color: str) -> str:
    """Full-width hive status card for the dashboard."""
    return f"""
    <div style="
        background:{bg};
        border-radius:16px;
        border:1.5px solid {border};
        padding:1rem 1.15rem 0.85rem;
        margin-bottom:0.75rem;
        box-shadow:0 2px 8px rgba(0,0,0,0.04);
    ">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.3rem;">
        <div style="font-size:1rem;font-weight:700;color:#1C1C1E;">{name}</div>
        <span style="
            background:{badge_bg};
            color:{badge_color};
            border-radius:999px;
            padding:3px 12px;
            font-size:0.74rem;
            font-weight:700;
            white-space:nowrap;
            border:1.5px solid {border};
        ">{status}</span>
      </div>
      <div style="font-size:0.76rem;color:#9A9080;margin-bottom:0.55rem;">
        🕐 Last check: {last_check}
      </div>
      <div style="margin-bottom:0.55rem;">
        <span style="
            font-size:0.68rem;
            font-weight:700;
            letter-spacing:0.08em;
            text-transform:uppercase;
            color:#8A8078;
            margin-right:0.4rem;
        ">Stress trend</span>
        <span style="font-size:0.82rem;font-weight:700;color:{trend_color};">{trend}</span>
      </div>
      <div style="font-size:0.84rem;color:#3A3028;line-height:1.45;margin-bottom:0.75rem;">{note}</div>
      <div style="
          border:1.5px solid {border};
          border-radius:10px;
          padding:0.45rem 0.75rem;
          text-align:center;
          font-size:0.82rem;
          font-weight:600;
          color:{badge_color};
          cursor:default;
      ">View details &rsaquo;</div>
    </div>"""


# ─────────────────────────────────────────────────────────────────
# APP HEADER
# ─────────────────────────────────────────────────────────────────

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

logo_col, title_col, icon_col = st.columns([1, 5, 1], gap="small")
with logo_col:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=52)
    else:
        st.markdown(
            "<div style='width:48px;height:48px;border-radius:12px;background:#FEF3C7;"
            "display:flex;align-items:center;justify-content:center;font-size:1.5rem;"
            "border:1.5px solid #F5B942;'>🐝</div>",
            unsafe_allow_html=True,
        )
with title_col:
    st.markdown(
        "<div style='font-size:0.65rem;font-weight:700;letter-spacing:0.12em;"
        "text-transform:uppercase;color:#9A9080;margin-bottom:1px;'>Good morning, beekeeper</div>"
        "<div style='font-size:1.15rem;font-weight:800;color:#1C1C1E;letter-spacing:-0.02em;'>Sonabee</div>",
        unsafe_allow_html=True,
    )
with icon_col:
    st.markdown(
        "<div style='display:flex;gap:0.6rem;justify-content:flex-end;padding-top:6px;'>"
        "<span style='font-size:1.2rem;cursor:default;'>🔔</span>"
        "<span style='font-size:1.2rem;cursor:default;'>⚙️</span></div>",
        unsafe_allow_html=True,
    )

st.markdown(
    "<hr style='border:none;border-top:1.5px solid #EDE8DF;margin:0.6rem 0 0.2rem;'>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────
# MODEL + SESSION STATE
# ─────────────────────────────────────────────────────────────────

model = load_model()
if model is None:
    st.warning("⚠️ No trained model found in `models/`. Run `python train_model.py` to generate one.")

for key in ("last_waveform", "last_sr"):
    if key not in st.session_state:
        st.session_state[key] = None

# ─────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────

tab_analyze, tab_dashboard, tab_specs, tab_how, tab_model_card = st.tabs(
    ["🎙 Analyze", "🏡 My Hives", "📊 Spectrogram", "ℹ️ How it works", "📋 Model card"]
)


# ══════════════════════════════════════════════════════════════════
# TAB 1 — ANALYZE
# ══════════════════════════════════════════════════════════════════
with tab_analyze:
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── 1. Upload section ─────────────────────────────────────────
    with st.container(border=True):
        st.markdown(
            "<div style='font-size:1rem;font-weight:700;color:#1C1C1E;margin-bottom:0.2rem;'>"
            "1. Upload a hive recording</div>"
            "<div style='font-size:0.83rem;color:#7A7068;margin-bottom:0.75rem;'>"
            "For best results, use a 30–60 second recording from inside the hive or at the entrance.</div>",
            unsafe_allow_html=True,
        )

        # Uploader — styled to dashed box via CSS above
        uploaded = st.file_uploader(
            "🎧  Drop a .wav file here or click to browse",
            type=["wav"],
            label_visibility="visible",
        )

        if uploaded is not None:
            st.audio(uploaded.getvalue(), format="audio/wav")
            rms_preview = preview_rms_from_bytes(uploaded.getvalue())
            if rms_preview < 0.01:
                sig_label = "📶 Very quiet — check mic placement"
            elif rms_preview > 0.06:
                sig_label = "📶 Strong signal — good buzz captured"
            else:
                sig_label = "📶 Good signal — typical colony buzz"
            st.caption(sig_label)

    # ── 2. Temperature section ────────────────────────────────────
    with st.container(border=True):
        st.markdown(
            "<div style='font-size:1rem;font-weight:700;color:#1C1C1E;margin-bottom:0.2rem;'>"
            "2. Provide ambient temperature</div>"
            "<div style='font-size:0.83rem;color:#7A7068;margin-bottom:0.85rem;'>"
            "Hive behavior shifts with environmental conditions, so Sonabee uses ambient "
            "temperature as additional context.</div>",
            unsafe_allow_html=True,
        )

        temp_mode = st.radio(
            "temp_mode",
            ["Manual entry", "Look up by location"],
            horizontal=True,
            label_visibility="collapsed",
        )

        temp_c = 25.0

        if temp_mode == "Manual entry":
            temp_c = st.slider(
                "Temperature (°C)",
                min_value=-5.0, max_value=45.0,
                value=25.0, step=0.5,
                label_visibility="collapsed",
                format="%.0f°C",
            )
        else:
            c1, c2 = st.columns([3, 1])
            city = c1.text_input("City", value="Berkeley",
                                 label_visibility="collapsed", placeholder="City")
            country = c2.text_input("CC", value="US",
                                    label_visibility="collapsed", placeholder="CC")
            if st.button("Fetch weather"):
                fetched = get_openweather_temp(city, country)
                if fetched is None:
                    st.warning("Couldn't fetch weather — check `OPENWEATHER_API_KEY` or use manual entry.")
                else:
                    temp_c = fetched
                    st.success(f"📍 {city}: **{fetched:.1f} °C**")

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── Analyze button ────────────────────────────────────────────
    analyze_btn = st.button("Analyze hive", type="primary", use_container_width=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Results ───────────────────────────────────────────────────
    if not uploaded:
        st.markdown(
            "<div style='text-align:center;padding:2rem 1rem;"
            "background:#FFFFFF;border-radius:18px;border:1.5px solid #EDE8DF;"
            "color:#B0A898;font-size:0.88rem;box-shadow:0 2px 12px rgba(0,0,0,0.04);'>"
            "🐝 Upload a recording above, then tap <strong>Analyze hive</strong>.</div>",
            unsafe_allow_html=True,
        )
    elif model is None:
        st.error("No trained model found. Run `python train_model.py` first.")

    elif analyze_btn:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name

            with st.spinner("Analyzing hive audio…"):
                audio_np, sr = sf.read(tmp_path)
                if audio_np.ndim > 1:
                    audio_np = audio_np.mean(axis=1)
                waveform = torch.from_numpy(audio_np).float().unsqueeze(0)
                st.session_state["last_waveform"] = waveform
                st.session_state["last_sr"] = sr

                fv = prepare_feature_vector(tmp_path, temp_c, model)
                if fv is None:
                    st.stop()

                proba = model.predict_proba(fv)[0]
                if proba.shape[0] == 2:
                    p_healthy, p_risk = float(proba[0]), float(proba[1])
                    label = int(np.argmax(proba))
                else:
                    p_risk = float(proba[0])
                    p_healthy = 1.0 - p_risk
                    label = int(p_risk >= 0.5)

                acoustics = analyze_acoustics(waveform, sr)
                status, _emoji, why, actions = interpret_prediction(label, p_risk, acoustics, temp_c)
                rms = acoustics["rms"]
                centroid = acoustics["centroid"]

            # ── Overall status row ──
            if label == 0:
                dot_color = "#22C55E"
                status_color = "#15803D"
                status_bg = "#F0FDF4"
                status_border = "#BBF7D0"
            elif p_risk > 0.9:
                dot_color = "#EF4444"
                status_color = "#991B1B"
                status_bg = "#FEF2F2"
                status_border = "#FECACA"
            else:
                dot_color = "#F59E0B"
                status_color = "#92400E"
                status_bg = "#FFFBEB"
                status_border = "#FDE68A"

            st.markdown(
                f"""
                <div style="
                    display:flex;justify-content:space-between;align-items:center;
                    background:{status_bg};
                    border:1.5px solid {status_border};
                    border-radius:14px;
                    padding:0.75rem 1.1rem;
                    margin-bottom:0.75rem;
                ">
                  <div style="
                      font-size:0.68rem;font-weight:700;
                      letter-spacing:0.12em;text-transform:uppercase;
                      color:#8A8078;
                  ">Overall status</div>
                  <div style="
                      display:flex;align-items:center;gap:0.4rem;
                      font-size:0.88rem;font-weight:700;color:{status_color};
                  ">
                    <span style="
                        width:8px;height:8px;border-radius:50%;
                        background:{dot_color};display:inline-block;
                    "></span>
                    {status}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Buzz intensity ──
            if rms < 0.01:
                buzz_val, buzz_sub, buzz_detail = "Low", f"RMS {rms:.2f}", "Quiet amplitude — low activity or distant microphone."
            elif rms > 0.06:
                buzz_val, buzz_sub, buzz_detail = "Elevated", f"RMS {rms:.2f}", "High amplitude — possible agitation or heavy forager activity."
            else:
                buzz_val, buzz_sub, buzz_detail = "Moderate", f"RMS {rms:.2f}", "Typical amplitude for a stable worker population."

            # ── Dominant frequency ──
            if centroid < 220:
                freq_val, freq_sub, freq_detail = "Low band", f"{int(centroid)} Hz", "Energy concentrated below 220 Hz — calm or clustered bees."
            elif centroid <= 340:
                freq_val, freq_sub, freq_detail = "Mid band", f"{int(centroid)} Hz", "Falls within the typical foraging and brood-care range."
            else:
                freq_val, freq_sub, freq_detail = "High band", f"{int(centroid)} Hz", "More energy above 340 Hz — often ventilation bursts or agitation."

            # ── Context ──
            if temp_c < 15:
                ctx_val, ctx_sub, ctx_detail = "Cold", f"{temp_c:.0f}°C", "Below-normal ambient temp — clustering and reduced flight expected."
            elif temp_c > 35:
                ctx_val, ctx_sub, ctx_detail = "Hot", f"{temp_c:.0f}°C", "High ambient temp — fanning and ventilation noise expected."
            else:
                ctx_val, ctx_sub, ctx_detail = "Neutral", f"{temp_c:.0f}°C", "Weather & temperature sit in a typical working band for the colony."

            # ── Three metric cards (distinct pastel backgrounds) ──
            st.markdown(
                _metric_card("Buzz intensity",   buzz_val, buzz_sub, buzz_detail, "#FDF6EC", "#C8860A") +
                _metric_card("Dominant frequency", freq_val, freq_sub, freq_detail, "#EDF6F5", "#0E7490") +
                _metric_card("Context",            ctx_val, ctx_sub,  ctx_detail,  "#F0EDF8", "#6D28D9"),
                unsafe_allow_html=True,
            )

            # ── Amber summary sentence ──
            if label == 0:
                summary = (
                    f"Buzz intensity, dominant frequency, and spectral variability all sit "
                    f"in the typical band for a stable hive at {temp_c:.0f}°."
                )
            else:
                summary = (
                    f"One or more acoustic indicators fall outside the expected range "
                    f"for a healthy hive at {temp_c:.0f}°. Consider an inspection."
                )
            st.markdown(
                f"<div style='text-align:center;padding:0.9rem 1.2rem;"
                f"font-size:0.88rem;font-weight:500;color:#C8860A;line-height:1.55;'>{summary}</div>",
                unsafe_allow_html=True,
            )

            # ── Confidence + expandable details ──
            st.caption(f"Confidence — Healthy: {p_healthy*100:.1f}%  ·  At-risk: {p_risk*100:.1f}%")

            with st.expander("🔎 Why this assessment", expanded=True):
                for pt in why:
                    st.markdown(f"- {pt}")

            with st.expander("✅ Recommended actions", expanded=(label != 0)):
                for act in actions:
                    st.markdown(f"- {act}")

            st.caption(
                "⚠️ Sonabee is experimental and is **not** a substitute for regular hive inspections "
                "or professional apiary advice."
            )

        finally:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass


# ══════════════════════════════════════════════════════════════════
# TAB 2 — MY HIVES DASHBOARD
# ══════════════════════════════════════════════════════════════════
with tab_dashboard:
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # Header row
    h1, h2 = st.columns([4, 1])
    with h1:
        st.markdown(
            "<div style='font-size:1.6rem;font-weight:800;color:#1C1C1E;"
            "letter-spacing:-0.03em;margin-bottom:0.15rem;'>My Hives</div>"
            "<div style='font-size:0.84rem;color:#7A7068;'>Monitor acoustic health across all your colonies</div>",
            unsafe_allow_html=True,
        )
    with h2:
        st.markdown(
            "<div style='display:flex;justify-content:flex-end;padding-top:6px;'>"
            "<div style='width:36px;height:36px;background:#F5B942;border-radius:50%;"
            "display:flex;align-items:center;justify-content:center;"
            "font-size:1.2rem;color:#fff;box-shadow:0 2px 8px rgba(245,185,66,0.45);'>"
            "+</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Hive data
    hives = [
        {
            "name": "Hive 1 — North Yard",
            "status": "Healthy",
            "last_check": "2h ago",
            "trend": "Low",
            "note": "Low acoustic stress; typical foraging pattern.",
            "bg": "#F0FDF4", "border": "#BBF7D0",
            "badge_bg": "#D1FAE5", "badge_color": "#15803D",
            "trend_color": "#15803D",
        },
        {
            "name": "Hive 2 — Driveway",
            "status": "Watch",
            "last_check": "47 min ago",
            "trend": "Moderate",
            "note": "Slightly elevated buzz amplitude; recheck soon.",
            "bg": "#FFFBEB", "border": "#FDE68A",
            "badge_bg": "#FEF3C7", "badge_color": "#B45309",
            "trend_color": "#D97706",
        },
        {
            "name": "Hive 3 — Orchard",
            "status": "At risk",
            "last_check": "18 min ago",
            "trend": "High",
            "note": "Strong, erratic buzz pattern; possible disturbance or queen issues.",
            "bg": "#FEF2F2", "border": "#FECACA",
            "badge_bg": "#FEE2E2", "badge_color": "#991B1B",
            "trend_color": "#DC2626",
        },
        {
            "name": "Hive 4 — New split",
            "status": "Awaiting baseline",
            "last_check": "—",
            "trend": "N/A",
            "note": "Awaiting first recording to establish acoustic baseline.",
            "bg": "#F9F9F9", "border": "#E5E7EB",
            "badge_bg": "#F3F4F6", "badge_color": "#6B7280",
            "trend_color": "#6B7280",
        },
    ]

    for h in hives:
        st.markdown(
            _hive_card(
                name=h["name"], status=h["status"],
                last_check=h["last_check"], trend=h["trend"], note=h["note"],
                bg=h["bg"], border=h["border"],
                badge_bg=h["badge_bg"], badge_color=h["badge_color"],
                trend_color=h["trend_color"],
            ),
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════
# TAB 3 — SPECTROGRAM EXPLORER
# ══════════════════════════════════════════════════════════════════
with tab_specs:
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:1rem;font-weight:700;color:#1C1C1E;margin-bottom:0.1rem;'>"
        "Spectrogram explorer</div>",
        unsafe_allow_html=True,
    )

    if st.session_state["last_waveform"] is None:
        st.info("Analyze a recording in the **🎙 Analyze** tab to visualize its spectrogram here.")
    else:
        wf = st.session_state["last_waveform"]
        sr = st.session_state["last_sr"]

        view = st.radio("View", ["Mel spectrogram", "MFCCs"], horizontal=True)
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        if view == "Mel spectrogram":
            st.pyplot(plot_spectrogram(wf, sr))
            st.caption(
                "Brighter bands = more energy. Higher energy around 2–4 kHz "
                "can indicate agitation or strong ventilation."
            )
        else:
            st.pyplot(plot_mfcc(wf, sr))
            st.caption(
                "Stable lower-order coefficients → steady brood buzz. "
                "Rapid changes across time → disturbance or environmental shifts."
            )


# ══════════════════════════════════════════════════════════════════
# TAB 4 — HOW IT WORKS
# ══════════════════════════════════════════════════════════════════
with tab_how:
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    st.markdown("#### How Sonabee works")
    st.markdown("""
Sonabee is a **software-first prototype** inspired by commercial hive monitors and recent
research in honeybee acoustics.

**Pipeline overview**
1. **Audio capture** — 30–60 second recording at or just inside the hive entrance
2. **Pre-processing** — normalize and convert to mel-frequency features
3. **Feature extraction** — 13 MFCC means + stds, RMS energy, amplitude std
4. **Context fusion** — ambient temperature appended as a contextual input
5. **Classification** — lightweight Random Forest outputs health / stress estimate
6. **Interpretation** — heuristics produce human-readable explanations

**Why acoustics?**
- Bees communicate via vibrations and sound (wing beats, piping, fanning)
- Acoustic signatures shift measurably with queen presence, swarming, parasites, and thermal stress
- Sound is non-invasive — monitor the colony without opening the hive
""")


# ══════════════════════════════════════════════════════════════════
# TAB 5 — MODEL CARD
# ══════════════════════════════════════════════════════════════════
with tab_model_card:
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    st.markdown("#### Model card")
    st.markdown("""
**Intended use**
- Early screening tool for potential hive stress
- Educational demo of bioacoustics + ML for bee health
- Not a replacement for in-person inspection or professional apiary advice

**Inputs**
- 30–60 second `.wav` hive recording
- Ambient temperature (°C) — manual entry or OpenWeather API

**Model**
- `StandardScaler` → `RandomForestClassifier` (300 trees, class-balanced)
- 29 features: 13 MFCC means + 13 MFCC stds + RMS + amplitude std + temperature
- Output: binary label (healthy / at-risk) + confidence probabilities

**Limitations**
- Trained on one dataset of hives — may degrade on very different environments
- Sensitive to loud background noise (speech, traffic, wind)
- Can misinterpret edge cases: queen introduction, small colonies, heavy manipulation

**Safety**
- Never use Sonabee as the sole basis for a management decision
- Use it alongside visual inspection, hive weight, mite counts, and beekeeper expertise
""")
