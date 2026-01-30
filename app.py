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


# -----------------------------
# SIMPLE AUDIO PREVIEW METRIC
# -----------------------------
def preview_rms_from_bytes(audio_bytes: bytes) -> float:
    """Quick RMS estimate from uploaded audio bytes."""
    try:
        buf = io.BytesIO(audio_bytes)
        audio_np, _ = sf.read(buf)
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        rms = float(np.sqrt(np.mean(audio_np ** 2)))
        return rms
    except Exception:
        return 0.0


# -----------------------------
# WEATHER + MODEL HELPERS
# -----------------------------
def get_openweather_temp(city: str, country: str | None = None) -> float | None:
    """Fetch current temp in °C via OpenWeather. Returns None on failure."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return None

    q = city if not country else f"{city},{country}"
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": q, "appid": api_key, "units": "metric"}

    try:
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return float(data["main"]["temp"])
    except Exception:
        return None


@st.cache_resource
def load_model():
    """Try several common model filenames, return first found."""
    candidate_paths = [
        "models/sonabee_rf.pkl",
        "models/model.pkl",
        "models/bee_model.pkl",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception:
                continue
    return None


def prepare_feature_vector(audio_path: str, temp_c: float, model) -> np.ndarray | None:
    """Extract audio features, merge with temperature if needed, fit to model input size."""
    try:
        feats = extract_features(audio_path)
        feats = np.asarray(feats).flatten()
    except Exception as e:
        st.error(f"Error extracting audio features: {e}")
        return None

    model_dim = getattr(model, "n_features_in_", None)

    # If we don't know model feature size, default to audio+temp
    if model_dim is None:
        fv = np.append(feats, temp_c)
        return fv.reshape(1, -1)

    # Decide whether temperature was used in training based on expected dimensionality
    if model_dim == len(feats) + 1:
        fv = np.append(feats, temp_c)
    elif model_dim == len(feats):
        fv = feats
    else:
        # Fallback: pad or truncate to fit
        if len(feats) < model_dim:
            pad_len = model_dim - len(feats)
            fv = np.pad(feats, (0, pad_len))
        else:
            fv = feats[:model_dim]

    return fv.reshape(1, -1)


# -----------------------------
# ACOUSTIC ANALYSIS + PLOTS
# -----------------------------
def analyze_acoustics(waveform: torch.Tensor, sr: int) -> dict:
    """Compute simple interpretable acoustic indicators for UI text."""
    # mono
    if waveform.ndim > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # RMS energy
    rms = torch.sqrt(torch.mean(waveform ** 2)).item()

    # Spectral centroid
    spec = torch.fft.rfft(waveform)
    mag = torch.abs(spec).squeeze(0)
    freqs = torch.fft.rfftfreq(waveform.shape[-1], d=1.0 / sr)
    centroid = (freqs * mag).sum() / (mag.sum() + 1e-9)
    centroid = centroid.item()

    # Simple heuristics for messaging
    indicators = []
    if rms < 0.01:
        indicators.append("Low overall hive activity (quiet buzz amplitude).")
    elif rms > 0.06:
        indicators.append("Elevated buzz intensity, suggesting agitation or heavy activity.")
    else:
        indicators.append("Moderate buzz level consistent with typical colony activity.")

    if 200 <= centroid <= 320:
        indicators.append("Dominant frequency band within expected brood/worker buzz range.")
    elif centroid > 320:
        indicators.append(
            "Higher-than-usual dominant frequency; may reflect agitation, ventilation, or disturbed bees."
        )
    else:
        indicators.append(
            "Lower dominant frequency; could indicate reduced activity or cold, clustered bees."
        )

    return {
        "rms": rms,
        "centroid": centroid,
        "messages": indicators,
    }


def plot_spectrogram(waveform: torch.Tensor, sr: int):
    if waveform.ndim > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=2048, hop_length=512, n_mels=64
    )(waveform)
    mel_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)

    fig, ax = plt.subplots(figsize=(7, 3))
    im = ax.imshow(
        mel_db.squeeze(0).numpy(),
        origin="lower",
        aspect="auto",
        extent=[0, waveform.shape[-1] / sr, 0, sr / 2],
        cmap="magma",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel band")
    ax.set_title("Mel Spectrogram")
    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    return fig


def plot_mfcc(waveform: torch.Tensor, sr: int):
    if waveform.ndim > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sr, n_mfcc=13, melkwargs={"n_fft": 2048, "hop_length": 512}
    )
    mfcc = mfcc_transform(waveform).squeeze(0).numpy()

    fig, ax = plt.subplots(figsize=(7, 3))
    im = ax.imshow(
        mfcc,
        origin="lower",
        aspect="auto",
        extent=[0, waveform.shape[-1] / sr, 1, mfcc.shape[0]],
        cmap="viridis",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MFCC Coefficient")
    ax.set_title("MFCCs Over Time")
    fig.colorbar(im, ax=ax, label="Coefficient value")
    fig.tight_layout()
    return fig


def interpret_prediction(label: int, prob_risk: float, acoustics: dict, temp_c: float):
    """
    Return status headline text, bullet points (why), and recommended actions.

    NOTE: returns exactly *three* things so you can safely do:
        headline, why, actions = interpret_prediction(...)
    """
    if label == 0:
        status_text = "Healthy"
        status_emoji = "🟢"
    else:
        if prob_risk > 0.9:
            status_text = "High Stress / At-Risk"
            status_emoji = "🔴"
        else:
            status_text = "Watch"
            status_emoji = "🟡"

    headline = f"{status_text} {status_emoji}"

    why = []
    why.extend(acoustics.get("messages", []))

    # Add simple weather rule-of-thumb
    if temp_c < 15:
        why.append("Ambient temperature is low; bees may be clustering to maintain brood warmth.")
    elif temp_c > 35:
        why.append("Ambient temperature is high; increased ventilation/fanning may change acoustic profile.")

    actions = []
    if label == 0:
        actions.append("Continue regular inspections every 1–2 weeks.")
        actions.append("Monitor for seasonal changes in activity and weight.")
        actions.append("Use Sonabee periodically to establish a healthy acoustic baseline.")
    else:
        actions.append("Inspect for queen presence and brood pattern within the next 24–48 hours.")
        actions.append("Check for signs of Varroa or other parasites.")
        actions.append("Verify food stores and recent weather stress (heat waves, storms, cold snaps).")
        actions.append("Compare this recording with past audio from the same hive, if available.")

    return headline, why, actions


# -----------------------------
# PAGE CONFIG & GLOBAL STYLE
# -----------------------------
st.set_page_config(
    page_title="Sonabee – Acoustic Hive Intelligence",
    page_icon="🐝",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFFDF7 !important;
        font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
        background-image:
            linear-gradient(30deg, rgba(251,211,141,0.06) 12%, transparent 12.5%, transparent 87%, rgba(251,211,141,0.06) 87.5%, rgba(251,211,141,0.06)),
            linear-gradient(150deg, rgba(251,211,141,0.06) 12%, transparent 12.5%, transparent 87%, rgba(251,211,141,0.06) 87.5%, rgba(251,211,141,0.06)),
            linear-gradient(90deg, rgba(251,211,141,0.02) 2%, transparent 0);
        background-size: 40px 70px;
        background-position: 0 0, 0 0, 0 0;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .sonabee-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #102A43;
        letter-spacing: -0.03em;
    }
    .sonabee-subtitle {
        font-size: 1.0rem;
        margin-top: 0.2rem;
        color: #62727b;
        max-width: 36rem;
    }
    .sonabee-main {
        background-color: #FFFFFF;
        max-width: 430px;
        margin: 0.25rem auto 2.5rem;
        padding: 1.6rem 1.4rem 1.9rem;
        border-radius: 32px;
        border: 1px solid #E5E9F0;
        box-shadow: 0 0 40px rgba(15, 23, 42, 0.14);
    }

    .status-pill {
        padding: 4px 11px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }
    .status-healthy { background-color: #D8F3DC; color: #2F855A; }
    .status-warning { background-color: #FFF3C4; color: #946200; }
    .status-risk    { background-color: #FED7D7; color: #9B1C1C; }
    .status-unknown { background-color: #E5E7EB; color: #374151; }

    /* ---------- HEALTH DONUT RING ---------- */
    .health-ring-wrapper {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.35rem;
    }
    .health-ring-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: #6B7280;
    }
    .health-ring {
        width: 120px;
        height: 120px;
        border-radius: 999px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 0 6px rgba(251, 191, 36, 0.15),
                    0 22px 40px rgba(15, 23, 42, 0.35);
        background: conic-gradient(#FDBA4D 0 300deg, #FFF7E5 300deg 360deg);
    }
    .health-ring-inner {
        width: 76px;
        height: 76px;
        border-radius: 999px;
        background: #FFFDF7;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.9rem;
        font-weight: 600;
        color: #92400E;
    }
    .health-ring--risk {
        background: conic-gradient(#F97373 0 300deg, #FFE5E5 300deg 360deg);
        box-shadow: 0 0 0 6px rgba(248, 113, 113, 0.15),
                    0 22px 40px rgba(127, 29, 29, 0.35);
    }
    .health-ring-inner--risk { color: #7F1D1D; }
    .health-ring--watch {
        background: conic-gradient(#FBBF24 0 300deg, #FFF7D6 300deg 360deg);
        box-shadow: 0 0 0 6px rgba(251, 191, 36, 0.15),
                    0 22px 40px rgba(120, 53, 15, 0.35);
    }
    .health-ring-inner--watch { color: #92400E; }

    /* ---------- ASSESSMENT CARD ---------- */
    .assessment-card {
        width: 100%;
        background: linear-gradient(145deg,#ECFDF3,#F5F5FF);
        border-radius: 1.4rem;
        padding: 1.1rem 1.1rem 1.2rem;
        border: 1px solid rgba(148,163,184,0.25);
        box-shadow: 0 18px 40px rgba(15,23,42,0.18);
        font-size: 0.82rem;
        color: #111827;
    }
    .assessment-healthy { background: linear-gradient(145deg,#ECFDF3,#EFF6FF); }
    .assessment-watch   { background: linear-gradient(145deg,#FEF9C3,#FFFBEB); }
    .assessment-risk    { background: linear-gradient(145deg,#FEE2E2,#FEF2F2); }

    .assessment-header {
        display:flex;
        align-items:center;
        gap:0.7rem;
        margin-bottom:0.75rem;
        flex-wrap:wrap;
    }
    .assessment-summary {
        font-size:0.9rem;
        font-weight:600;
        color:#111827;
    }
    .assessment-metrics {
        display:grid;
        grid-template-columns:repeat(3,minmax(0,1fr));
        gap:0.5rem;
        margin-bottom:0.8rem;
    }
    .metric-chip {
        background-color: rgba(255,255,255,0.85);
        border-radius:0.7rem;
        border:1px solid rgba(148,163,184,0.5);
        padding:0.35rem 0.55rem;
    }
    .metric-chip span {
        text-transform:uppercase;
        letter-spacing:0.08em;
        font-size:0.65rem;
        color:#6B7280;
    }
    .metric-chip strong {
        display:block;
        font-size:0.8rem;
        color:#111827;
    }
    .metric-chip div {
        font-size:0.72rem;
        color:#4B5563;
    }

    .assessment-probs {
        font-size:0.78rem;
        color:#374151;
        margin-bottom:0.8rem;
    }
    .assessment-section-title {
        margin-top:0.6rem;
        margin-bottom:0.2rem;
        font-size:0.78rem;
        text-transform:uppercase;
        letter-spacing:0.08em;
        color:#4B5563;
    }
    .assessment-footnote {
        margin-top:0.6rem;
        font-size:0.72rem;
        color:#4B5563;
    }

    .pulse-once {
        animation:pulse-once 1.6s ease-out 1;
    }
    @keyframes pulse-once {
        0%   { transform:scale(0.9); box-shadow:0 0 0 0 rgba(59,130,246,0.4); }
        70%  { transform:scale(1.03); box-shadow:0 0 0 10px rgba(59,130,246,0); }
        100% { transform:scale(1); box-shadow:none; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# HEADER
# -----------------------------
logo_col, title_col = st.columns([0.5, 3.5], gap="small")

with logo_col:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=150)
    else:
        st.markdown("### 🐝")

with title_col:
    st.markdown('<div class="sonabee-title">Sonabee</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sonabee-subtitle">Acoustic hive intelligence for early detection of colony stress. '
        'Upload a beehive recording and Sonabee will analyze its soundscape using bioacoustic features and machine learning.</div>',
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-top:-25px'></div>", unsafe_allow_html=True)
st.markdown('<div class="sonabee-main">', unsafe_allow_html=True)

tab_analyze, tab_dashboard, tab_specs, tab_how, tab_model = st.tabs(
    ["Analyze Recording", "Hive Dashboard", "Spectrogram Explorer", "How Sonabee Works", "Model Card"]
)

model = load_model()
if model is None:
    st.error(
        "No trained model found in the `models/` directory. "
        "Please train and save a model before using the app."
    )

if "last_waveform" not in st.session_state:
    st.session_state["last_waveform"] = None
    st.session_state["last_sr"] = None


# -----------------------------
# TAB 1: ANALYZE
# -----------------------------
with tab_analyze:
    uploaded = None
    temp_c = None
    weather_note = None

    st.markdown("#### 1. Upload a hive recording")
    st.markdown(
        "For best results, use a **30–60 second** recording from inside or just at the hive entrance, "
        "minimizing speech and wind."
    )
    uploaded = st.file_uploader("Upload a .wav file", type=["wav"])

    audio_bytes = None
    preview_rms = None

    if uploaded is not None:
        audio_bytes = uploaded.getvalue()
        st.audio(audio_bytes, format="audio/wav")
        preview_rms = preview_rms_from_bytes(audio_bytes)

        if preview_rms is not None:
            if preview_rms < 0.01:
                sig_text = "Very quiet signal – low hive activity or distant mic."
            elif preview_rms > 0.06:
                sig_text = "Strong signal – high buzz amplitude captured."
            else:
                sig_text = "Good signal level – typical colony buzz captured."
            st.caption(f"Signal preview: {sig_text}")

    st.markdown("#### 2. Provide ambient temperature")
    temp_mode = st.radio(
        "How would you like to provide temperature?",
        ["Manual entry", "Look up by location"],
        horizontal=True,
    )

    if temp_mode == "Manual entry":
        temp_c = st.slider("Ambient temperature (°C)", min_value=-5.0, max_value=45.0, value=25.0)
    else:
        city = st.text_input("City", value="Berkeley")
        country = st.text_input("Country code (optional, e.g., US)", value="US")
        if st.button("Fetch weather"):
            t = get_openweather_temp(city, country)
            if t is None:
                st.warning(
                    "Couldn't fetch weather automatically. Check your API key or network, or switch to manual entry."
                )
            else:
                temp_c = t
                weather_note = f"Current weather in {city}: {t:.1f} °C"

    if temp_c is None:
        temp_c = 25.0

    if weather_note:
        st.info(weather_note)

    analyze_btn = st.button("Analyze hive", type="primary", use_container_width=True)

    # --- 3. Hive health assessment ---
    st.markdown("#### 3. Hive health assessment")

    if not uploaded:
        st.info(
            "Upload a recording and set the ambient temperature above, then click **Analyze hive** "
            "to see an acoustic health assessment."
        )
    elif model is None:
        st.error(
            "No trained model found in the `models/` directory. "
            "Please train and save a model before using the analysis."
        )
    elif analyze_btn:
        try:
            # Save audio to temp file for feature extraction
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name

            # Load audio (for acoustics + spectrogram tab)
            audio_np, sr = sf.read(tmp_path)
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)
            waveform = torch.from_numpy(audio_np).float().unsqueeze(0)
            st.session_state["last_waveform"] = waveform
            st.session_state["last_sr"] = sr

            # Build feature vector
            fv = prepare_feature_vector(tmp_path, temp_c, model)
            if fv is None:
                st.stop()

            # Model prediction
            proba = model.predict_proba(fv)[0]
            if proba.shape[0] == 2:
                p_healthy = float(proba[0])
                p_risk = float(proba[1])
                label = int(np.argmax(proba))
            else:
                p_risk = float(proba[0])
                p_healthy = 1.0 - p_risk
                label = int(p_risk >= 0.5)

            # Acoustic heuristics
            acoustics = analyze_acoustics(waveform, sr)
            headline, why, actions = interpret_prediction(label, p_risk, acoustics, temp_c)

            rms = acoustics["rms"]
            centroid = acoustics["centroid"]

            # ----- Buzz intensity chip -----
            if rms < 0.01:
                buzz_label = "Low"
                buzz_desc = "Lower-than-average amplitude – quiet colony or low activity."
            elif rms > 0.06:
                buzz_label = "Elevated"
                buzz_desc = "Higher-than-average amplitude – strong buzz, possible agitation."
            else:
                buzz_label = "Moderate"
                buzz_desc = "Typical amplitude for a stable worker population."

            # ----- Dominant frequency chip -----
            if centroid < 220:
                freq_label = "Low band"
                freq_desc = "Energy concentrated below ~220 Hz – calmer, clustered bees."
            elif centroid <= 340:
                freq_label = "Mid band"
                freq_desc = "Dominant energy 220–340 Hz – typical worker/brood buzz."
            else:
                freq_label = "High band"
                freq_desc = "More energy above ~340 Hz – often ventilation bursts or agitation."

            # ----- Context chip -----
            if temp_c < 15:
                ctx_label = "Cold"
                ctx_desc = "Below-normal ambient temp; clustering and reduced flight expected."
            elif temp_c > 35:
                ctx_label = "Hot"
                ctx_desc = "High ambient temp; fanning/ventilation noise expected."
            else:
                ctx_label = "Neutral"
                ctx_desc = "Weather & temperature sit in a typical working band for the colony."

            # Map to pill + donut state
            if label == 0:
                status_text = "Healthy"
                status_class = "status-healthy"
                gradient_class = "assessment-healthy"
                ring_state_class = "health-ring"
                ring_inner_class = "health-ring-inner"
            elif p_risk < 0.9:
                status_text = "Watch"
                status_class = "status-warning"
                gradient_class = "assessment-watch"
                ring_state_class = "health-ring health-ring--watch"
                ring_inner_class = "health-ring-inner health-ring-inner--watch"
            else:
                status_text = "At-Risk"
                status_class = "status-risk"
                gradient_class = "assessment-risk"
                ring_state_class = "health-ring health-ring--risk"
                ring_inner_class = "health-ring-inner health-ring-inner--risk"

            summary = (
                "Acoustic pattern consistent with a healthy, active colony."
                if label == 0
                else "Acoustic pattern shows signs of emerging or elevated stress."
            )

            why_items = "".join(f"<li>{w}</li>" for w in why)
            action_items = "".join(f"<li>{a}</li>" for a in actions)

            # ----- Nice card + orange donut ring -----
            card_html = f"""
<div class="assessment-card {gradient_class}">
  <div style="display:flex;flex-direction:row;gap:1.2rem;align-items:flex-start;">
    <!-- Left: text -->
    <div style="flex:1;min-width:0;">
      <div class="assessment-header">
        <span class="status-pill {status_class} pulse-once">
          {status_text}
        </span>
        <div class="assessment-summary">{summary}</div>
      </div>

      <div class="assessment-metrics">
        <div class="metric-chip">
          <span>Buzz intensity</span>
          <strong>{buzz_label}</strong>
          <div>{buzz_desc}</div>
        </div>
        <div class="metric-chip">
          <span>Dominant frequency</span>
          <strong>{int(centroid)} Hz</strong>
          <div>{freq_desc}</div>
        </div>
        <div class="metric-chip">
          <span>Context</span>
          <strong>{ctx_label}</strong>
          <div>{ctx_desc}</div>
        </div>
      </div>

      <div class="assessment-probs">
        <b>Estimated probabilities</b><br/>
        · Healthy: {p_healthy*100:.1f}%<br/>
        · Stressed / at-risk: {p_risk*100:.1f}%
      </div>

      <div class="assessment-section-title">Why this assessment</div>
      <ul>
        {why_items}
      </ul>

      <div class="assessment-section-title">Recommended actions</div>
      <ul>
        {action_items}
      </ul>

      <div class="assessment-footnote">
        &#9888;&#65039; Sonabee is an experimental tool and is <b>not</b> a substitute for regular hive
        inspections or professional advice. Use it as an additional signal, not as a sole decision-maker.
      </div>
    </div>

    <!-- Right: donut ring -->
    <div class="health-ring-wrapper">
      <div class="health-ring-label">Analysis result</div>
      <div class="{ring_state_class}">
        <div class="{ring_inner_class}">
          {status_text}
        </div>
      </div>
    </div>
  </div>
</div>
"""
            st.markdown(card_html, unsafe_allow_html=True)

        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# -----------------------------
# TAB 2: HIVE DASHBOARD (simple mock)
# -----------------------------
with tab_dashboard:
    st.markdown("#### Hive dashboard (concept)")
    st.caption(
        "Imagine Sonabee connected to multiple hives. This dashboard mocks what a multi-hive view could look like."
    )

    st.button("➕ Add hive", help="Prototype only – this would register a new hive profile.")

    col1, col2 = st.columns(2, gap="medium")

    hive_cards = [
        {
            "name": "Hive 1 – North Yard",
            "status": "Healthy",
            "status_class": "status-healthy",
            "emoji": "🟢",
            "last_check": "2 hours ago",
            "extra": "Stable brood buzz and consistent centroid over last 7 checks.",
        },
        {
            "name": "Hive 2 – Orchard",
            "status": "Watch",
            "status_class": "status-warning",
            "emoji": "🟡",
            "last_check": "30 minutes ago",
            "extra": "Slight upward drift in stress score after recent heat wave.",
        },
        {
            "name": "Hive 3 – Rooftop",
            "status": "At-Risk",
            "status_class": "status-risk",
            "emoji": "🔴",
            "last_check": "10 minutes ago",
            "extra": "High acoustic variability; MFCC drift suggests queen or parasite issue.",
        },
        {
            "name": "Hive 4 – New Nuc",
            "status": "No Data",
            "status_class": "status-unknown",
            "emoji": "⚪️",
            "last_check": "—",
            "extra": "Awaiting first recording to establish baseline.",
        },
    ]

    cols = [col1, col2, col1, col2]
    for card, col in zip(hive_cards, cols):
        with col:
            st.markdown(
                f"""
                <div style="background:linear-gradient(135deg,#FFFFFF,#F9FAFB);border-radius:1.1rem;padding:0.9rem;border:1px solid #E5E7EB;box-shadow:0 6px 18px rgba(15,23,42,0.06);margin-bottom:0.6rem;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.25rem;">
                        <div style="font-weight:600;font-size:0.95rem;">{card['name']}</div>
                        <span class="status-pill {card['status_class']}">
                            {card['emoji']} {card['status']}
                        </span>
                    </div>
                    <div style="font-size:0.78rem;color:#6B7280;">Last check: {card['last_check']}</div>
                    <div style="font-size:0.8rem;color:#4B5563;margin-top:0.25rem;">{card['extra']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# -----------------------------
# TAB 3: SPECTROGRAM EXPLORER
# -----------------------------
with tab_specs:
    st.markdown("#### Spectrogram explorer")
    if st.session_state["last_waveform"] is None:
        st.info("Analyze a recording in the first tab to visualize its spectrogram and MFCCs here.")
    else:
        wf = st.session_state["last_waveform"]
        sr = st.session_state["last_sr"]

        view = st.radio(
            "Choose a view",
            ["Mel spectrogram", "MFCCs"],
            horizontal=True,
        )

        if view == "Mel spectrogram":
            fig1 = plot_spectrogram(wf, sr)
            st.pyplot(fig1)
            st.caption(
                "Brighter bands show where the hive is putting more energy. "
                "Higher energy in bands around 2–4 kHz can indicate agitation or strong ventilation."
            )
        else:
            fig2 = plot_mfcc(wf, sr)
            st.pyplot(fig2)
            st.caption(
                "MFCCs compress the frequency content into a few smooth curves. "
                "Stable lower-order coefficients often correlate with steady brood buzz; "
                "rapid changes can signal disturbance or environmental shifts."
            )

# -----------------------------
# TAB 4: HOW IT WORKS
# -----------------------------
with tab_how:
    st.markdown("#### How Sonabee works")
    st.markdown(
        """
        Sonabee is a **software-first prototype** inspired by commercial hive monitors and recent research in honeybee acoustics.

        **Pipeline overview**
        1. **Audio capture** – A short 30–60s recording is taken from inside or at the entrance of the hive.
        2. **Pre-processing** – Audio is normalized and converted to mel-frequency features.
        3. **Feature extraction** – We compute MFCCs and spectral descriptors (energy, centroid, variability).
        4. **Context fusion** – Ambient temperature is included, since hive behavior shifts with environmental conditions.
        5. **Classification** – A lightweight machine learning model (e.g., random forest) outputs a health / stress estimate.
        6. **Interpretation** – Heuristics based on bee acoustics research are used to generate human-readable explanations.

        **Why acoustics?**
        - Bees communicate via **vibrations and sound** (wing beats, piping, fanning).
        - Studies show acoustic signatures change with **queen presence, swarming, parasites, and thermal stress**.
        - Sound is non-invasive: you can monitor the colony without opening the hive.

        Sonabee aims to **lower the barrier** by using **simple recordings + software** instead of proprietary hardware.
        """
    )

# -----------------------------
# TAB 5: MODEL CARD
# -----------------------------
with tab_model:
    st.markdown("#### Model card (prototype)")

    st.markdown(
        """
        **Intended use**

        - Early **screening tool** for potential hive stress.
        - Educational demo of bioacoustics + ML for bee health.
        - Not intended to replace in-person hive inspection or professional apiary management.

        **Inputs**

        - 30–60s hive audio recording (`.wav`)
        - Ambient temperature (°C), manually provided or fetched via weather API.

        **Model**

        - Backend: classical ML classifier (e.g., Random Forest or similar) trained on:
          - MFCC-based acoustic features from hive recordings
          - Optional ambient temperature as context
        - Output: binary label (healthy vs. stressed/at-risk) + confidence scores.

        **Limitations**

        - Trained on a specific dataset of hives and environments; performance may degrade on very different setups.
        - Sensitive to loud background noise (speech, traffic, heavy wind).
        - Can misinterpret **edge cases** such as queen introduction, aggressive manipulations, or very small colonies.

        **Safety & ethics**

        - Sonabee should **never** be the sole basis for management decisions.
        - Use it as an extra lens on your hive, alongside visual inspection, weight, mites, and local beekeeper expertise.
        """
    )

st.markdown("</div>", unsafe_allow_html=True)
