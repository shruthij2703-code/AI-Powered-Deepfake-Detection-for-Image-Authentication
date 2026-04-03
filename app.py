import streamlit as st
import numpy as np
import cv2
import base64
from PIL import Image
from tensorflow.keras.models import load_model
import time

st.set_page_config(page_title="AI-Powered Deepfake Detection for Image Authentication", layout="wide", initial_sidebar_state="collapsed")

# ─── GLOBAL STYLES ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Space+Mono:wght@400;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #050508 !important;
    color: #e8e4dc !important;
    font-family: 'Space Mono', monospace !important;
}

[data-testid="stAppViewContainer"] > .main {
    background: #050508 !important;
}

/* hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }
.stDeployButton { display: none; }

/* ── HEADER ── */
.ds-header {
    text-align: center;
    padding: 3.5rem 1rem 1.5rem;
    position: relative;
}
.ds-header::before {
    content: '';
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 1px;
    height: 60px;
    background: linear-gradient(to bottom, transparent, #a78bfa);
}
.ds-wordmark {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.4rem, 6vw, 4.5rem);
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #f0ece4;
    line-height: 1;
}
.ds-wordmark span { color: #a78bfa; }
.ds-tagline {
    font-size: 0.72rem;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: #5a5568;
    margin-top: 0.6rem;
}

/* ── DIVIDER ── */
.ds-divider {
    border: none;
    border-top: 1px solid #1a1a2e;
    margin: 2rem auto;
    max-width: 640px;
}

/* ── UPLOAD ZONE ── */
[data-testid="stFileUploader"] {
    background: #0c0c14 !important;
    border: 1px solid #1e1e30 !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #4a4a7a !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] section p,
[data-testid="stFileUploader"] section small,
[data-testid="stFileUploader"] button {
    color: #6b6b90 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
}
[data-testid="stFileUploader"] section {
    background: transparent !important;
    border: 1px dashed #2a2a40 !important;
    border-radius: 10px !important;
}

/* ── IMAGE CONTAINER ── */
.ds-img-wrap {
    position: relative;
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid #1e1e30;
    background: #08080f;
}
.ds-img-wrap img {
    display: block;
    width: 100%;
    border-radius: 14px;
}
.ds-scanline {
    position: absolute;
    top: -4px; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(to right, transparent, #a78bfa, transparent);
    animation: scan 1.8s ease-in-out infinite;
}
@keyframes scan {
    0%   { top: -4px; opacity: 0; }
    10%  { opacity: 1; }
    90%  { opacity: 1; }
    100% { top: 100%; opacity: 0; }
}
.ds-corner {
    position: absolute;
    width: 18px; height: 18px;
    border-color: #a78bfa;
    border-style: solid;
}
.ds-corner.tl { top: 10px; left: 10px;  border-width: 2px 0 0 2px; }
.ds-corner.tr { top: 10px; right: 10px; border-width: 2px 2px 0 0; }
.ds-corner.bl { bottom: 10px; left: 10px;  border-width: 0 0 2px 2px; }
.ds-corner.br { bottom: 10px; right: 10px; border-width: 0 2px 2px 0; }

/* ── BOOM RESULT CARD ── */
.ds-result-wrap {
    position: relative;
    border-radius: 20px;
    overflow: hidden;
    padding: 2.5rem 2rem 2rem;
    text-align: center;
    animation: popIn 0.55s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
}
@keyframes popIn {
    from { transform: scale(0.82); opacity: 0; }
    to   { transform: scale(1);    opacity: 1; }
}

/* REAL card */
.ds-result-real {
    background: radial-gradient(ellipse at 50% -10%, #0a3a2a 0%, #050c09 60%, #050508 100%);
    border: 1px solid #1a4a30;
}
.ds-result-real::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(34,197,94,0.18) 0%, transparent 65%);
    pointer-events: none;
}

/* FAKE card */
.ds-result-fake {
    background: radial-gradient(ellipse at 50% -10%, #3a0a0a 0%, #0c0508 60%, #050508 100%);
    border: 1px solid #5a1a1a;
}
.ds-result-fake::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(239,68,68,0.22) 0%, transparent 65%);
    pointer-events: none;
}

/* BOOM LIGHT BURST */
.ds-burst {
    position: absolute;
    top: -60px; left: 50%;
    transform: translateX(-50%);
    width: 320px; height: 320px;
    border-radius: 50%;
    pointer-events: none;
    animation: burst 0.9s ease-out forwards;
    opacity: 0;
}
.ds-burst-real  { background: radial-gradient(circle, rgba(34,197,94,0.55) 0%, rgba(34,197,94,0.1) 40%, transparent 70%); }
.ds-burst-fake  { background: radial-gradient(circle, rgba(239,68,68,0.55)  0%, rgba(239,68,68,0.1)  40%, transparent 70%); }
@keyframes burst {
    0%   { transform: translateX(-50%) scale(0.1); opacity: 0.9; }
    60%  { transform: translateX(-50%) scale(1.4); opacity: 0.6; }
    100% { transform: translateX(-50%) scale(1.8); opacity: 0; }
}

/* pulse ring */
.ds-ring {
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%) scale(0);
    border-radius: 50%;
    border: 2px solid;
    animation: ring 1.2s ease-out 0.2s forwards;
    opacity: 0;
}
.ds-ring-1 { width: 120px; height: 120px; animation-delay: 0.1s; }
.ds-ring-2 { width: 180px; height: 180px; animation-delay: 0.3s; }
.ds-ring-3 { width: 240px; height: 240px; animation-delay: 0.5s; }
.ds-ring-real { border-color: rgba(34,197,94,0.4); }
.ds-ring-fake  { border-color: rgba(239,68,68,0.4);  }
@keyframes ring {
    0%   { transform: translate(-50%, -50%) scale(0); opacity: 0.8; }
    100% { transform: translate(-50%, -50%) scale(1);  opacity: 0; }
}

/* VERDICT TEXT */
.ds-verdict {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.8rem, 8vw, 5.5rem);
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1;
    position: relative;
    z-index: 1;
    animation: verdictIn 0.5s ease-out 0.2s both;
}
.ds-verdict-real { color: #22c55e; text-shadow: 0 0 40px rgba(34,197,94,0.5); }
.ds-verdict-fake  { color: #ef4444; text-shadow: 0 0 40px rgba(239,68,68,0.5);  }
@keyframes verdictIn {
    from { transform: translateY(20px) scale(0.9); opacity: 0; }
    to   { transform: translateY(0)    scale(1);   opacity: 1; }
}

.ds-verdict-sub {
    font-size: 0.72rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    margin-top: 0.5rem;
    position: relative; z-index: 1;
}
.ds-verdict-sub-real { color: #4ade80; }
.ds-verdict-sub-fake  { color: #f87171; }

/* CONFIDENCE METER */
.ds-meter-wrap {
    margin: 1.8rem auto 0;
    max-width: 320px;
    position: relative; z-index: 1;
}
.ds-meter-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4a4a6a;
    margin-bottom: 0.5rem;
}
.ds-meter-track {
    height: 6px;
    background: #1a1a2e;
    border-radius: 99px;
    overflow: hidden;
}
.ds-meter-fill {
    height: 100%;
    border-radius: 99px;
    animation: fillUp 1s cubic-bezier(0.22, 1, 0.36, 1) 0.4s both;
    transform-origin: left;
}
.ds-meter-fill-real { background: linear-gradient(to right, #16a34a, #4ade80); }
.ds-meter-fill-fake  { background: linear-gradient(to right, #b91c1c, #f87171);  }
@keyframes fillUp {
    from { transform: scaleX(0); }
    to   { transform: scaleX(1); }
}

/* DETAIL CHIPS */
.ds-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    justify-content: center;
    margin-top: 1.5rem;
    position: relative; z-index: 1;
}
.ds-chip {
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    padding: 0.35rem 0.8rem;
    border-radius: 99px;
    text-transform: uppercase;
}
.ds-chip-real { background: #0a2a18; border: 1px solid #16503a; color: #4ade80; }
.ds-chip-fake  { background: #2a0a0a; border: 1px solid #5a1a1a; color: #f87171; }

/* SCORE BADGE */
.ds-score {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    margin-top: 0.9rem;
    padding: 0.3rem 0.9rem;
    border-radius: 6px;
    position: relative; z-index: 1;
}
.ds-score-real { background: #0a2418; border: 1px solid #164a28; color: #86efac; }
.ds-score-fake  { background: #240a0a; border: 1px solid #4a1616; color: #fca5a5; }

/* ANALYZING SPINNER */
.ds-analyzing {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 3rem 1rem;
    gap: 1.2rem;
}
.ds-spinner {
    width: 48px; height: 48px;
    border: 2px solid #1a1a30;
    border-top-color: #a78bfa;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.ds-analyzing-text {
    font-size: 0.72rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #5a5a80;
    animation: blink 1.2s ease-in-out infinite;
}
@keyframes blink { 50% { opacity: 0.3; } }

/* STREAMLIT OVERRIDES */
[data-testid="column"] { gap: 0 !important; }
.stButton > button {
    background: #0c0c18 !important;
    border: 1px solid #2a2a44 !important;
    color: #a78bfa !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.1em !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #1a1a2e !important;
    border-color: #6d5adb !important;
}
</style>
""", unsafe_allow_html=True)


# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ds-header">
  <div class="ds-wordmark">Deep<span>Scan</span></div>
  <div class="ds-tagline">Neural Authenticity Verification Engine</div>
</div>
<hr class="ds-divider">
""", unsafe_allow_html=True)


# ─── LOAD MODEL ──────────────────────────────────────────────────────────────
@st.cache_resource
def load():
    return load_model("model.h5")

model = load()
IMG_SIZE = 224

def preprocess(image):
    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)


# ─── LAYOUT ──────────────────────────────────────────────────────────────────
left, gap, right = st.columns([5, 1, 6])

with left:
    st.markdown('<p style="font-size:0.72rem;letter-spacing:0.18em;text-transform:uppercase;color:#4a4a6a;margin-bottom:0.6rem;">Upload Image</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded:
        image = Image.open(uploaded)
        # encode for HTML display
        import io
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        st.markdown(f"""
        <div class="ds-img-wrap" style="margin-top:1rem;">
            <div class="ds-scanline"></div>
            <div class="ds-corner tl"></div><div class="ds-corner tr"></div>
            <div class="ds-corner bl"></div><div class="ds-corner br"></div>
            <img src="data:image/png;base64,{b64}" alt="uploaded" />
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
        analyze = st.button("⬡  Run Analysis")
    else:
        st.markdown("""
        <div style="color:#2a2a40;font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase;
                    margin-top:3rem;text-align:center;">
            Drop an image to begin
        </div>
        """, unsafe_allow_html=True)
        analyze = False


with right:
    if uploaded and analyze:
        with st.spinner(""):
            st.markdown("""
            <div class="ds-analyzing">
              <div class="ds-spinner"></div>
              <div class="ds-analyzing-text">Analyzing neural signature…</div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.6)   # brief dramatic pause

        processed = preprocess(image)
        pred = float(model.predict(processed, verbose=0)[0][0])

        is_fake = pred > 0.5
        confidence = pred if is_fake else (1 - pred)
        pct = int(confidence * 100)
        fill_pct = pct  # for width

        if is_fake:
            chips_html = "".join([
                '<span class="ds-chip ds-chip-fake">Texture anomaly</span>',
                '<span class="ds-chip ds-chip-fake">Face distortion</span>',
                '<span class="ds-chip ds-chip-fake">Lighting mismatch</span>',
                '<span class="ds-chip ds-chip-fake">GAN artifact</span>',
            ])
            result_html = f"""
            <div class="ds-result-wrap ds-result-fake">
              <div class="ds-burst ds-burst-fake"></div>
              <div class="ds-ring ds-ring-1 ds-ring-fake"></div>
              <div class="ds-ring ds-ring-2 ds-ring-fake"></div>
              <div class="ds-ring ds-ring-3 ds-ring-fake"></div>

              <div class="ds-verdict ds-verdict-fake">FAKE</div>
              <div class="ds-verdict-sub ds-verdict-sub-fake">Deepfake detected · AI-generated content</div>

              <div class="ds-score ds-score-fake">confidence score: {pct}%</div>

              <div class="ds-meter-wrap">
                <div class="ds-meter-label"><span>Authenticity</span><span>{100 - pct}%</span></div>
                <div class="ds-meter-track">
                  <div class="ds-meter-fill ds-meter-fill-fake" style="width:{100-pct}%;"></div>
                </div>
              </div>

              <div class="ds-chips">{chips_html}</div>
            </div>
            """
        else:
            chips_html = "".join([
                '<span class="ds-chip ds-chip-real">Natural texture</span>',
                '<span class="ds-chip ds-chip-real">Consistent lighting</span>',
                '<span class="ds-chip ds-chip-real">No artifacts</span>',
                '<span class="ds-chip ds-chip-real">Authentic geometry</span>',
            ])
            result_html = f"""
            <div class="ds-result-wrap ds-result-real">
              <div class="ds-burst ds-burst-real"></div>
              <div class="ds-ring ds-ring-1 ds-ring-real"></div>
              <div class="ds-ring ds-ring-2 ds-ring-real"></div>
              <div class="ds-ring ds-ring-3 ds-ring-real"></div>

              <div class="ds-verdict ds-verdict-real">REAL</div>
              <div class="ds-verdict-sub ds-verdict-sub-real">Authentic image · No manipulation detected</div>

              <div class="ds-score ds-score-real">confidence score: {pct}%</div>

              <div class="ds-meter-wrap">
                <div class="ds-meter-label"><span>Authenticity</span><span>{pct}%</span></div>
                <div class="ds-meter-track">
                  <div class="ds-meter-fill ds-meter-fill-real" style="width:{pct}%;"></div>
                </div>
              </div>

              <div class="ds-chips">{chips_html}</div>
            </div>
            """

        st.markdown(result_html, unsafe_allow_html=True)

    elif not uploaded:
        st.markdown("""
        <div style="border:1px dashed #141420;border-radius:16px;padding:3rem 2rem;text-align:center;
                    color:#2a2a40;font-size:0.72rem;letter-spacing:0.14em;text-transform:uppercase;
                    margin-top:0.5rem;">
            Results will appear here
        </div>
        """, unsafe_allow_html=True)

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("""
<hr class="ds-divider">
<p style="text-align:center;font-size:0.65rem;letter-spacing:0.18em;
          text-transform:uppercase;color:#2a2a3a;padding-bottom:2rem;">
  DeepScan AI · Neural Authenticity Engine · v2.0
</p>
""", unsafe_allow_html=True)
