"""
main.py  —  Indian Music Vibe Predictor  v4
===========================================
UI improvements in this version
────────────────────────────────
  UI v4 – Polished background: animated aurora gradient, subtle dot-grid
           texture, glass-morphism sidebar, glowing gradient button,
           richer panel depth, enhanced contrast throughout.

Run:  streamlit run main.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils import (
    load_model_payload,
    predict_artist,
    get_vibe,
    top_k_predictions,
    get_artist_profile,
    ARTIST_META,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Music Vibe Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# ENHANCED BACKGROUND + FULL UI STYLING  (v4)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ─── Fonts ─────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Yeseva+One&family=Hind:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

/* ─── CSS variables ──────────────────────────────────────────────────────── */
:root {
  --orange:  #fb923c;
  --rose:    #f43f5e;
  --gold:    #fbbf24;
  --purple:  #a78bfa;
  --teal:    #2dd4bf;
  --bg:      #06040e;
  --surface: rgba(255,255,255,0.04);
  --border:  rgba(255,255,255,0.09);
  --text:    #f0ece4;
  --muted:   #6b7280;
}

/* ─── Base reset ─────────────────────────────────────────────────────────── */
html, body {
  background: var(--bg) !important;
  color: var(--text);
  font-family: 'Hind', sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ══════════════════════════════════════════════════════════════════════════
   ENHANCED BACKGROUND LAYER STACK
   1. Base body colour       : very deep indigo-black
   2. ::before pseudo        : animated aurora radial gradient
   3. Body background-image  : subtle SVG dot-grid texture
   ══════════════════════════════════════════════════════════════════════════ */

/* Layer 1 — animated aurora overlay */
[data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  position: relative;
}

[data-testid="stAppViewContainer"]::before {
  content: "";
  position: fixed;
  inset: 0;
  z-index: 0;
  /* Multi-stop aurora: saffron NW, deep-rose centre, violet SE, teal NE */
  background:
    radial-gradient(ellipse 72% 55% at  8%  8%,  rgba(251,100,20,0.28)  0%, transparent 55%),
    radial-gradient(ellipse 60% 55% at 92% 93%,  rgba(120,40,220,0.26)  0%, transparent 52%),
    radial-gradient(ellipse 55% 50% at 50% 42%,  rgba(200,25,65,0.14)   0%, transparent 62%),
    radial-gradient(ellipse 50% 38% at 88%  6%,  rgba(30,200,195,0.14)  0%, transparent 46%),
    radial-gradient(ellipse 40% 35% at 18% 88%,  rgba(251,191,36,0.10)  0%, transparent 50%),
    linear-gradient(160deg, #0e0618 0%, #0b0514 40%, #0d0b20 70%, #06040e 100%);
  animation: auroraShift 14s ease-in-out infinite alternate;
  pointer-events: none;
}

@keyframes auroraShift {
  0%   {
    background:
      radial-gradient(ellipse 72% 55% at  8%  8%,  rgba(251,100,20,0.28)  0%, transparent 55%),
      radial-gradient(ellipse 60% 55% at 92% 93%,  rgba(120,40,220,0.26)  0%, transparent 52%),
      radial-gradient(ellipse 55% 50% at 50% 42%,  rgba(200,25,65,0.14)   0%, transparent 62%),
      radial-gradient(ellipse 50% 38% at 88%  6%,  rgba(30,200,195,0.14)  0%, transparent 46%),
      radial-gradient(ellipse 40% 35% at 18% 88%,  rgba(251,191,36,0.10)  0%, transparent 50%),
      linear-gradient(160deg, #0e0618 0%, #0b0514 40%, #0d0b20 70%, #06040e 100%);
  }
  50%  {
    background:
      radial-gradient(ellipse 65% 60% at 15% 15%,  rgba(251,146,60,0.32)  0%, transparent 50%),
      radial-gradient(ellipse 55% 60% at 85% 88%,  rgba(167,139,250,0.28) 0%, transparent 48%),
      radial-gradient(ellipse 60% 48% at 42% 55%,  rgba(244,63,94,0.16)   0%, transparent 58%),
      radial-gradient(ellipse 45% 42% at 82% 10%,  rgba(45,212,191,0.16)  0%, transparent 44%),
      radial-gradient(ellipse 38% 38% at 22% 82%,  rgba(251,191,36,0.14)  0%, transparent 48%),
      linear-gradient(160deg, #100820 0%, #0c0618 40%, #0f0d25 70%, #06040e 100%);
  }
  100% {
    background:
      radial-gradient(ellipse 68% 52% at 12% 12%,  rgba(255,80,10,0.24)   0%, transparent 54%),
      radial-gradient(ellipse 58% 52% at 90% 90%,  rgba(100,30,200,0.24)  0%, transparent 50%),
      radial-gradient(ellipse 52% 46% at 55% 38%,  rgba(220,30,70,0.13)   0%, transparent 60%),
      radial-gradient(ellipse 48% 36% at 85%  8%,  rgba(20,190,185,0.13)  0%, transparent 44%),
      radial-gradient(ellipse 36% 32% at 20% 85%,  rgba(251,191,36,0.09)  0%, transparent 48%),
      linear-gradient(160deg, #0c0516 0%, #0a0412 40%, #0c0a1e 70%, #06040e 100%);
  }
}

/* Layer 2 — subtle dot-grid texture overlay */
[data-testid="stAppViewContainer"]::after {
  content: "";
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  /* Radial dots 24 px apart, very faint */
  background-image:
    radial-gradient(circle, rgba(255,255,255,0.055) 1px, transparent 1px);
  background-size: 24px 24px;
  /* Fade the texture toward the edges so it doesn't compete with the glow */
  -webkit-mask-image:
    radial-gradient(ellipse 80% 80% at 50% 50%, black 40%, transparent 100%);
  mask-image:
    radial-gradient(ellipse 80% 80% at 50% 50%, black 40%, transparent 100%);
  animation: dotDrift 30s linear infinite;
}

@keyframes dotDrift {
  0%   { background-position: 0 0; }
  100% { background-position: 24px 24px; }
}

/* ─── Floating music-note layer ──────────────────────────────────────────── */
.note-layer {
  position: fixed;
  inset: 0;
  z-index: 0;
  font-size: 1.9rem;
  color: rgba(251,146,60,0.045);
  line-height: 5rem;
  letter-spacing: 3.8rem;
  word-spacing: 2.8rem;
  overflow: hidden;
  pointer-events: none;
  animation: noteFloat 20s linear infinite;
  white-space: pre-wrap;
  user-select: none;
}
@keyframes noteFloat {
  0%   { transform: translateY(0)    rotate(0deg); }
  50%  { transform: translateY(-28px) rotate(1.2deg); }
  100% { transform: translateY(0)    rotate(0deg); }
}

/* Vinyl pulse ring behind the hero */
.vinyl-ring {
  position: absolute;
  left: 50%; top: 0;
  transform: translateX(-50%) translateY(-60%);
  width: 580px; height: 580px;
  border-radius: 50%;
  background: conic-gradient(
    from 0deg,
    rgba(251,146,60,0.07),
    rgba(244,63,94,0.07),
    rgba(167,139,250,0.07),
    rgba(45,212,191,0.05),
    rgba(251,146,60,0.07)
  );
  animation: vinylSpin 24s linear infinite;
  pointer-events: none;
  z-index: 0;
}
@keyframes vinylSpin { to { transform: translateX(-50%) translateY(-60%) rotate(360deg); } }

/* Audio wave bar row (decorative, CSS only) */
.wave-row {
  display: flex;
  align-items: flex-end;
  justify-content: center;
  gap: 3px;
  height: 28px;
  margin: 0.6rem 0 1.4rem;
}
.wave-bar {
  width: 4px;
  border-radius: 2px;
  animation: waveAnim 1.2s ease-in-out infinite alternate;
}
.wave-bar:nth-child(1)  { height:10px; animation-delay:0.0s; background: var(--orange); }
.wave-bar:nth-child(2)  { height:20px; animation-delay:0.1s; background: var(--gold);   }
.wave-bar:nth-child(3)  { height:14px; animation-delay:0.2s; background: var(--rose);   }
.wave-bar:nth-child(4)  { height:25px; animation-delay:0.3s; background: var(--orange); }
.wave-bar:nth-child(5)  { height:18px; animation-delay:0.4s; background: var(--purple); }
.wave-bar:nth-child(6)  { height:22px; animation-delay:0.5s; background: var(--gold);   }
.wave-bar:nth-child(7)  { height:12px; animation-delay:0.6s; background: var(--rose);   }
.wave-bar:nth-child(8)  { height:28px; animation-delay:0.7s; background: var(--orange); }
.wave-bar:nth-child(9)  { height:16px; animation-delay:0.8s; background: var(--teal);   }
.wave-bar:nth-child(10) { height:22px; animation-delay:0.9s; background: var(--gold);   }
.wave-bar:nth-child(11) { height:10px; animation-delay:1.0s; background: var(--purple); }
.wave-bar:nth-child(12) { height:19px; animation-delay:1.1s; background: var(--rose);   }
@keyframes waveAnim {
  from { transform: scaleY(0.3); opacity:0.5; }
  to   { transform: scaleY(1.0); opacity:1.0; }
}

/* ─── All content sits above the background ──────────────────────────────── */
[data-testid="stAppViewContainer"] > .main > .block-container {
  position: relative;
  z-index: 1;
}

/* ─── Glass-morphism sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background:
    linear-gradient(180deg, rgba(18,8,35,0.92) 0%, rgba(10,6,22,0.96) 100%) !important;
  border-right: 1px solid rgba(251,146,60,0.14) !important;
  backdrop-filter: blur(18px) !important;
  -webkit-backdrop-filter: blur(18px) !important;
}
[data-testid="stSidebar"]::before {
  content: "";
  position: absolute;
  inset: 0;
  background:
    radial-gradient(ellipse 90% 40% at 50% 0%,   rgba(251,100,20,0.10) 0%, transparent 55%),
    radial-gradient(ellipse 70% 30% at 50% 100%, rgba(120,40,220,0.10) 0%, transparent 50%);
  pointer-events: none;
  z-index: 0;
}

/* ─── Glowing gradient Analyze button ───────────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stButton"] > button,
[data-testid="stSidebar"] .stButton > button {
  background: linear-gradient(135deg, #e05a10 0%, #c4254a 55%, #7c22c8 100%) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 12px !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.05em !important;
  padding: 0.6rem 1rem !important;
  box-shadow: 0 0 18px rgba(228, 80, 20, 0.40), 0 2px 8px rgba(0,0,0,0.5) !important;
  transition: box-shadow 0.25s ease, transform 0.15s ease !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button:hover,
[data-testid="stSidebar"] .stButton > button:hover {
  box-shadow: 0 0 30px rgba(228, 80, 20, 0.65), 0 4px 16px rgba(0,0,0,0.6) !important;
  transform: translateY(-1px) !important;
}

/* Slider accent colour */
[data-testid="stSlider"] > div > div > div > div {
  background: linear-gradient(90deg, var(--orange), var(--rose)) !important;
}

/* ─── Hero ───────────────────────────────────────────────────────────────── */
.hero {
  text-align: center;
  padding: 2.2rem 0 0.6rem;
  position: relative;
  overflow: visible;
}
.hero-title {
  font-family: 'Yeseva One', serif;
  font-size: clamp(2rem, 4.8vw, 3.8rem);
  background: linear-gradient(135deg, #fff 10%, var(--orange) 40%, var(--gold) 68%, var(--rose) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1.1;
  margin-bottom: 0.35rem;
  filter: drop-shadow(0 0 18px rgba(251,146,60,0.30));
}
.hero-sub {
  font-size: 0.97rem;
  font-weight: 300;
  color: #9ca3af;
  letter-spacing: 0.06em;
}

/* ─── Panels ─────────────────────────────────────────────────────────────── */
.panel {
  background: linear-gradient(145deg, rgba(255,255,255,0.055) 0%, rgba(255,255,255,0.02) 100%);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 22px;
  padding: 1.8rem 1.7rem 2rem;
  box-shadow:
    0 4px 32px rgba(0,0,0,0.55),
    inset 0 1px 0 rgba(255,255,255,0.07);
  backdrop-filter: blur(6px);
  -webkit-backdrop-filter: blur(6px);
}
.section-label {
  font-family: 'Space Mono', monospace;
  font-size: 0.60rem;
  letter-spacing: 0.2em;
  color: var(--muted);
  text-transform: uppercase;
  margin-bottom: 1.2rem;
}

/* ─── Tooltip / feature description rows ────────────────────────────────── */
.feat-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.1rem;
}
.feat-label {
  font-weight: 600;
  font-size: 0.88rem;
  color: #e5e0d8;
}
.tip-wrap {
  position: relative;
  display: inline-flex;
  align-items: center;
}
.tip-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 16px; height: 16px;
  border-radius: 50%;
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.18);
  font-size: 0.62rem;
  color: var(--muted);
  cursor: help;
  font-family: 'Space Mono', monospace;
  font-weight: 700;
  flex-shrink: 0;
}
.tip-icon:hover { background: rgba(251,146,60,0.15); border-color: var(--orange); color: var(--orange); }
.tip-wrap .tip-box {
  visibility: hidden;
  opacity: 0;
  position: absolute;
  left: 50%; top: calc(100% + 8px);
  transform: translateX(-50%);
  background: #1c1826;
  border: 1px solid rgba(251,146,60,0.25);
  border-radius: 10px;
  padding: 0.65rem 0.85rem;
  width: 230px;
  font-size: 0.76rem;
  font-family: 'Hind', sans-serif;
  color: #d1c9c0;
  line-height: 1.55;
  z-index: 999;
  pointer-events: none;
  transition: opacity 0.2s ease;
  box-shadow: 0 8px 24px rgba(0,0,0,0.5);
}
.tip-wrap .tip-box::before {
  content: "";
  position: absolute;
  top: -6px; left: 50%;
  transform: translateX(-50%);
  border: 6px solid transparent;
  border-bottom-color: rgba(251,146,60,0.25);
  border-top: none;
}
.tip-wrap:hover .tip-box { visibility: visible; opacity: 1; }

/* ─── Feature snapshot mini-bars ─────────────────────────────────────────── */
.feat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.7rem; margin-top: 0.3rem; }
.feat-snap-name {
  font-size: 0.66rem; color: var(--muted);
  font-family: 'Space Mono', monospace; text-transform: uppercase;
  letter-spacing: 0.08em; margin-bottom: 0.22rem;
}
.feat-bar-bg  { height: 4px; background: rgba(255,255,255,0.07); border-radius: 999px; overflow: hidden; }
.feat-bar-fill{ height: 100%; border-radius: 999px; }

/* ─── Artist result card ─────────────────────────────────────────────────── */
.artist-card {
  border-radius: 18px;
  padding: 1.45rem 1.6rem;
  margin-bottom: 1rem;
}
.artist-icon  { font-size: 2.5rem; display: block; margin-bottom: 0.3rem; }
.artist-name  {
  font-family: 'Yeseva One', serif;
  font-size: 1.72rem; color: #fff;
  line-height: 1.15; margin-bottom: 0.22rem;
}
.artist-tagline { font-size: 0.83rem; font-weight:300; color:rgba(255,255,255,0.58); font-style:italic; }

/* ─── Vibe badge ─────────────────────────────────────────────────────────── */
.vibe-badge {
  display: inline-flex; align-items: center; gap: 0.4rem;
  background: rgba(251,146,60,0.09);
  border: 1px solid rgba(251,146,60,0.26);
  border-radius: 999px; padding: 0.28rem 0.9rem;
  font-family: 'Space Mono', monospace;
  font-size: 0.74rem; color: var(--orange);
  margin-bottom: 0.85rem; letter-spacing: 0.04em;
}

/* ─── Vibe description ───────────────────────────────────────────────────── */
.vibe-desc {
  background: rgba(255,255,255,0.02);
  border-left: 3px solid;
  border-radius: 0 12px 12px 0;
  padding: 0.9rem 1.1rem;
  font-size: 0.87rem; line-height: 1.66;
  color: #c9d1d9; margin-bottom: 1.1rem;
}

/* ─── Top-3 podium cards ─────────────────────────────────────────────────── */
.podium { display: flex; flex-direction: column; gap: 0.6rem; }
.podium-card {
  display: flex; align-items: center; gap: 0.85rem;
  border-radius: 13px;
  padding: 0.75rem 1rem;
  border: 1px solid rgba(255,255,255,0.07);
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.podium-card:hover {
  border-color: rgba(251,146,60,0.35);
  box-shadow: 0 0 12px rgba(251,146,60,0.12);
}
.rank-badge {
  font-family: 'Yeseva One', serif;
  font-size: 1.55rem;
  line-height: 1;
  min-width: 2rem; text-align: center;
}
.podium-info { flex: 1; min-width: 0; }
.podium-name {
  font-family: 'Hind', sans-serif;
  font-weight: 600; font-size: 0.92rem;
  color: #f0ece4;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.podium-bar-row { display: flex; align-items: center; gap: 0.5rem; margin-top: 0.3rem; }
.podium-bar-bg {
  flex: 1; height: 5px;
  background: rgba(255,255,255,0.07);
  border-radius: 999px; overflow: hidden;
}
.podium-bar-fill { height: 100%; border-radius: 999px; }
.podium-pct {
  font-family: 'Space Mono', monospace;
  font-size: 0.72rem; color: var(--muted);
  min-width: 2.5rem; text-align: right;
}

/* ─── Responsive gap ─────────────────────────────────────────────────────── */
[data-testid="stColumns"] > div:first-child { padding-right: 0.8rem; }
[data-testid="stColumns"] > div:last-child  { padding-left:  0.8rem; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Data + Model loading (cached)
# ══════════════════════════════════════════════════════════════════════════════

DATASET_PATH = "dataset/spotify_tracks.csv"


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"danceability", "energy", "acousticness", "valence", "artist_name"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return df


@st.cache_resource(show_spinner="🎵  Loading model…")
def load_model_cached():
    return load_model_payload()

try:
    payload = load_model_cached()
    clf, le = payload["model"], payload["label_encoder"]
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

def plot_radar_chart(d: float, e: float, ac: float, v: float) -> go.Figure:
    categories = ["Danceability", "Energy", "Acousticness", "Valence"]
    values = [d, e, ac, v]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="Your mood",
            line=dict(color="#fb923c", width=2),
            fillcolor="rgba(251,146,60,0.18)",
        )
    )
    fig.update_layout(
        title="Your Music Mood Profile",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 1], showticklabels=False, ticks=""),
            angularaxis=dict(tickfont=dict(color="rgba(240,236,228,0.75)")),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=45, b=20),
        height=340,
        showlegend=False,
        font=dict(color="rgba(240,236,228,0.82)"),
    )
    return fig


def plot_artist_map(df: pd.DataFrame, d: float, e: float) -> go.Figure:
    means = (
        df.groupby("artist_name")[["energy", "danceability"]]
        .mean()
        .reset_index()
        .rename(columns={"artist_name": "artist"})
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=means["energy"],
            y=means["danceability"],
            mode="markers+text",
            text=means["artist"],
            textposition="top center",
            marker=dict(size=10, color="rgba(96,165,250,0.7)", line=dict(width=1, color="rgba(255,255,255,0.18)")),
            hovertemplate="<b>%{text}</b><br>Energy=%{x:.2f}<br>Danceability=%{y:.2f}<extra></extra>",
            name="Artists",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[e],
            y=[d],
            mode="markers",
            marker=dict(size=15, color="#fb923c", symbol="star", line=dict(width=1, color="rgba(255,255,255,0.25)")),
            hovertemplate="<b>You</b><br>Energy=%{x:.2f}<br>Danceability=%{y:.2f}<extra></extra>",
            name="You",
        )
    )
    fig.update_layout(
        title="Artist Sound Map (Energy × Danceability)",
        xaxis=dict(title="Energy", range=[0, 1], gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        yaxis=dict(title="Danceability", range=[0, 1], gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(l=40, r=20, t=55, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(color="rgba(240,236,228,0.82)"),
    )
    return fig


def render_model_insights(model_payload: dict, df: pd.DataFrame):
    metrics = model_payload.get("metrics") or {}
    model_type = metrics.get("model_type") or type(model_payload.get("model")).__name__

    n_rows = int(metrics.get("n_rows") or len(df))
    n_artists = int(metrics.get("n_artists") or df["artist_name"].nunique())
    acc = metrics.get("accuracy")
    trained_at = metrics.get("trained_at_utc")

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">🤖&nbsp; Model Information</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    c1.metric("Model type", model_type)
    c2.metric("Dataset size", f"{n_rows:,}")
    c3.metric("Artists", f"{n_artists}")
    c4.metric("Accuracy", f"{acc*100:.1f}%" if isinstance(acc, (int, float)) else "N/A")

    if trained_at:
        st.caption(f"Trained at (UTC): {trained_at}")
    elif acc is None:
        st.caption("Tip: run `python train_model.py` to save accuracy into `model/model.pkl` for this panel.")

    st.markdown("</div>", unsafe_allow_html=True)


def sidebar_controls() -> tuple[float, float, float, float, bool]:
    st.sidebar.markdown("## 🎚 Mood Controls")
    st.sidebar.caption("Set your Spotify-style audio features (0 → 1). Then analyze for your top matches.")

    danceability = st.sidebar.slider(
        "💃 Danceability",
        0.0,
        1.0,
        0.60,
        0.01,
        help="How suitable the track is for dancing. Higher = groove/club energy.",
    )
    energy = st.sidebar.slider(
        "⚡ Energy",
        0.0,
        1.0,
        0.55,
        0.01,
        help="Intensity and activity. Higher = louder, faster, more energetic.",
    )
    acousticness = st.sidebar.slider(
        "🪕 Acousticness",
        0.0,
        1.0,
        0.35,
        0.01,
        help="Confidence the track is acoustic/unplugged. Higher = live instruments, less synthesis.",
    )
    valence = st.sidebar.slider(
        "😊 Valence",
        0.0,
        1.0,
        0.50,
        0.01,
        help="Musical positivity. Low = melancholic, high = cheerful/euphoric.",
    )

    st.sidebar.divider()
    analyze = st.sidebar.button("🎧 Analyze your music mood", use_container_width=True)
    return danceability, energy, acousticness, valence, analyze

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="vinyl-ring"></div>
  <div class="hero-title">🎵 Indian Music Vibe Predictor</div>
  <div class="hero-sub">Slide your mood — find your Bollywood &amp; Desi sonic match in real time</div>
</div>
<!-- animated equaliser wave -->
<div class="wave-row">
  <div class="wave-bar"></div><div class="wave-bar"></div>
  <div class="wave-bar"></div><div class="wave-bar"></div>
  <div class="wave-bar"></div><div class="wave-bar"></div>
  <div class="wave-bar"></div><div class="wave-bar"></div>
  <div class="wave-bar"></div><div class="wave-bar"></div>
  <div class="wave-bar"></div><div class="wave-bar"></div>
</div>
<!-- floating music notes layer -->
<div class="note-layer">♩ ♪ ♫ ♬ ♩ ♪ ♫ ♬ ♩ ♪ ♫ ♬ ♩ ♪</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
danceability, energy, acousticness, valence, analyze_clicked = sidebar_controls()

try:
    df = load_data(DATASET_PATH)
except Exception as exc:
    st.error(f"Dataset problem: {exc}")
    st.stop()


def run_analysis() -> dict:
    with st.spinner("Analyzing your music mood..."):
        artist, proba_dict = predict_artist(clf, le, danceability, energy, acousticness, valence)
        vibe_label, vibe_emoji, vibe_desc = get_vibe(danceability, energy, acousticness, valence)
        top3 = top_k_predictions(proba_dict, k=3)
        radar = plot_radar_chart(danceability, energy, acousticness, valence)
        amap = plot_artist_map(df, danceability, energy)
        return {
            "artist": artist,
            "proba_dict": proba_dict,
            "top3": top3,
            "vibe": (vibe_label, vibe_emoji, vibe_desc),
            "radar": radar,
            "artist_map": amap,
        }


if "analysis" not in st.session_state:
    st.session_state["analysis"] = run_analysis()
elif analyze_clicked:
    st.session_state["analysis"] = run_analysis()

analysis = st.session_state["analysis"]
artist = analysis["artist"]
proba_dict = analysis["proba_dict"]
top3 = analysis["top3"]
vibe_label, vibe_emoji, vibe_desc = analysis["vibe"]

col_left, col_right = st.columns([1.05, 1.25], gap="large")

# ── Podium colour constants (kept in sync with CSS variables) ─────────────────
PODIUM_BAR_COLORS = [
    "linear-gradient(90deg,#fb923c,#f43f5e)",   # orange → rose  (rank 1)
    "linear-gradient(90deg,#a78bfa,#60a5fa)",   # purple → blue  (rank 2)
    "linear-gradient(90deg,#34d399,#2dd4bf)",   # green  → teal  (rank 3)
]
PODIUM_CARD_BGS = [
    "rgba(251,146,60,0.08)",   # orange tint
    "rgba(167,139,250,0.06)",  # purple tint
    "rgba(52,211,153,0.06)",   # teal tint
]

# ─────────────────────────────────────────────────────────────────────────────
# LEFT PANEL — Mood profile + charts
# ─────────────────────────────────────────────────────────────────────────────
with col_left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">📊&nbsp; Feature Snapshot</div>', unsafe_allow_html=True)

    def progress_row(name: str, value: float):
        left, right = st.columns([0.55, 0.45])
        left.markdown(f'<div class="feat-snap-name">{name}</div>', unsafe_allow_html=True)
        right.markdown(
            f'<div class="feat-snap-name" style="text-align:right;">{value:.2f}</div>',
            unsafe_allow_html=True,
        )
        st.progress(value)

    progress_row("Energy", energy)
    progress_row("Danceability", danceability)
    progress_row("Acousticness", acousticness)
    progress_row("Valence", valence)

    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(analysis["radar"], use_container_width=True, config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(analysis["artist_map"], use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)  # close .panel

# ─────────────────────────────────────────────────────────────────────────────
# RIGHT PANEL — Results
# ─────────────────────────────────────────────────────────────────────────────
with col_right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">🔮&nbsp; Your Music Match</div>', unsafe_allow_html=True)

    meta        = ARTIST_META.get(artist, {"color": "#fb923c", "icon": "🎵", "tagline": ""})
    card_color  = meta["color"]
    artist_icon = meta["icon"]
    artist_tag  = meta["tagline"]

    # ── #1 Artist card ────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="artist-card"
         style="background:linear-gradient(135deg,{card_color}26 0%,{card_color}06 100%);
                border:1px solid {card_color}4a;">
      <span class="artist-icon">{artist_icon}</span>
      <div class="artist-name">{artist}</div>
      <div class="artist-tagline">{artist_tag}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Vibe badge + description ──────────────────────────────────────────
    st.markdown(
        f'<div class="vibe-badge">{vibe_emoji}&nbsp;&nbsp;{vibe_label}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="vibe-desc" style="border-color:{card_color}99;">{vibe_desc}</div>',
        unsafe_allow_html=True,
    )

    # ── TOP-3 PODIUM CARDS ────────────────────────────────────────────────
    st.markdown(
        '<div class="section-label">🏆&nbsp; Top 3 Predicted Matches</div>',
        unsafe_allow_html=True,
    )

    medals   = ["🥇", "🥈", "🥉"]
    max_conf = top3[0][1] if top3 else 1.0

    podium_html = '<div class="podium">'
    for i, (name, conf) in enumerate(top3):
        pct         = conf * 100
        bar_width   = (conf / max_conf) * 100
        artist_meta = ARTIST_META.get(name, {"icon": "🎵"})
        icon        = artist_meta["icon"]
        podium_html += f"""
        <div class="podium-card" style="background:{PODIUM_CARD_BGS[i]};">
          <div class="rank-badge">{medals[i]}</div>
          <div style="font-size:1.4rem;line-height:1;">{icon}</div>
          <div class="podium-info">
            <div class="podium-name">{name}</div>
            <div class="podium-bar-row">
              <div class="podium-bar-bg">
                <div class="podium-bar-fill"
                     style="width:{bar_width:.1f}%;background:{PODIUM_BAR_COLORS[i]};"></div>
              </div>
              <div class="podium-pct">{pct:.1f}%</div>
            </div>
          </div>
        </div>"""
    podium_html += "</div>"
    st.markdown(podium_html, unsafe_allow_html=True)

    # ── Artist profile ────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">🎙️&nbsp; Artist Profile</div>', unsafe_allow_html=True)

    prof = get_artist_profile(artist)
    st.markdown(f"**About**: {prof['description']}")
    st.markdown("**Signature style**")
    for s in prof.get("signature_style", [])[:4]:
        st.markdown(f"- {s}")
    if prof.get("top_songs"):
        st.markdown("**Top songs**")
        st.markdown(", ".join(prof["top_songs"]))
    st.markdown("</div>", unsafe_allow_html=True)  # close .panel

# ─────────────────────────────────────────────────────────────────────────────
# Model insights (full width)
# ─────────────────────────────────────────────────────────────────────────────
render_model_insights(payload, df)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:2rem;padding-bottom:1.2rem;position:relative;z-index:1;">
  <span style="font-family:'Space Mono',monospace;font-size:0.60rem;
               color:#374151;letter-spacing:0.14em;">
    🇮🇳&nbsp; INDIAN MUSIC VIBE PREDICTOR &nbsp;·&nbsp;
    STREAMLIT ML DASHBOARD &nbsp;·&nbsp;
    BUILT WITH STREAMLIT
  </span>
</div>
""", unsafe_allow_html=True)
