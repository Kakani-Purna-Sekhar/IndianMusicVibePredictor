"""
utils.py  (Indian Edition)
--------------------------
Helper functions shared between train_model.py and app.py.
Covers:
  • Loading the saved model
  • Predicting an artist from slider values
  • Deriving a human-readable "vibe" description (Indian music context)
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Tuple, Any

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# Model I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_model() -> Tuple[object, object]:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. Please run:  python train_model.py"
        )
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["label_encoder"]


def load_model_payload() -> Dict[str, Any]:
    """
    Load the entire serialized payload (model, encoder, optional metrics).
    Kept separate from `load_model()` so the app can show model insights when available.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. Please run:  python train_model.py"
        )
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    return payload


def predict_artist(clf, le, danceability, energy, acousticness, valence):
    X = np.array([[danceability, energy, acousticness, valence]])
    pred_idx = clf.predict(X)[0]
    artist   = le.inverse_transform([pred_idx])[0]
    proba    = clf.predict_proba(X)[0]
    proba_dict = {
        le.inverse_transform([i])[0]: round(float(p), 4)
        for i, p in enumerate(proba)
    }
    proba_dict = dict(sorted(proba_dict.items(), key=lambda x: x[1], reverse=True))
    return artist, proba_dict


def top_k_predictions(proba_dict: Dict[str, float], k: int = 3) -> List[Tuple[str, float]]:
    """Return top-k (artist, probability) pairs from a sorted probability dict."""
    if not proba_dict:
        return []
    return list(proba_dict.items())[:k]


# ─────────────────────────────────────────────────────────────────────────────
# Vibe Engine  – Indian music context
# ─────────────────────────────────────────────────────────────────────────────

_VIBES = [
    (
        "Punjabi Party Anthem",
        "🥁",
        "Dhol beats, punchy bass, and lyrics that turn any room into a dance floor. "
        "This is the vibe of sangeet nights, rooftop parties, and weddings that go till dawn. "
        "Crank it up — your bhangra moves are being summoned.",
        lambda d, e, ac, v: d >= 0.78 and e >= 0.75 and v >= 0.65,
    ),
    (
        "Desi Bass Drop",
        "🔊",
        "Heavy sub-bass collides with folk samples in a wall of electronic energy. "
        "Think festival mosh pits, bass-first speakers, and Nucleya-level drops that "
        "physically move the crowd.",
        lambda d, e, ac, v: e >= 0.82 and ac < 0.12,
    ),
    (
        "Bollywood Dance Floor",
        "🎬",
        "High-octane filmy beats with infectious hooks straight out of a blockbuster. "
        "Choreography optional, energy mandatory. The kind of song that hijacks every "
        "playlist at every celebration.",
        lambda d, e, ac, v: d >= 0.70 and e >= 0.68 and v >= 0.55,
    ),
    (
        "Soulful Bollywood Ballad",
        "💔",
        "Velvet vocals over lush orchestration — the architecture of longing. "
        "Rain-window music, late-night drives, and every heartbreak montage ever filmed "
        "in Bollywood history lives here.",
        lambda d, e, ac, v: e < 0.48 and ac >= 0.45 and v < 0.45,
    ),
    (
        "Classical Melody Vibes",
        "🪕",
        "Rooted in ragas, carried by pristine vocals, and built on centuries of "
        "Indian classical tradition. Meditative, precise, and transcendent — "
        "music that feels like a sunrise over the Ganges.",
        lambda d, e, ac, v: ac >= 0.58 and e < 0.55,
    ),
    (
        "Cinematic Fusion",
        "🎻",
        "Orchestral swells entwined with tabla, sitar harmonics, and electronic textures. "
        "A.R. Rahman territory — music that feels like a journey across landscapes "
        "and lifetimes simultaneously.",
        lambda d, e, ac, v: 0.40 <= ac < 0.60 and 0.45 <= e < 0.70,
    ),
    (
        "Indo-Western Moody R&B",
        "🌙",
        "Midnight Punjabi verses over minimal trap beats — cool, cinematic, and "
        "achingly romantic. AP Dhillon territory: the sound of longing in two languages, "
        "driving through city lights at 2 AM.",
        lambda d, e, ac, v: 0.55 <= d < 0.75 and e < 0.62 and v < 0.45,
    ),
    (
        "Peppy Filmy Pop",
        "🌟",
        "Bright, feel-good Bollywood pop with catchy refrains and an irresistible "
        "spring in its step. The sonic equivalent of the cheerful title montage in "
        "every coming-of-age film.",
        lambda d, e, ac, v: d >= 0.60 and v >= 0.60 and e >= 0.55,
    ),
]

_DEFAULT_VIBE = (
    "Eclectic Desi Fusion",
    "🎵",
    "A beautifully balanced blend that borrows from classical roots, filmy tradition, "
    "and contemporary production — impossible to pin down, unforgettable to hear. "
    "Truly the spirit of modern Indian music.",
)


def get_vibe(danceability, energy, acousticness, valence):
    for label, emoji, desc, cond in _VIBES:
        if cond(danceability, energy, acousticness, valence):
            return label, emoji, desc
    return _DEFAULT_VIBE


# ─────────────────────────────────────────────────────────────────────────────
# Artist metadata
# ─────────────────────────────────────────────────────────────────────────────

ARTIST_META = {
    "Arijit Singh": {
        "color":   "#c0785a",
        "icon":    "🌹",
        "tagline": "The voice of a generation — soulful, aching, and impossibly romantic.",
    },
    "A.R. Rahman": {
        "color":   "#4a90c4",
        "icon":    "🕌",
        "tagline": "Cinematic maestro — where classical India meets global electronica.",
    },
    "Pritam": {
        "color":   "#e8b84b",
        "icon":    "🎬",
        "tagline": "Bollywood's feel-good hitmaker — every track is an instant party.",
    },
    "Badshah": {
        "color":   "#e05c2a",
        "icon":    "👑",
        "tagline": "Desi hip-hop royalty — bangers that break chart records effortlessly.",
    },
    "Nucleya": {
        "color":   "#22c55e",
        "icon":    "💥",
        "tagline": "Bass-drop architect — folk samples detonated through massive sub-woofers.",
    },
    "Shreya Ghoshal": {
        "color":   "#c084fc",
        "icon":    "🪷",
        "tagline": "Classical purity meets Bollywood warmth — a voice like still water.",
    },
    "Shankar–Ehsaan–Loy": {
        "color":   "#f97316",
        "icon":    "🎸",
        "tagline": "Fusion trailblazers — Bollywood rock with an irresistible groove.",
    },
    "Jubin Nautiyal": {
        "color":   "#60a5fa",
        "icon":    "🌧️",
        "tagline": "Tender romantic pop — his falsetto lives rent-free in your heart.",
    },
    "Yo Yo Honey Singh": {
        "color":   "#facc15",
        "icon":    "🎤",
        "tagline": "Punjabi club pioneer — the OG who made desi rap go mainstream.",
    },
    "AP Dhillon": {
        "color":   "#a78bfa",
        "icon":    "🌃",
        "tagline": "Indo-Western midnight R&B — brooding, bilingual, and irresistibly cool.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Artist profile cards (dashboard content)
# ─────────────────────────────────────────────────────────────────────────────

ARTIST_PROFILES: Dict[str, Dict[str, Any]] = {
    "Arijit Singh": {
        "description": "Soulful romantic ballads with a deep, emotional vocal signature.",
        "signature_style": ["Heartfelt melodies", "Orchestral ballads", "Modern Bollywood romance"],
        "top_songs": ["Tum Hi Ho", "Channa Mereya", "Kesariya"],
    },
    "A.R. Rahman": {
        "description": "Cinematic composer blending Indian classical textures with global production.",
        "signature_style": ["Orchestral fusion", "World-music palettes", "Innovative sound design"],
        "top_songs": ["Jai Ho", "Kun Faya Kun", "Tum Tak"],
    },
    "Pritam": {
        "description": "Bollywood hitmaker known for catchy hooks and dance-floor energy.",
        "signature_style": ["Upbeat pop", "Festival-ready rhythms", "Earworm choruses"],
        "top_songs": ["Shayad", "Badtameez Dil", "Agar Tum Saath Ho"],
    },
    "Badshah": {
        "description": "Chart-topping desi hip-hop with punchy beats and party-first hooks.",
        "signature_style": ["Club rap", "High energy", "Hook-driven singles"],
        "top_songs": ["DJ Waley Babu", "Genda Phool", "Paagal"],
    },
    "Nucleya": {
        "description": "Bass-heavy electronic producer mixing Indian samples with festival EDM.",
        "signature_style": ["Massive drops", "Folk/voice samples", "Underground-to-mainstream energy"],
        "top_songs": ["Laung Gawacha", "Bass Rani", "Jumbo"],
    },
    "Shreya Ghoshal": {
        "description": "Powerful, classically trained vocalist with versatile Bollywood range.",
        "signature_style": ["Classical ornamentation", "Romantic playback", "Pure acoustic timbre"],
        "top_songs": ["Teri Ore", "Sun Raha Hai (Female)", "Deewani Mastani"],
    },
    "Shankar–Ehsaan–Loy": {
        "description": "Fusion-forward trio known for Bollywood rock grooves and big melodies.",
        "signature_style": ["Bollywood rock", "Guitar-led hooks", "Uplifting choruses"],
        "top_songs": ["Mitwa", "Kajra Re", "Dil Dhadakne Do"],
    },
    "Jubin Nautiyal": {
        "description": "Tender romantic pop vocals with soft acoustic-leaning arrangements.",
        "signature_style": ["Gentle ballads", "Modern romantic pop", "Warm vocal tone"],
        "top_songs": ["Raataan Lambiyan", "Lut Gaye", "Tum Hi Aana"],
    },
    "Yo Yo Honey Singh": {
        "description": "Punjabi club pioneer with punchy rap, catchy chants, and bold production.",
        "signature_style": ["Punjabi pop-rap", "Club-ready beats", "Swagger-forward hooks"],
        "top_songs": ["Brown Rang", "Angreji Beat", "Lungi Dance"],
    },
    "AP Dhillon": {
        "description": "Indo-Western moody Punjabi R&B with cool, nocturnal vibes.",
        "signature_style": ["Minimal trap beats", "Moody melodies", "Late-night drive energy"],
        "top_songs": ["Brown Munde", "Excuses", "Insane"],
    },
}


def get_artist_profile(artist_name: str) -> Dict[str, Any]:
    """Safe accessor for artist profile content used in the dashboard."""
    return ARTIST_PROFILES.get(
        artist_name,
        {
            "description": "A unique sonic signature shaped by modern Indian music.",
            "signature_style": ["Fusion-first sound", "Distinct mood palette"],
            "top_songs": [],
        },
    )
