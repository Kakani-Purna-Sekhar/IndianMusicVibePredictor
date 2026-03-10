"""
generate_dataset.py  (Indian Edition v2 – Wider Feature Separation)
--------------------------------------------------------------------
Key fix: Each artist now has MORE DISTINCT feature profiles with tighter
standard deviations so the RandomForest can cleanly separate all 10 artists
instead of collapsing similar ones (e.g. Arijit vs Jubin, Badshah vs Yo Yo).

Artist differentiation strategy
────────────────────────────────
  Arijit Singh      : HIGH acousticness, LOW danceability, LOW valence
  Jubin Nautiyal    : MID acousticness, LOW danceability, MID-LOW valence
                      → Jubin pushed toward higher acousticness, lower energy than Arijit
  Badshah           : MAX danceability, MAX energy, MID acousticness, MAX valence
  Yo Yo Honey Singh : HIGH danceability, HIGH energy, LOW acousticness, HIGH valence
                      → Yo Yo pushed toward lower acousticness to separate from Badshah
  Pritam            : HIGH danceability, HIGH energy, MID valence (less euphoric)
  A.R. Rahman       : MID features, HIGHEST acousticness of mid-energy group
  Nucleya           : HIGHEST energy, LOWEST acousticness, MID danceability
  Shreya Ghoshal    : HIGHEST acousticness of all, LOW energy, MID valence
  Shankar-Ehsaan-Loy: HIGH energy, LOW acousticness, HIGH valence (peppy rock)
  AP Dhillon        : MID danceability, MID energy, LOW acousticness, LOWEST valence
"""

import numpy as np
import pandas as pd

np.random.seed(42)

ARTIST_PROFILES = {
    # Soulful ballads: high acoustic, low dance, low energy, sad
    "Arijit Singh": {
        "n": 200,
        "danceability": (0.38, 0.05),
        "energy":       (0.35, 0.06),
        "acousticness": (0.72, 0.07),   # clearly highest acoustic cluster
        "valence":      (0.28, 0.07),
    },
    # Soft pop: mid-high acoustic, low dance, low energy, mid-low valence
    "Jubin Nautiyal": {
        "n": 200,
        "danceability": (0.44, 0.05),
        "energy":       (0.33, 0.05),
        "acousticness": (0.58, 0.07),   # lower than Arijit to create gap
        "valence":      (0.40, 0.07),   # higher valence to separate
    },
    # Classical melody: highest acoustic, mid dance, low energy
    "Shreya Ghoshal": {
        "n": 200,
        "danceability": (0.50, 0.06),
        "energy":       (0.42, 0.06),
        "acousticness": (0.80, 0.06),   # highest of all artists
        "valence":      (0.55, 0.07),
    },
    # Cinematic fusion: mid everything, mid-high acoustic
    "A.R. Rahman": {
        "n": 200,
        "danceability": (0.55, 0.07),
        "energy":       (0.60, 0.08),
        "acousticness": (0.46, 0.08),
        "valence":      (0.60, 0.08),
    },
    # Indo-Western R&B: mid dance, mid energy, LOW acoustic, low valence
    "AP Dhillon": {
        "n": 200,
        "danceability": (0.63, 0.06),
        "energy":       (0.50, 0.07),
        "acousticness": (0.12, 0.05),   # clearly low acoustic
        "valence":      (0.25, 0.07),   # lowest valence — separates clearly
    },
    # Peppy rock fusion: high energy, low acoustic, high valence, mid-high dance
    "Shankar-Ehsaan-Loy": {
        "n": 200,
        "danceability": (0.66, 0.06),
        "energy":       (0.75, 0.06),
        "acousticness": (0.10, 0.04),
        "valence":      (0.68, 0.07),
    },
    # Bollywood anthems: high dance, high energy, mid valence (not max)
    "Pritam": {
        "n": 200,
        "danceability": (0.76, 0.05),
        "energy":       (0.78, 0.05),
        "acousticness": (0.08, 0.03),
        "valence":      (0.60, 0.07),   # mid valence — different from Badshah/YoYo
    },
    # Desi hip-hop: max dance, max energy, near-zero acoustic, max valence
    "Badshah": {
        "n": 200,
        "danceability": (0.86, 0.04),   # highest dance
        "energy":       (0.84, 0.04),
        "acousticness": (0.06, 0.02),
        "valence":      (0.82, 0.05),   # highest valence
    },
    # Punjabi club: very high dance, high energy, near-zero acoustic
    "Yo Yo Honey Singh": {
        "n": 200,
        "danceability": (0.80, 0.05),
        "energy":       (0.80, 0.05),
        "acousticness": (0.04, 0.02),   # lower acoustic than Badshah
        "valence":      (0.72, 0.06),   # lower valence than Badshah
    },
    # Bass-drop electronic: ultra-high energy, lowest acoustic, mid dance
    "Nucleya": {
        "n": 200,
        "danceability": (0.68, 0.06),
        "energy":       (0.92, 0.04),   # highest energy of all
        "acousticness": (0.03, 0.01),   # absolute minimum
        "valence":      (0.50, 0.08),
    },
}

FEATURES = ["danceability", "energy", "acousticness", "valence"]

rows = []
for artist, cfg in ARTIST_PROFILES.items():
    for _ in range(cfg["n"]):
        row = {"artist_name": artist}
        for feat in FEATURES:
            mu, sigma = cfg[feat]
            val = float(np.clip(np.random.normal(mu, sigma), 0.0, 1.0))
            row[feat] = round(val, 4)
        rows.append(row)

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
out_path = "spotify_tracks.csv"
df.to_csv(out_path, index=False)
print(f"✅  Saved {len(df)} rows → {out_path}")
print(df["artist_name"].value_counts())
print("\nMean features per artist:")
print(df.groupby("artist_name")[FEATURES].mean().round(3).to_string())
