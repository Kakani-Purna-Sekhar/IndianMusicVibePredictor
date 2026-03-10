# 🎧 Mood to Music Vibe Predictor

An interactive Streamlit web app that predicts which pop artist or music era best matches your current mood — powered by Spotify audio features and a Random Forest classifier.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎚 Live Sliders | Tune danceability, energy, acousticness, and valence in real time |
| 🔮 Instant Predictions | RandomForest model updates the match on every slider change |
| 💅 Vibe Engine | Rule-based system labels your mood (Dark Pop, Indie Acoustic, etc.) |
| 📈 Confidence Bars | Top-5 artist matches shown with relative confidence scores |
| 🎨 Custom Dark UI | Concert-hall aesthetic with gradient accents |

---

## 🗂 Project Structure

```
music_vibe_predictor/
├── dataset/
│   ├── generate_dataset.py   # Generates synthetic Spotify-style CSV
│   └── spotify_tracks.csv    # Auto-generated dataset (1 600 rows)
├── model/
│   └── model.pkl             # Trained model (auto-generated)
├── app.py                    # Streamlit app — main entry point
├── train_model.py            # Model training script
├── utils.py                  # Prediction helpers & vibe rules
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1 · Install dependencies

```bash
pip install -r requirements.txt
```

### 2 · Generate the dataset (first time only)

```bash
cd dataset
python generate_dataset.py
cd ..
```

> **Using your own Kaggle dataset?**  
> Download a Spotify Tracks CSV from Kaggle, rename it `spotify_tracks.csv`,  
> place it in `dataset/`, and make sure it has these columns:  
> `danceability`, `energy`, `acousticness`, `valence`, `artist_name`.

### 3 · Train the model

```bash
python train_model.py
```

This saves `model/model.pkl` (~61 % accuracy on 10 artists).

### 4 · Launch the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🎛 Audio Features Explained

| Feature | Range | What it means |
|---|---|---|
| **Danceability** | 0–1 | How suitable the track is for dancing |
| **Energy** | 0–1 | Perceived intensity and activity level |
| **Acousticness** | 0–1 | Confidence the track is acoustic (unplugged) |
| **Valence** | 0–1 | Musical positivity — sad (0) → euphoric (1) |

---

## 🎶 Artists in the Model

| Artist / Era | Signature Sound |
|---|---|
| Taylor Swift – Folklore | Indie folk, introspective |
| Taylor Swift – 1989 | Polished synth-pop |
| Taylor Swift – Reputation | Dark, cinematic pop |
| Taylor Swift – Fearless | Country-pop, uplifting |
| Billie Eilish | Whispery, minimal dark pop |
| Dua Lipa | Disco-pop, high energy |
| Olivia Rodrigo | Alt-pop, emotional rock |
| Harry Styles | Retro rock, feel-good |
| The Weeknd | Nocturnal synth-noir |
| Ariana Grande | R&B-pop, vocal-forward |

---

## 🛠 Tech Stack

- **Python 3.10+**
- **Streamlit** – interactive web UI
- **scikit-learn** – RandomForestClassifier
- **pandas / numpy** – data handling
- **pickle** – model serialisation

---

## 📝 Extending the Project

- **Swap in real Kaggle data** – download the [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) and replace `spotify_tracks.csv`.
- **Add more artists** – extend `ARTIST_PROFILES` in `generate_dataset.py` and re-train.
- **Try other models** – swap `RandomForestClassifier` for `GradientBoostingClassifier` or `SVC` in `train_model.py`.
- **Add a radar chart** – use `plotly` to visualise the four features as a spider chart.
