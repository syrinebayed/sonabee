# Sonabee — Acoustic Hive Intelligence

Sonabee is a lightweight machine-learning toolkit for monitoring honeybee hive health using sound.

I built this project out of a long-standing obsession with bees and a curiosity about whether subtle acoustic patterns — often ignored as background noise — could act as early signals of colony stress. Beekeepers frequently notice problems too late, after visible collapse has already begun. Sonabee explores whether listening more carefully can shift that timeline earlier.

The system processes raw hive audio, extracts interpretable acoustic features, and trains a simple but effective classifier to flag potentially at-risk hives. The emphasis is on practical signals, fast iteration, and transparency, rather than black-box models.

## What It Does

- Extracts MFCCs and basic energy features from hive audio
- Combines acoustic signals with ambient temperature data
- Trains a Random Forest classifier to assess hive health risk
- Provides a Streamlit interface for inspection, visualization, and rapid testing
  
## Repo Structure

- `app.py` — Streamlit app for interactive inspection and quick predictions
- `build_dataset.py` — builds data/features_with_weather.csv from audio + metadata
- `features.py` — audio feature extraction (MFCC, RMS energy, amplitude statistics)
- `train_model.py` — trains and saves a Random Forest model
- `data/features_with_weather.csv` — example dataset produced by the pipeline
- `sound_files/` — raw hive audio used for feature extraction
  
## Set Up

Install dependencies using:

```bash
pip install -r requirements.txt
```

Minimum tested Python: 3.8+

## Quick start

1. Prepare the metadata CSV `all_data_updated.csv` in the repo root and put WAV files in `sound_files/`.
2. Build features CSV:

```bash
python build_dataset.py
# -> creates data/features_with_weather.csv
```

3. Train the model:

```bash
python train_model.py
# -> saves models/sonabee_rf.pkl
```

4. Run the Streamlit app (optional OpenWeather temp):

```bash
export OPENWEATHER_API_KEY=your_api_key_here
streamlit run app.py
```

The app will try common model filenames under `models/` and will use the saved pipeline.

## Notes & implementation details

- `features.py` uses `soundfile` to read audio, resamples to 16 kHz, and computes 13 MFCCs plus mean/std and two simple energy stats.
- `build_dataset.py` appends the recorded weather temperature as an additional feature before saving `data/features_with_weather.csv`.
- `train_model.py` trains a pipeline with `StandardScaler` and `RandomForestClassifier` (300 trees, class-balanced) and saves it via `joblib`.
- `app.py` contains helpers to extract features, prepare vectors to match model input size (handles models trained with/without temperature), visualizations (mel spectrogram / MFCCs) and a small rule-of-thumb interpreter for results.

