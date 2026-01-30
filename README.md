# Sonabee — Acoustic Hive Intelligence

Lightweight toolkit for extracting acoustic features from hive audio, building a dataset, training a Random Forest classifier, and running a Streamlit inspection UI.

## Summary

Sonabee processes raw hive audio to extract MFCC and simple energy features, combines those with ambient/weather temperature, and trains a classifier to detect at-risk hives. A Streamlit app (`app.py`) provides a simple UI for uploading audio, fetching temperature (optional via OpenWeather), visualizing acoustics, and showing a health assessment.

## Repo structure

- `app.py` — Streamlit app for inspection and quick predictions
- `build_dataset.py` — create `data/features_with_weather.csv` from audio and metadata
- `features.py` — audio feature extraction (MFCC, RMS, amplitude std)
- `train_model.py` — train and save a RandomForest model to `models/`
- `data/features_with_weather.csv` — example dataset produced by `build_dataset.py`
- `sound_files/` — audio files used by `build_dataset.py`

## Requirements

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
