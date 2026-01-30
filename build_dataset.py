import pandas as pd
import numpy as np
import os

print(">>> running build_dataset.py")

from features import extract_features

CSV_PATH = "all_data_updated.csv"   # metadata CSV
AUDIO_DIR = "sound_files"           # folder with wav files

# Make sure data/ exists
os.makedirs("data", exist_ok=True)

# 1. Load metadata
df = pd.read_csv(CSV_PATH)

# 2. Create health label from queen + hive temp + frames
def label_health(row):
    # Adjust thresholds as needed
    queen_ok = row["queen presence"] == 1
    temp_ok = 33 <= row["hive temp"] <= 36
    frames_ok = row["frames"] >= 6

    if queen_ok and temp_ok and frames_ok:
        return 0   # healthy
    else:
        return 1   # at-risk / unhealthy

df["health_binary"] = df.apply(label_health, axis=1)

# 3. Build dataset
X = []
y = []

for i, row in df.iterrows():
    # Prefix from .raw name
    raw_name = row["file name"]           # e.g. '2022-06-08--17-21-53_1.raw'
    prefix = raw_name.replace(".raw", "") # e.g. '2022-06-08--17-21-53_1'

    weather = row["weather temp"]
    label = df.loc[i, "health_binary"]

    # Find matching wav file like '...__segment0.wav'
    candidates = [
        f for f in os.listdir(AUDIO_DIR)
        if prefix in f and f.endswith(".wav")
    ]

    if len(candidates) == 0:
        # No audio found for this row, skip
        continue

    audio_path = os.path.join(AUDIO_DIR, candidates[0])

    try:
        audio_feats = extract_features(audio_path)
        if audio_feats is None:
            continue
    except Exception as e:
        print("Error on file", audio_path, ":", e)
        continue

    # Append weather temp as last feature
    full_features = np.hstack([audio_feats, weather])

    X.append(full_features)
    y.append(label)

# 4. Save features + labels
feature_df = pd.DataFrame(X)
feature_df["label"] = y
out_path = "data/features_with_weather.csv"
feature_df.to_csv(out_path, index=False)

print("DONE. Saved dataset to", out_path, "with shape:", feature_df.shape)
