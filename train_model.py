import os
import pandas as pd
import joblib

print(">>> running train_model.py")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Make sure models/ folder exists
os.makedirs("models", exist_ok=True)

# 1. Load the dataset we built
df = pd.read_csv("data/features_with_weather.csv")

# X = all features, y = label
X = df.drop("label", axis=1).values
y = df["label"].values  # 0 = healthy, 1 = at-risk (from label_health)

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Define model pipeline: scale features + RandomForest
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    ))
])

# 4. Train
clf.fit(X_train, y_train)

# 5. Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# 6. Save the trained model
model_path = "models/sonabee_rf.pkl"
joblib.dump(clf, model_path)
print("Saved model to", model_path)
