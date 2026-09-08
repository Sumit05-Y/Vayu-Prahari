from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "synthetic_anomalies_2020_2024.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "isolation_forest.joblib"
)

FEATURE_PATH = (
    PROJECT_ROOT
    / "models"
    / "feature_columns.joblib"
)


# ============================================================
# 2. CHECK FILES
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}\n"
        "Run: python src/train.py"
    )

if not FEATURE_PATH.exists():
    raise FileNotFoundError(
        f"Feature list not found:\n{FEATURE_PATH}\n"
        "Run: python src/train.py"
    )


# ============================================================
# 3. LOAD MODEL
# ============================================================

print("=" * 60)
print("VAYU PRAHARI - ISOLATION FOREST PREDICTION")
print("=" * 60)

print("\nLoading model...")

model = joblib.load(
    MODEL_PATH
)

feature_columns = joblib.load(
    FEATURE_PATH
)

print("Model loaded successfully.")


# ============================================================
# 4. LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_PATH
)

print("Dataset shape:", df.shape)


# ============================================================
# 5. TIMESTAMP
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

df = (
    df
    .sort_values("timestamp")
    .reset_index(drop=True)
)


# ============================================================
# 6. CONVERT FEATURES TO NUMERIC
# ============================================================

for column in feature_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# 7. SELECT 2024 TEST DATA
# ============================================================

test_df = df[
    df["timestamp"] >= "2024-01-01"
].copy()


if test_df.empty:
    raise ValueError(
        "2024 test dataset is empty."
    )


print("\nTest dataset shape:")
print(test_df.shape)


# ============================================================
# 8. REMOVE INCOMPLETE FEATURE ROWS
# ============================================================

before_drop = len(test_df)

test_df = test_df.dropna(
    subset=feature_columns
).copy()

after_drop = len(test_df)


print("\nRows before missing-feature removal:")
print(before_drop)

print("Rows after missing-feature removal:")
print(after_drop)


# ============================================================
# 9. CREATE X_TEST
# ============================================================

X_test = test_df[
    feature_columns
]


print("\nX_test shape:")
print(X_test.shape)


if X_test.empty:
    raise ValueError(
        "X_test is empty."
    )


# ============================================================
# 10. MODEL PREDICTION
# ============================================================

print("\nRunning predictions...")

raw_predictions = model.predict(
    X_test
)


# ============================================================
# 11. CONVERT ISOLATION FOREST OUTPUT
# ============================================================
#
# Isolation Forest:
#
# +1 = normal
# -1 = anomaly
#
# Vayu Prahari:
#
# 0 = normal
# 1 = anomaly
# ============================================================

predictions = np.where(
    raw_predictions == -1,
    1,
    0
)


# ============================================================
# 12. ANOMALY SCORE
# ============================================================

scores = model.decision_function(
    X_test
)


# ============================================================
# 13. ADD RESULTS
# ============================================================

test_df["prediction"] = predictions

test_df["anomaly_score"] = scores


# ============================================================
# 14. SAVE RESULTS
# ============================================================

RESULT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "day5_vscode_test_results.csv"
)

test_df.to_csv(
    RESULT_PATH,
    index=False
)


# ============================================================
# 15. DISPLAY SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION COMPLETE")
print("=" * 60)

print("\nPredicted labels:")
print(
    test_df["prediction"]
    .value_counts()
    .sort_index()
)

print("\nActual labels:")
print(
    test_df["anomaly"]
    .value_counts()
    .sort_index()
)

print("\nResults saved to:")
print(RESULT_PATH)


# ============================================================
# 16. SHOW SAMPLE RESULTS
# ============================================================

display_columns = [
    "timestamp",
    "temperature",
    "humidity",
    "pressure",
    "anomaly",
    "anomaly_type",
    "prediction",
    "anomaly_score"
]

print("\nSample predictions:")

print(
    test_df[
        display_columns
    ].head(20).to_string(index=False)
)