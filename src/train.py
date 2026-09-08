from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest


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

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "isolation_forest.joblib"
FEATURE_PATH = MODEL_DIR / "feature_columns.joblib"


# ============================================================
# 2. MODEL FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "temperature",
    "humidity",
    "pressure",

    "temp_prev",
    "humidity_prev",
    "pressure_prev",

    "temp_diff",
    "humidity_diff",
    "pressure_diff",

    "temp_rate",
    "humidity_rate",
    "pressure_rate",

    "temp_roll_mean_3",
    "humidity_roll_mean_3",
    "pressure_roll_mean_3",

    "temp_roll_std_3",
    "humidity_roll_std_3",
    "pressure_roll_std_3",

    "hour",
    "month",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos"
]


# ============================================================
# 3. CHECK FILES
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 4. LOAD DATA
# ============================================================

print("=" * 60)
print("VAYU PRAHARI - ISOLATION FOREST TRAINING")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = (
    FEATURE_COLUMNS
    + [
        "timestamp",
        "anomaly",
        "anomaly_type"
    ]
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "The following required columns are missing:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# 6. TIMESTAMP
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

if df["timestamp"].isna().any():
    raise ValueError(
        "Some timestamp values could not be converted."
    )


df = (
    df
    .sort_values("timestamp")
    .reset_index(drop=True)
)


# ============================================================
# 7. DATASET INFORMATION
# ============================================================

print("\nDate range:")
print(
    df["timestamp"].min(),
    "to",
    df["timestamp"].max()
)

print("\nRows by year:")
print(
    df["timestamp"]
    .dt.year
    .value_counts()
    .sort_index()
)

print("\nAnomaly distribution:")
print(
    df["anomaly"].value_counts()
)


# ============================================================
# 8. CREATE MODEL DATAFRAME
# ============================================================

model_columns = (
    FEATURE_COLUMNS
    + [
        "timestamp",
        "anomaly",
        "anomaly_type"
    ]
)

model_df = df[model_columns].copy()


# ============================================================
# 9. CONVERT FEATURES TO NUMERIC
# ============================================================

for column in FEATURE_COLUMNS:
    model_df[column] = pd.to_numeric(
        model_df[column],
        errors="coerce"
    )


# ============================================================
# 10. REMOVE ROWS WITH MISSING MODEL FEATURES
# ============================================================

before_drop = len(model_df)

model_df = model_df.dropna(
    subset=FEATURE_COLUMNS
).copy()

after_drop = len(model_df)

print("\nRows before missing-feature removal:", before_drop)
print("Rows after missing-feature removal :", after_drop)

print(
    "Rows removed:",
    before_drop - after_drop
)


# ============================================================
# 11. CHECK DATE RANGE AFTER CLEANING
# ============================================================

print("\nModel dataframe date range:")
print(
    model_df["timestamp"].min(),
    "to",
    model_df["timestamp"].max()
)

print("\nModel rows by year:")
print(
    model_df["timestamp"]
    .dt.year
    .value_counts()
    .sort_index()
)


# ============================================================
# 12. TRAINING DATA
# ============================================================
#
# Train only on:
#   - 2020 through 2023
#   - known normal observations
#
# This prevents known synthetic faults from becoming
# part of the model's normal training behaviour.
# ============================================================

train_df = model_df[
    (model_df["timestamp"] < "2024-01-01")
    &
    (model_df["anomaly"] == 0)
].copy()


if train_df.empty:
    raise ValueError(
        "Training dataset is empty."
    )


print("\nTraining dataset shape:")
print(train_df.shape)

print("\nTraining anomaly labels:")
print(
    train_df["anomaly"].value_counts()
)


# ============================================================
# 13. CREATE TRAINING MATRIX
# ============================================================

X_train = train_df[
    FEATURE_COLUMNS
]


print("\nX_train shape:")
print(X_train.shape)


# ============================================================
# 14. CREATE ISOLATION FOREST
# ============================================================

model = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=42
)


# ============================================================
# 15. TRAIN MODEL
# ============================================================

print("\nTraining Isolation Forest...")

model.fit(X_train)

print("Training completed.")


# ============================================================
# 16. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    FEATURE_COLUMNS,
    FEATURE_PATH
)


# ============================================================
# 17. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print("\nModel:")
print(MODEL_PATH)

print("\nFeature list:")
print(FEATURE_PATH)

print("\nNumber of training rows:")
print(len(X_train))

print("\nNumber of features:")
print(len(FEATURE_COLUMNS))