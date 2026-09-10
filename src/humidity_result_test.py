import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "synthetic_anomalies_2020_2024.csv"
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 70)
print("VAYU PRAHARI - HUMIDITY RULE TEST")
print("=" * 70)

print("\nDataset:")
print(DATA_FILE)


if not DATA_FILE.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_FILE}"
    )


df = pd.read_csv(
    DATA_FILE
)


# ============================================================
# 3. PREPARE DATA
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

df["humidity"] = pd.to_numeric(
    df["humidity"],
    errors="coerce"
)

df["anomaly"] = pd.to_numeric(
    df["anomaly"],
    errors="coerce"
).astype(int)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


df["year"] = (
    df["timestamp"].dt.year
)


# ============================================================
# 4. USE 2024 ONLY
# ============================================================

test_df = df[
    df["year"] == 2024
].copy()

test_df = test_df.reset_index(
    drop=True
)


print("\n2024 rows:")
print(len(test_df))

print("\nActual anomaly counts:")

print(
    test_df[
        "anomaly"
    ].value_counts()
)

print("\nActual anomaly types:")

print(
    test_df[
        test_df["anomaly"] == 1
    ]["anomaly_type"].value_counts()
)


# ============================================================
# 5. CREATE PREVIOUS / NEXT HUMIDITY
# ============================================================

test_df["humidity_prev"] = (
    test_df[
        "humidity"
    ].shift(1)
)

test_df["humidity_next"] = (
    test_df[
        "humidity"
    ].shift(-1)
)

test_df["timestamp_prev"] = (
    test_df[
        "timestamp"
    ].shift(1)
)

test_df["timestamp_next"] = (
    test_df[
        "timestamp"
    ].shift(-1)
)


# ============================================================
# 6. CHECK NEIGHBOR TIME GAPS
# ============================================================

prev_gap_hours = (
    (
        test_df["timestamp"]
        - test_df["timestamp_prev"]
    )
    .dt.total_seconds()
    / 3600
)

next_gap_hours = (
    (
        test_df["timestamp_next"]
        - test_df["timestamp"]
    )
    .dt.total_seconds()
    / 3600
)


valid_neighbors = (
    (prev_gap_hours == 1)
    &
    (next_gap_hours == 1)
)


# ============================================================
# 7. TEST MULTIPLE HUMIDITY JUMP THRESHOLDS
# ============================================================

thresholds = [
    5,
    10,
    15,
    20,
    25
]


results = []


for jump_threshold in thresholds:

    current = (
        test_df["humidity"]
    )

    previous = (
        test_df["humidity_prev"]
    )

    next_value = (
        test_df["humidity_next"]
    )


    # ========================================================
    # HUMIDITY SPIKE
    # ========================================================

    humidity_spike = (
        valid_neighbors

        & current.notna()
        & previous.notna()
        & next_value.notna()

        & (current >= 99)

        & (previous < 99)

        & (next_value < 99)

        & (
            current
            - np.maximum(
                previous,
                next_value
            )
            >= jump_threshold
        )
    )


    # ========================================================
    # HUMIDITY DROP
    # ========================================================

    humidity_drop = (
        valid_neighbors

        & current.notna()
        & previous.notna()
        & next_value.notna()

        & (current <= 6)

        & (previous > 6)

        & (next_value > 6)

        & (
            np.minimum(
                previous,
                next_value
            )
            - current
            >= jump_threshold
        )
    )


    # ========================================================
    # FINAL HUMIDITY RULE
    # ========================================================

    prediction = (
        humidity_spike
        |
        humidity_drop
    ).astype(int)


    actual = (
        test_df["anomaly"]
        .astype(int)
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        actual,
        prediction
    )


    tn, fp, fn, tp = (
        cm.ravel()
    )


    # ========================================================
    # METRICS - ALL ANOMALIES
    # ========================================================

    precision = precision_score(
        actual,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        actual,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        actual,
        prediction,
        zero_division=0
    )

    fpr = (
        fp
        / (
            fp
            + tn
        )
        if (
            fp
            + tn
        ) > 0
        else 0
    )


    # ========================================================
    # HUMIDITY ANOMALY DETECTION
    # ========================================================

    humidity_actual = (
        test_df[
            "anomaly_type"
        ].isin(
            [
                "humidity_spike",
                "humidity_drop"
            ]
        ).astype(int)
    )


    humidity_precision = precision_score(
        humidity_actual,
        prediction,
        zero_division=0
    )

    humidity_recall = recall_score(
        humidity_actual,
        prediction,
        zero_division=0
    )

    humidity_f1 = f1_score(
        humidity_actual,
        prediction,
        zero_division=0
    )


    # ========================================================
    # STORE RESULTS
    # ========================================================

    results.append(
        {
            "jump_threshold":
                jump_threshold,

            "predicted_anomalies":
                int(prediction.sum()),

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "fpr":
                fpr,

            "TP":
                int(tp),

            "FP":
                int(fp),

            "FN":
                int(fn),

            "TN":
                int(tn),

            "humidity_precision":
                humidity_precision,

            "humidity_recall":
                humidity_recall,

            "humidity_f1":
                humidity_f1
        }
    )


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n")
print("=" * 70)
print("HUMIDITY RULE THRESHOLD RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 9. SHOW ACTUAL HUMIDITY ANOMALIES
# ============================================================

print("\n")
print("=" * 70)
print("ACTUAL HUMIDITY ANOMALIES")
print("=" * 70)


humidity_anomalies = test_df[
    test_df[
        "anomaly_type"
    ].isin(
        [
            "humidity_spike",
            "humidity_drop"
        ]
    )
][
    [
        "timestamp",
        "humidity_prev",
        "humidity",
        "humidity_next",
        "anomaly_type"
    ]
]


print(
    humidity_anomalies.to_string(
        index=False
    )
)


# ============================================================
# 10. TEST SELECTED THRESHOLD IN DETAIL
# ============================================================

selected_threshold = 10


print("\n")
print("=" * 70)
print(
    f"DETAILED TEST - THRESHOLD {selected_threshold}"
)
print("=" * 70)


current = (
    test_df["humidity"]
)

previous = (
    test_df["humidity_prev"]
)

next_value = (
    test_df["humidity_next"]
)


humidity_spike = (
    valid_neighbors

    & current.notna()
    & previous.notna()
    & next_value.notna()

    & (current >= 99)

    & (previous < 99)

    & (next_value < 99)

    & (
        current
        - np.maximum(
            previous,
            next_value
        )
        >= selected_threshold
    )
)


humidity_drop = (
    valid_neighbors

    & current.notna()
    & previous.notna()
    & next_value.notna()

    & (current <= 6)

    & (previous > 6)

    & (next_value > 6)

    & (
        np.minimum(
            previous,
            next_value
        )
        - current
        >= selected_threshold
    )
)


selected_prediction = (
    humidity_spike
    |
    humidity_drop
).astype(int)


detected = test_df[
    selected_prediction == 1
][
    [
        "timestamp",
        "humidity_prev",
        "humidity",
        "humidity_next",
        "anomaly",
        "anomaly_type"
    ]
]


print(
    detected.to_string(
        index=False
    )
)


# ============================================================
# 11. SUCCESS
# ============================================================

print("\n")
print("=" * 70)
print("HUMIDITY RULE TEST COMPLETE")
print("=" * 70)