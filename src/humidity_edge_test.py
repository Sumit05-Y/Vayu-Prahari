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
# 1. PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "synthetic_anomalies_2020_2024.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df["humidity"] = pd.to_numeric(
    df["humidity"],
    errors="coerce"
)

df["anomaly"] = df["anomaly"].astype(int)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)

df["year"] = df["timestamp"].dt.year


# ============================================================
# 3. SELECT 2024
# ============================================================

test_df = df[
    df["year"] == 2024
].copy()

test_df = test_df.reset_index(
    drop=True
)


# ============================================================
# 4. PREVIOUS / NEXT VALUES
# ============================================================

test_df["humidity_prev"] = (
    test_df["humidity"].shift(1)
)

test_df["humidity_next"] = (
    test_df["humidity"].shift(-1)
)

test_df["timestamp_prev"] = (
    test_df["timestamp"].shift(1)
)

test_df["timestamp_next"] = (
    test_df["timestamp"].shift(-1)
)


prev_gap = (
    test_df["timestamp"]
    - test_df["timestamp_prev"]
).dt.total_seconds() / 3600

next_gap = (
    test_df["timestamp_next"]
    - test_df["timestamp"]
).dt.total_seconds() / 3600


valid_neighbors = (
    (prev_gap == 1)
    &
    (next_gap == 1)
)


h = test_df["humidity"]
prev = test_df["humidity_prev"]
next_h = test_df["humidity_next"]


# ============================================================
# 5. TEST DIFFERENT EDGE-SPIKE RULES
# ============================================================

rules = {

    "Rule A": (
        valid_neighbors
        &
        (h == 100)
        &
        (prev >= 95)
        &
        (next_h <= 97)
    ),

    "Rule B": (
        valid_neighbors
        &
        (h == 100)
        &
        (prev >= 95)
        &
        (next_h < 100)
        &
        ((prev - next_h) >= 2)
    ),

    "Rule C": (
        valid_neighbors
        &
        (h >= 99)
        &
        (prev >= 95)
        &
        (next_h <= 97)
        &
        ((h - next_h) >= 2)
    )
}


# ============================================================
# 6. EVALUATE EACH RULE
# ============================================================

print("=" * 70)
print("HUMIDITY EDGE-SPIKE TEST")
print("=" * 70)

actual_humidity = (
    test_df["anomaly_type"]
    .isin(
        [
            "humidity_spike",
            "humidity_drop"
        ]
    )
    .astype(int)
)


for name, rule in rules.items():

    prediction = rule.astype(int)

    actual = (
        test_df["anomaly"]
        .astype(int)
    )

    cm = confusion_matrix(
        actual,
        prediction
    )

    tn, fp, fn, tp = cm.ravel()

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
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    humidity_precision = precision_score(
        actual_humidity,
        prediction,
        zero_division=0
    )

    humidity_recall = recall_score(
        actual_humidity,
        prediction,
        zero_division=0
    )

    humidity_f1 = f1_score(
        actual_humidity,
        prediction,
        zero_division=0
    )

    print("\n" + "-" * 70)
    print(name)

    print(
        "\nPredicted alerts:",
        int(prediction.sum())
    )

    print(
        "TP:",
        tp
    )

    print(
        "FP:",
        fp
    )

    print(
        "FN:",
        fn
    )

    print(
        "TN:",
        tn
    )

    print(
        "Precision:",
        f"{precision:.6f}"
    )

    print(
        "Recall:",
        f"{recall:.6f}"
    )

    print(
        "F1:",
        f"{f1:.6f}"
    )

    print(
        "FPR:",
        f"{fpr:.6f}"
    )

    print(
        "Humidity recall:",
        f"{humidity_recall:.6f}"
    )

    print(
        "Humidity F1:",
        f"{humidity_f1:.6f}"
    )


# ============================================================
# 7. SHOW WHAT EACH RULE DETECTS
# ============================================================

print("\n")
print("=" * 70)
print("ACTUAL HUMIDITY ANOMALIES")
print("=" * 70)

print(
    test_df[
        test_df["anomaly_type"].isin(
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
    ].to_string(
        index=False
    )
)


# ============================================================
# 8. SHOW RULE DETECTIONS
# ============================================================

for name, rule in rules.items():

    print("\n")
    print("=" * 70)
    print(name, "- DETECTED RECORDS")
    print("=" * 70)

    detected = test_df[
        rule
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

    if len(detected) == 0:

        print(
            "No records detected."
        )

    else:

        print(
            detected.to_string(
                index=False
            )
        )