import numpy as np
import pandas as pd
import joblib

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "synthetic_anomalies_2020_2024.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "isolation_forest_final.joblib"
)

FEATURE_FILE = (
    PROJECT_ROOT
    / "models"
    / "feature_columns_final.joblib"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "day7"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_FILE = (
    REPORT_DIR
    / "final_detector_v3_2024_results.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "final_detector_v3_2024_summary.csv"
)

TYPE_FILE = (
    REPORT_DIR
    / "final_detector_v3_2024_types.csv"
)


# ============================================================
# 2. LOAD MODEL
# ============================================================

print("=" * 70)
print("VAYU PRAHARI - FINAL DETECTOR V3")
print("=" * 70)

print("\nLoading final frozen model...")

model = joblib.load(
    MODEL_FILE
)

feature_columns = joblib.load(
    FEATURE_FILE
)

print("Model loaded successfully.")


# ============================================================
# 3. LOAD DATA
# ============================================================

df = pd.read_csv(
    DATA_FILE
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

for column in [
    "temperature",
    "humidity",
    "pressure"
]:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df = df.sort_values(
    "timestamp"
).reset_index(
    drop=True
)

df["year"] = (
    df["timestamp"].dt.year
)


# ============================================================
# 4. 2024 TEST DATA
# ============================================================

test_df = df[
    df["year"] == 2024
].copy()

test_df = test_df.reset_index(
    drop=True
)

print("\n2024 rows:", len(test_df))

print("\nActual anomalies:")

print(
    test_df[
        "anomaly"
    ].value_counts()
)


# ============================================================
# 5. ISOLATION FOREST
# ============================================================

valid_rows = (
    test_df[
        feature_columns
    ]
    .notna()
    .all(axis=1)
)

X_test = test_df.loc[
    valid_rows,
    feature_columns
]

raw_prediction = model.predict(
    X_test
)

if_prediction = (
    raw_prediction == -1
).astype(int)

if_full = np.zeros(
    len(test_df),
    dtype=int
)

if_full[
    valid_rows.to_numpy()
] = if_prediction


print(
    "\nIsolation Forest alerts:",
    int(if_full.sum())
)


# ============================================================
# 6. MISSING TEMPERATURE RULE
# ============================================================

missing_rule = (
    test_df[
        "temperature"
    ].isna()
)


# ============================================================
# 7. FROZEN TEMPERATURE RULE
# ============================================================

def detect_frozen_temperature(
    data,
    min_consecutive=5,
    tolerance=0.05
):

    temperature = (
        data[
            "temperature"
        ].to_numpy(
            dtype=float
        )
    )

    timestamps = (
        data[
            "timestamp"
        ].to_numpy()
    )

    n = len(data)

    result = np.zeros(
        n,
        dtype=bool
    )

    run_start = 0

    for i in range(
        1,
        n + 1
    ):

        if i < n:

            previous_temp = (
                temperature[
                    i - 1
                ]
            )

            current_temp = (
                temperature[i]
            )

            previous_time = (
                timestamps[
                    i - 1
                ]
            )

            current_time = (
                timestamps[i]
            )

            gap_hours = (
                (
                    current_time
                    -
                    previous_time
                )
                / np.timedelta64(
                    1,
                    "h"
                )
            )

            connected = (
                not np.isnan(
                    previous_temp
                )
                and not np.isnan(
                    current_temp
                )
                and abs(
                    current_temp
                    -
                    previous_temp
                ) <= tolerance
                and gap_hours == 1
            )

            if connected:
                continue

        segment = temperature[
            run_start:i
        ]

        if (
            len(segment)
            >= min_consecutive
            and not np.isnan(
                segment
            ).any()
        ):

            temperature_range = (
                np.max(segment)
                -
                np.min(segment)
            )

            if (
                temperature_range
                <= tolerance
            ):

                result[
                    run_start:i
                ] = True

        run_start = i

    return result


frozen_rule = (
    detect_frozen_temperature(
        test_df
    )
)


# ============================================================
# 8. TEMPERATURE DRIFT RULE
# ============================================================

def detect_temperature_drift(
    data,
    min_consecutive=8,
    min_step=0.6,
    max_step=1.1,
    min_total_change=5.5,
    max_step_variation=0.25,
    min_r2=0.995
):

    temperature = (
        data[
            "temperature"
        ].to_numpy(
            dtype=float
        )
    )

    timestamps = (
        data[
            "timestamp"
        ].to_numpy()
    )

    n = len(data)

    result = np.zeros(
        n,
        dtype=bool
    )

    for start in range(
        0,
        n - min_consecutive + 1
    ):

        end = (
            start
            +
            min_consecutive
        )

        segment = temperature[
            start:end
        ]

        segment_times = timestamps[
            start:end
        ]

        if np.isnan(
            segment
        ).any():
            continue

        gaps = (
            np.diff(
                segment_times
            )
            / np.timedelta64(
                1,
                "h"
            )
        )

        if not np.all(
            gaps == 1
        ):
            continue

        diffs = np.diff(
            segment
        )

        if len(diffs) == 0:
            continue

        increasing = np.all(
            diffs > 0
        )

        decreasing = np.all(
            diffs < 0
        )

        if not (
            increasing
            or decreasing
        ):
            continue

        step_sizes = np.abs(
            diffs
        )

        if not np.all(
            (
                step_sizes
                >= min_step
            )
            &
            (
                step_sizes
                <= max_step
            )
        ):
            continue

        variation = (
            step_sizes.max()
            -
            step_sizes.min()
        )

        if (
            variation
            > max_step_variation
        ):
            continue

        total_change = abs(
            segment[-1]
            -
            segment[0]
        )

        if (
            total_change
            < min_total_change
        ):
            continue

        x = np.arange(
            len(segment)
        )

        slope, intercept = np.polyfit(
            x,
            segment,
            1
        )

        predicted = (
            slope * x
            +
            intercept
        )

        residual_sum = np.sum(
            (
                segment
                -
                predicted
            ) ** 2
        )

        total_sum = np.sum(
            (
                segment
                -
                segment.mean()
            ) ** 2
        )

        if total_sum == 0:
            continue

        r2 = (
            1
            -
            (
                residual_sum
                /
                total_sum
            )
        )

        if r2 < min_r2:
            continue

        result[
            start:end
        ] = True

    return result


drift_rule = (
    detect_temperature_drift(
        test_df
    )
)


# ============================================================
# 9. HUMIDITY RULE
# ============================================================

test_df["humidity_prev_raw"] = (
    test_df[
        "humidity"
    ].shift(1)
)

test_df["humidity_next_raw"] = (
    test_df[
        "humidity"
    ].shift(-1)
)

test_df["timestamp_prev_raw"] = (
    test_df[
        "timestamp"
    ].shift(1)
)

test_df["timestamp_next_raw"] = (
    test_df[
        "timestamp"
    ].shift(-1)
)


previous_gap = (
    (
        test_df["timestamp"]
        -
        test_df["timestamp_prev_raw"]
    )
    .dt.total_seconds()
    / 3600
)

next_gap = (
    (
        test_df["timestamp_next_raw"]
        -
        test_df["timestamp"]
    )
    .dt.total_seconds()
    / 3600
)

valid_neighbors = (
    (previous_gap == 1)
    &
    (next_gap == 1)
)


current_humidity = (
    test_df["humidity"]
)

previous_humidity = (
    test_df[
        "humidity_prev_raw"
    ]
)

next_humidity = (
    test_df[
        "humidity_next_raw"
    ]
)


humidity_spike_rule = (
    valid_neighbors
    &
    current_humidity.notna()
    &
    previous_humidity.notna()
    &
    next_humidity.notna()
    &
    (current_humidity >= 99)
    &
    (previous_humidity < 99)
    &
    (next_humidity < 99)
    &
    (
        current_humidity
        -
        np.maximum(
            previous_humidity,
            next_humidity
        )
        >= 10
    )
)


humidity_drop_rule = (
    valid_neighbors
    &
    current_humidity.notna()
    &
    previous_humidity.notna()
    &
    next_humidity.notna()
    &
    (current_humidity <= 6)
    &
    (previous_humidity > 6)
    &
    (next_humidity > 6)
    &
    (
        np.minimum(
            previous_humidity,
            next_humidity
        )
        -
        current_humidity
        >= 10
    )
)


humidity_rule = (
    humidity_spike_rule
    |
    humidity_drop_rule
)


# ============================================================
# 10. MULTIVARIATE RULE
#
# LOCKED FROM 2023 VALIDATION
# ============================================================

test_df["temperature_prev_raw"] = (
    test_df[
        "temperature"
    ].shift(1)
)

test_df["temperature_next_raw"] = (
    test_df[
        "temperature"
    ].shift(-1)
)

test_df["pressure_prev_raw"] = (
    test_df[
        "pressure"
    ].shift(1)
)

test_df["pressure_next_raw"] = (
    test_df[
        "pressure"
    ].shift(-1)
)


temperature_expected = (
    (
        test_df[
            "temperature_prev_raw"
        ]
        +
        test_df[
            "temperature_next_raw"
        ]
    )
    / 2
)

humidity_expected = (
    (
        test_df[
            "humidity_prev_raw"
        ]
        +
        test_df[
            "humidity_next_raw"
        ]
    )
    / 2
)

pressure_expected = (
    (
        test_df[
            "pressure_prev_raw"
        ]
        +
        test_df[
            "pressure_next_raw"
        ]
    )
    / 2
)


temperature_residual = (
    (
        test_df[
            "temperature"
        ]
        -
        temperature_expected
    )
    .abs()
)

humidity_residual = (
    (
        test_df[
            "humidity"
        ]
        -
        humidity_expected
    )
    .abs()
)

pressure_residual = (
    (
        test_df[
            "pressure"
        ]
        -
        pressure_expected
    )
    .abs()
)


multivariate_valid = (
    valid_neighbors

    & test_df[
        [
            "temperature_prev_raw",
            "temperature",
            "temperature_next_raw"
        ]
    ]
    .notna()
    .all(axis=1)

    & test_df[
        [
            "humidity_prev_raw",
            "humidity",
            "humidity_next_raw"
        ]
    ]
    .notna()
    .all(axis=1)

    & test_df[
        [
            "pressure_prev_raw",
            "pressure",
            "pressure_next_raw"
        ]
    ]
    .notna()
    .all(axis=1)
)


multivariate_rule = (
    multivariate_valid

    & (
        temperature_residual
        >= 8
    )

    & (
        humidity_residual
        <= 5
    )

    & (
        pressure_residual
        <= 1
    )
)


# ============================================================
# 11. OLD RULE LAYER
# ============================================================

old_rule_prediction = (
    missing_rule
    |
    frozen_rule
    |
    drift_rule
    |
    humidity_rule
).astype(int)


# ============================================================
# 12. FINAL V3 RULE LAYER
# ============================================================

new_rule_prediction = (
    old_rule_prediction
    |
    multivariate_rule
).astype(int)


# ============================================================
# 13. FINAL PREDICTIONS
# ============================================================

v2_prediction = (
    (
        if_full == 1
    )
    |
    (
        old_rule_prediction == 1
    )
).astype(int)


v3_prediction = (
    (
        if_full == 1
    )
    |
    (
        new_rule_prediction == 1
    )
).astype(int)


actual = (
    test_df[
        "anomaly"
    ]
    .astype(int)
    .to_numpy()
)


# ============================================================
# 14. METRIC FUNCTION
# ============================================================

def calculate_metrics(
    actual,
    prediction
):

    accuracy = accuracy_score(
        actual,
        prediction
    )

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

    tn, fp, fn, tp = (
        confusion_matrix(
            actual,
            prediction
        ).ravel()
    )

    fpr = (
        fp /
        (
            fp + tn
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn)
    }


v2_metrics = calculate_metrics(
    actual,
    v2_prediction
)

v3_metrics = calculate_metrics(
    actual,
    v3_prediction
)


# ============================================================
# 15. PRINT COMPARISON
# ============================================================

print("\n")
print("=" * 70)
print("FINAL V2 VS V3 - 2024")
print("=" * 70)

print(
    "\nMetric                V2          V3"
)

print(
    f"Accuracy          "
    f"{v2_metrics['accuracy']:.6f}    "
    f"{v3_metrics['accuracy']:.6f}"
)

print(
    f"Precision         "
    f"{v2_metrics['precision']:.6f}    "
    f"{v3_metrics['precision']:.6f}"
)

print(
    f"Recall            "
    f"{v2_metrics['recall']:.6f}    "
    f"{v3_metrics['recall']:.6f}"
)

print(
    f"F1                "
    f"{v2_metrics['f1']:.6f}    "
    f"{v3_metrics['f1']:.6f}"
)

print(
    f"FPR               "
    f"{v2_metrics['fpr']:.6f}    "
    f"{v3_metrics['fpr']:.6f}"
)


print("\n")
print("=" * 70)
print("V3 CONFUSION MATRIX")
print("=" * 70)

print(
    f"""
                 Predicted
                 Normal   Anomaly

Actual Normal      {v3_metrics['TN']:6d}   {v3_metrics['FP']:7d}
Actual Anomaly     {v3_metrics['FN']:6d}   {v3_metrics['TP']:7d}
"""
)


# ============================================================
# 16. ALERT COUNTS
# ============================================================

print("=" * 70)
print("ALERT COUNTS")
print("=" * 70)

print(
    "\nIsolation Forest:",
    int(if_full.sum())
)

print(
    "Old rules:",
    int(old_rule_prediction.sum())
)

print(
    "Multivariate rule:",
    int(multivariate_rule.sum())
)

print(
    "V2 final alerts:",
    int(v2_prediction.sum())
)

print(
    "V3 final alerts:",
    int(v3_prediction.sum())
)


# ============================================================
# 17. TYPE DETECTION
# ============================================================

anomaly_rows = test_df[
    test_df["anomaly"] == 1
].copy()


type_results = []


for anomaly_type, group in (
    anomaly_rows.groupby(
        "anomaly_type"
    )
):

    indices = group.index

    v2_detected = int(
        v2_prediction[
            indices
        ].sum()
    )

    v3_detected = int(
        v3_prediction[
            indices
        ].sum()
    )

    total = len(
        indices
    )

    type_results.append(
        {
            "anomaly_type":
                anomaly_type,

            "actual_count":
                total,

            "v2_detected":
                v2_detected,

            "v3_detected":
                v3_detected,

            "v2_missed":
                total - v2_detected,

            "v3_missed":
                total - v3_detected,

            "v3_detection_rate":
                (
                    v3_detected
                    / total
                )
        }
    )


type_results_df = pd.DataFrame(
    type_results
)


print("\n")
print("=" * 70)
print("ANOMALY TYPE COMPARISON")
print("=" * 70)

print(
    type_results_df.to_string(
        index=False
    )
)


# ============================================================
# 18. MISSED V3 ANOMALIES
# ============================================================

missed_mask = (
    (actual == 1)
    &
    (v3_prediction == 0)
)


missed = test_df.loc[
    missed_mask,
    [
        "timestamp",
        "temperature",
        "humidity",
        "pressure",
        "anomaly_type"
    ]
]


print("\n")
print("=" * 70)
print("MISSED V3 ANOMALIES")
print("=" * 70)

if len(missed) == 0:

    print(
        "\nNo anomalies missed."
    )

else:

    print(
        missed.to_string(
            index=False
        )
    )


# ============================================================
# 19. SAVE DETAILED RESULTS
# ============================================================

test_df["if_anomaly"] = (
    if_full
)

test_df["v2_rule_anomaly"] = (
    old_rule_prediction
)

test_df["multivariate_anomaly"] = (
    multivariate_rule.astype(int)
)

test_df["v2_final_anomaly"] = (
    v2_prediction
)

test_df["v3_final_anomaly"] = (
    v3_prediction
)


test_df.to_csv(
    RESULT_FILE,
    index=False
)


# ============================================================
# 20. SAVE SUMMARY
# ============================================================

summary = pd.DataFrame(
    [
        {
            "model": "V2",
            **v2_metrics
        },
        {
            "model": "V3",
            **v3_metrics
        }
    ]
)


summary.to_csv(
    SUMMARY_FILE,
    index=False
)

type_results_df.to_csv(
    TYPE_FILE,
    index=False
)


# ============================================================
# 21. COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("V3 EVALUATION COMPLETE")
print("=" * 70)

print("\nSaved:")

print(
    RESULT_FILE
)

print(
    SUMMARY_FILE
)

print(
    TYPE_FILE
)