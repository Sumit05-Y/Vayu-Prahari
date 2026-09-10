import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import joblib


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

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "day6"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MODEL_FILE = (
    MODEL_DIR
    / "isolation_forest_final.joblib"
)

FEATURE_FILE = (
    MODEL_DIR
    / "feature_columns_final.joblib"
)

RESULT_FILE = (
    REPORT_DIR
    / "final_2024_results.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "final_2024_summary.csv"
)

TYPE_FILE = (
    REPORT_DIR
    / "final_2024_anomaly_types.csv"
)


# ============================================================
# 2. FINAL FROZEN PARAMETERS
# ============================================================

N_ESTIMATORS = 300
MAX_SAMPLES = 0.6
MAX_FEATURES = 1.0
CONTAMINATION = 0.005

HUMIDITY_THRESHOLD = 10


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 70)
print("VAYU PRAHARI - FINAL MODEL TRAINING")
print("=" * 70)

print("\nDataset:")
print(DATA_FILE)


if not DATA_FILE.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_FILE}"
    )


df = pd.read_csv(
    DATA_FILE
)


# ============================================================
# 4. PREPARE DATA
# ============================================================

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
# 5. EXACT 24 FEATURES
# ============================================================

feature_columns = [
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
# 6. TRAIN FINAL MODEL
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TRAINING: 2020-2023 NORMAL DATA")
print("=" * 70)


train_df = df[
    (df["year"] >= 2020)
    &
    (df["year"] <= 2023)
    &
    (df["anomaly"] == 0)
].copy()


rows_before = len(train_df)


train_df = train_df.dropna(
    subset=feature_columns
).copy()


rows_after = len(train_df)


X_train = train_df[
    feature_columns
]


print(
    "\nTraining rows before dropna:",
    rows_before
)

print(
    "Training rows after dropna:",
    rows_after
)

print(
    "Training shape:",
    X_train.shape
)


# ============================================================
# 7. CREATE FINAL ISOLATION FOREST
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING ISOLATION FOREST")
print("=" * 70)


model = IsolationForest(
    n_estimators=N_ESTIMATORS,
    max_samples=MAX_SAMPLES,
    max_features=MAX_FEATURES,
    contamination=CONTAMINATION,
    random_state=42,
    n_jobs=-1
)


model.fit(
    X_train
)


print("\nFinal model trained successfully.")


# ============================================================
# 8. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)

joblib.dump(
    feature_columns,
    FEATURE_FILE
)


print("\nModel saved:")
print(
    MODEL_FILE
)

print("\nFeature list saved:")
print(
    FEATURE_FILE
)


# ============================================================
# 9. SELECT 2024 TEST DATA
# ============================================================

test_df = df[
    df["year"] == 2024
].copy()

test_df = test_df.reset_index(
    drop=True
)


print("\n")
print("=" * 70)
print("2024 FINAL TEST")
print("=" * 70)

print(
    "\n2024 rows:",
    len(test_df)
)

print("\nActual anomalies:")

print(
    test_df[
        "anomaly"
    ].value_counts()
)


# ============================================================
# 10. ISOLATION FOREST PREDICTION
# ============================================================

valid_rows = (
    test_df[
        feature_columns
    ]
    .notna()
    .all(
        axis=1
    )
)


X_test = (
    test_df.loc[
        valid_rows,
        feature_columns
    ]
)


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
# 11. MISSING TEMPERATURE RULE
# ============================================================

missing_rule = (
    test_df[
        "temperature"
    ].isna()
)


# ============================================================
# 12. FROZEN TEMPERATURE RULE
# ============================================================

def detect_frozen_temperature(
    data,
    min_consecutive=5,
    tolerance=0.05
):

    temperature = (
        data["temperature"]
        .to_numpy(
            dtype=float
        )
    )

    timestamps = (
        data["timestamp"]
        .to_numpy()
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

            time_gap_hours = (
                (
                    current_time
                    - previous_time
                )
                / np.timedelta64(
                    1,
                    "h"
                )
            )

            valid_connection = (
                not np.isnan(
                    previous_temp
                )
                and not np.isnan(
                    current_temp
                )
                and abs(
                    current_temp
                    - previous_temp
                ) <= tolerance
                and time_gap_hours == 1
            )

            if valid_connection:

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
                - np.min(segment)
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
# 13. TEMPERATURE DRIFT RULE
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
        data["temperature"]
        .to_numpy(
            dtype=float
        )
    )

    timestamps = (
        data["timestamp"]
        .to_numpy()
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
            + min_consecutive
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

        time_diffs = (
            np.diff(
                segment_times
            )
            / np.timedelta64(
                1,
                "h"
            )
        )

        if not np.all(
            time_diffs == 1
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

        step_variation = (
            step_sizes.max()
            - step_sizes.min()
        )

        if (
            step_variation
            > max_step_variation
        ):

            continue

        total_change = abs(
            segment[-1]
            - segment[0]
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
            + intercept
        )

        residual_sum = np.sum(
            (
                segment
                - predicted
            ) ** 2
        )

        total_sum = np.sum(
            (
                segment
                - segment.mean()
            ) ** 2
        )

        if total_sum == 0:

            continue

        r2 = (
            1
            - (
                residual_sum
                / total_sum
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
# 14. HUMIDITY RULE
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


prev_gap = (
    (
        test_df[
            "timestamp"
        ]
        -
        test_df[
            "timestamp_prev"
        ]
    )
    .dt.total_seconds()
    / 3600
)


next_gap = (
    (
        test_df[
            "timestamp_next"
        ]
        -
        test_df[
            "timestamp"
        ]
    )
    .dt.total_seconds()
    / 3600
)


valid_neighbors = (
    (prev_gap == 1)
    &
    (next_gap == 1)
)


current_humidity = (
    test_df[
        "humidity"
    ]
)

previous_humidity = (
    test_df[
        "humidity_prev"
    ]
)

next_humidity = (
    test_df[
        "humidity_next"
    ]
)


humidity_spike_rule = (
    valid_neighbors

    & current_humidity.notna()
    & previous_humidity.notna()
    & next_humidity.notna()

    & (
        current_humidity
        >= 99
    )

    & (
        previous_humidity
        < 99
    )

    & (
        next_humidity
        < 99
    )

    & (
        current_humidity
        -
        np.maximum(
            previous_humidity,
            next_humidity
        )
        >= HUMIDITY_THRESHOLD
    )
)


humidity_drop_rule = (
    valid_neighbors

    & current_humidity.notna()
    & previous_humidity.notna()
    & next_humidity.notna()

    & (
        current_humidity
        <= 6
    )

    & (
        previous_humidity
        > 6
    )

    & (
        next_humidity
        > 6
    )

    & (
        np.minimum(
            previous_humidity,
            next_humidity
        )
        -
        current_humidity
        >= HUMIDITY_THRESHOLD
    )
)


humidity_rule = (
    humidity_spike_rule
    |
    humidity_drop_rule
)


# ============================================================
# 15. COMBINE ALL DETECTORS
# ============================================================

rule_prediction = (
    missing_rule
    |
    frozen_rule
    |
    drift_rule
    |
    humidity_rule
).astype(int)


final_prediction = (
    (
        if_full == 1
    )
    |
    (
        rule_prediction == 1
    )
).astype(int)


# ============================================================
# 16. ACTUAL LABELS
# ============================================================

actual = (
    test_df[
        "anomaly"
    ]
    .astype(int)
    .to_numpy()
)


# ============================================================
# 17. METRICS
# ============================================================

accuracy = accuracy_score(
    actual,
    final_prediction
)

precision = precision_score(
    actual,
    final_prediction,
    zero_division=0
)

recall = recall_score(
    actual,
    final_prediction,
    zero_division=0
)

f1 = f1_score(
    actual,
    final_prediction,
    zero_division=0
)


tn, fp, fn, tp = (
    confusion_matrix(
        actual,
        final_prediction
    ).ravel()
)


fpr = (
    fp
    / (
        fp + tn
    )
    if (
        fp + tn
    ) > 0
    else 0
)


# ============================================================
# 18. PRINT FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL VAYU PRAHARI 2024 RESULTS")
print("=" * 70)


print(
    "\nIsolation Forest alerts:",
    int(if_full.sum())
)

print(
    "Rule alerts:",
    int(rule_prediction.sum())
)

print(
    "Final alerts:",
    int(final_prediction.sum())
)


print(
    "\nActual anomalies:",
    int(actual.sum())
)


print(
    "\nTrue Positives:",
    int(tp)
)

print(
    "False Positives:",
    int(fp)
)

print(
    "False Negatives:",
    int(fn)
)

print(
    "True Negatives:",
    int(tn)
)


print(
    "\nAccuracy:",
    f"{accuracy:.6f}"
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
    "False Positive Rate:",
    f"{fpr:.6f}"
)


# ============================================================
# 19. CONFUSION MATRIX
# ============================================================

print("\n")
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(
    f"""
                 Predicted
                 Normal   Anomaly

Actual Normal    {tn:6d}   {fp:7d}
Actual Anomaly  {fn:6d}   {tp:7d}
"""
)


# ============================================================
# 20. ANOMALY TYPE DETECTION
# ============================================================

print("\n")
print("=" * 70)
print("ANOMALY TYPE DETECTION")
print("=" * 70)


anomaly_rows = test_df[
    test_df["anomaly"] == 1
].copy()

anomaly_indices = (
    anomaly_rows.index
)


type_results = []


for anomaly_type, group in (
    anomaly_rows.groupby(
        "anomaly_type"
    )
):

    indices = (
        group.index
    )

    detected_count = int(
        final_prediction[
            indices
        ].sum()
    )

    total_count = len(
        indices
    )


    detection_rate = (
        detected_count
        / total_count
        if total_count > 0
        else 0
    )


    type_results.append(
        {
            "anomaly_type":
                anomaly_type,

            "actual_count":
                total_count,

            "detected_count":
                detected_count,

            "missed_count":
                total_count
                - detected_count,

            "detection_rate":
                detection_rate
        }
    )


type_results_df = pd.DataFrame(
    type_results
)

type_results_df = (
    type_results_df.sort_values(
        "anomaly_type"
    )
    .reset_index(
        drop=True
    )
)


print(
    type_results_df.to_string(
        index=False
    )
)


# ============================================================
# 21. DISPLAY MISSED ANOMALIES
# ============================================================

missed_mask = (
    (
        actual == 1
    )
    &
    (
        final_prediction == 0
    )
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
print("MISSED 2024 ANOMALIES")
print("=" * 70)


if len(missed) == 0:

    print(
        "\nNo anomalies were missed."
    )

else:

    print(
        missed.to_string(
            index=False
        )
    )


# ============================================================
# 22. SAVE DETAILED RESULTS
# ============================================================

test_df["if_anomaly"] = (
    if_full
)

test_df["rule_anomaly"] = (
    rule_prediction
)

test_df["final_anomaly"] = (
    final_prediction
)


test_df["correct_detection"] = (
    (
        test_df["anomaly"] == 1
    )
    &
    (
        test_df["final_anomaly"] == 1
    )
)


test_df["false_positive"] = (
    (
        test_df["anomaly"] == 0
    )
    &
    (
        test_df["final_anomaly"] == 1
    )
)


test_df.to_csv(
    RESULT_FILE,
    index=False
)


# ============================================================
# 23. SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    [
        {
            "test_year": 2024,

            "training_years":
                "2020-2023",

            "n_estimators":
                N_ESTIMATORS,

            "max_samples":
                MAX_SAMPLES,

            "max_features":
                MAX_FEATURES,

            "contamination":
                CONTAMINATION,

            "humidity_threshold":
                HUMIDITY_THRESHOLD,

            "actual_anomalies":
                int(actual.sum()),

            "if_alerts":
                int(if_full.sum()),

            "rule_alerts":
                int(rule_prediction.sum()),

            "final_alerts":
                int(final_prediction.sum()),

            "TP":
                int(tp),

            "FP":
                int(fp),

            "FN":
                int(fn),

            "TN":
                int(tn),

            "accuracy":
                accuracy,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "fpr":
                fpr
        }
    ]
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


type_results_df.to_csv(
    TYPE_FILE,
    index=False
)


# ============================================================
# 24. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("FINAL MODEL PIPELINE COMPLETE")
print("=" * 70)

print("\nSaved files:")

print(
    "\nModel:"
)

print(
    MODEL_FILE
)

print(
    "\nFeatures:"
)

print(
    FEATURE_FILE
)

print(
    "\nDetailed 2024 results:"
)

print(
    RESULT_FILE
)

print(
    "\nSummary:"
)

print(
    SUMMARY_FILE
)

print(
    "\nAnomaly type results:"
)

print(
    TYPE_FILE
)

print("\n")
print("=" * 70)
print("PARAMETERS ARE NOW FROZEN")
print("=" * 70)