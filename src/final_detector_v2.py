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
# 1. PROJECT PATHS
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
    / "isolation_forest_v2.joblib"
)

FEATURE_FILE = (
    PROJECT_ROOT
    / "models"
    / "feature_columns_v2.joblib"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "day6"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_FILE = (
    REPORT_DIR
    / "day6_final_detector_v2_2024_results.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "day6_final_detector_v2_summary.csv"
)

TYPE_RESULT_FILE = (
    REPORT_DIR
    / "day6_final_detector_v2_anomaly_types.csv"
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 70)
print("VAYU PRAHARI - FINAL ANOMALY DETECTOR V2")
print("=" * 70)

print("\nDataset:")
print(DATA_FILE)

print("\nIsolation Forest:")
print(MODEL_FILE)


if not DATA_FILE.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_FILE}"
    )


if not MODEL_FILE.exists():

    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_FILE}"
    )


if not FEATURE_FILE.exists():

    raise FileNotFoundError(
        f"\nFeature list not found:\n{FEATURE_FILE}"
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

df["temperature"] = pd.to_numeric(
    df["temperature"],
    errors="coerce"
)

df["humidity"] = pd.to_numeric(
    df["humidity"],
    errors="coerce"
)

df["pressure"] = pd.to_numeric(
    df["pressure"],
    errors="coerce"
)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)

df["year"] = (
    df["timestamp"].dt.year
)


print("\nDataset shape:")
print(df.shape)

print("\nDate range:")
print(
    df["timestamp"].min(),
    "to",
    df["timestamp"].max()
)


# ============================================================
# 4. LOAD ISOLATION FOREST
# ============================================================

print("\n")
print("=" * 70)
print("LOADING ISOLATION FOREST V2")
print("=" * 70)

model = joblib.load(
    MODEL_FILE
)

feature_columns = joblib.load(
    FEATURE_FILE
)

print("\nNumber of features:")
print(len(feature_columns))


# ============================================================
# 5. SELECT 2024 TEST DATA
# ============================================================

test_df = df[
    df["year"] == 2024
].copy()

test_df = test_df.reset_index(
    drop=True
)


print("\n")
print("=" * 70)
print("2024 TEST DATA")
print("=" * 70)

print("\nRows:")
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
# 6. ISOLATION FOREST
# ============================================================

test_df["if_anomaly"] = 0

test_df["if_score"] = np.nan


valid_if_rows = (
    test_df[
        feature_columns
    ]
    .notna()
    .all(axis=1)
)


X_test_if = (
    test_df.loc[
        valid_if_rows,
        feature_columns
    ]
)


print("\nRows usable by Isolation Forest:")
print(len(X_test_if))


print("\n")
print("=" * 70)
print("RUNNING ISOLATION FOREST")
print("=" * 70)


if len(X_test_if) > 0:

    raw_prediction = model.predict(
        X_test_if
    )

    if_prediction = (
        raw_prediction == -1
    ).astype(int)

    if_score = model.decision_function(
        X_test_if
    )

    test_df.loc[
        valid_if_rows,
        "if_anomaly"
    ] = if_prediction

    test_df.loc[
        valid_if_rows,
        "if_score"
    ] = if_score


print("\nIsolation Forest alerts:")
print(
    int(
        test_df[
            "if_anomaly"
        ].sum()
    )
)


# ============================================================
# 7. RULE - MISSING TEMPERATURE
# ============================================================

missing_rule = (
    test_df[
        "temperature"
    ]
    .isna()
)


# ============================================================
# 8. RULE - FROZEN TEMPERATURE
# ============================================================

def detect_frozen_temperature(
    data,
    min_consecutive=5,
    tolerance=0.05
):

    temperature = (
        data["temperature"]
        .to_numpy(dtype=float)
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
                temperature[i - 1]
            )

            current_temp = (
                temperature[i]
            )

            previous_time = (
                timestamps[i - 1]
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

        segment_length = len(
            segment
        )

        if (
            segment_length
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
        test_df,
        min_consecutive=5,
        tolerance=0.05
    )
)


# ============================================================
# 9. RULE - TEMPERATURE DRIFT
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
        .to_numpy(dtype=float)
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
            np.diff(segment_times)
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
        test_df,
        min_consecutive=8,
        min_step=0.6,
        max_step=1.1,
        min_total_change=5.5,
        max_step_variation=0.25,
        min_r2=0.995
    )
)


# ============================================================
# 10. RULE - HUMIDITY SPIKE / DROP
# ============================================================

HUMIDITY_JUMP_THRESHOLD = 10


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


prev_gap_hours = (
    (
        test_df[
            "timestamp"
        ]
        - test_df[
            "timestamp_prev"
        ]
    )
    .dt.total_seconds()
    / 3600
)

next_gap_hours = (
    (
        test_df[
            "timestamp_next"
        ]
        - test_df[
            "timestamp"
        ]
    )
    .dt.total_seconds()
    / 3600
)


valid_humidity_neighbors = (
    (prev_gap_hours == 1)
    &
    (next_gap_hours == 1)
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


# ------------------------------------------------------------
# Humidity spike
# ------------------------------------------------------------

humidity_spike_rule = (
    valid_humidity_neighbors

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
        - np.maximum(
            previous_humidity,
            next_humidity
        )
        >= HUMIDITY_JUMP_THRESHOLD
    )
)


# ------------------------------------------------------------
# Humidity drop
# ------------------------------------------------------------

humidity_drop_rule = (
    valid_humidity_neighbors

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
        - current_humidity
        >= HUMIDITY_JUMP_THRESHOLD
    )
)


humidity_rule = (
    humidity_spike_rule
    |
    humidity_drop_rule
)


# ============================================================
# 11. STORE RULE RESULTS
# ============================================================

test_df["rule_missing_temperature"] = (
    missing_rule.astype(int)
)

test_df["rule_temp_frozen"] = (
    frozen_rule.astype(int)
)

test_df["rule_temp_drift"] = (
    drift_rule.astype(int)
)

test_df["rule_humidity_spike"] = (
    humidity_spike_rule.astype(int)
)

test_df["rule_humidity_drop"] = (
    humidity_drop_rule.astype(int)
)

test_df["rule_humidity_anomaly"] = (
    humidity_rule.astype(int)
)


# ============================================================
# 12. COMBINE ALL SENSOR RULES
# ============================================================

test_df["rule_anomaly"] = (
    missing_rule
    |
    frozen_rule
    |
    drift_rule
    |
    humidity_rule
).astype(int)


print("\n")
print("=" * 70)
print("TEMPORAL + SENSOR RULE RESULTS")
print("=" * 70)

print(
    "\nMissing temperature alerts:",
    int(
        test_df[
            "rule_missing_temperature"
        ].sum()
    )
)

print(
    "Frozen temperature alerts:",
    int(
        test_df[
            "rule_temp_frozen"
        ].sum()
    )
)

print(
    "Temperature drift alerts:",
    int(
        test_df[
            "rule_temp_drift"
        ].sum()
    )
)

print(
    "Humidity alerts:",
    int(
        test_df[
            "rule_humidity_anomaly"
        ].sum()
    )
)

print(
    "Total rule alerts:",
    int(
        test_df[
            "rule_anomaly"
        ].sum()
    )
)


# ============================================================
# 13. RULE ANOMALY TYPE
# ============================================================

test_df["rule_anomaly_type"] = np.select(
    [
        test_df[
            "rule_missing_temperature"
        ] == 1,

        test_df[
            "rule_temp_frozen"
        ] == 1,

        test_df[
            "rule_temp_drift"
        ] == 1,

        test_df[
            "rule_humidity_spike"
        ] == 1,

        test_df[
            "rule_humidity_drop"
        ] == 1
    ],
    [
        "missing_temperature",
        "temp_frozen",
        "temp_drift",
        "humidity_spike",
        "humidity_drop"
    ],
    default="normal"
)


# ============================================================
# 14. FINAL COMBINATION
# ============================================================

test_df["final_anomaly"] = (
    (
        test_df[
            "if_anomaly"
        ] == 1
    )
    |
    (
        test_df[
            "rule_anomaly"
        ] == 1
    )
).astype(int)


# ============================================================
# 15. FINAL ANOMALY TYPE
# ============================================================

test_df["final_anomaly_type"] = np.select(
    [
        test_df[
            "rule_missing_temperature"
        ] == 1,

        test_df[
            "rule_temp_frozen"
        ] == 1,

        test_df[
            "rule_temp_drift"
        ] == 1,

        test_df[
            "rule_humidity_spike"
        ] == 1,

        test_df[
            "rule_humidity_drop"
        ] == 1,

        test_df[
            "if_anomaly"
        ] == 1
    ],
    [
        "missing_temperature",
        "temp_frozen",
        "temp_drift",
        "humidity_spike",
        "humidity_drop",
        "isolation_forest_anomaly"
    ],
    default="normal"
)


# ============================================================
# 16. EVALUATE FINAL DETECTOR
# ============================================================

y_true = (
    test_df[
        "anomaly"
    ].astype(int)
)

y_pred = (
    test_df[
        "final_anomaly"
    ].astype(int)
)


cm = confusion_matrix(
    y_true,
    y_pred
)

tn, fp, fn, tp = (
    cm.ravel()
)


accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
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


# ============================================================
# 17. FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL COMBINED DETECTOR V2")
print("=" * 70)

print(
    "\nAccuracy :",
    f"{accuracy:.6f}"
)

print(
    "Precision:",
    f"{precision:.6f}"
)

print(
    "Recall   :",
    f"{recall:.6f}"
)

print(
    "F1 Score :",
    f"{f1:.6f}"
)

print(
    "FPR      :",
    f"{fpr:.6f}"
)

print(
    "\nActual anomalies:",
    int(y_true.sum())
)

print(
    "Final predicted anomalies:",
    int(y_pred.sum())
)

print(
    "True Positives:",
    tp
)

print(
    "False Positives:",
    fp
)

print(
    "False Negatives:",
    fn
)

print(
    "True Negatives:",
    tn
)

print("\nConfusion matrix:")
print(cm)


# ============================================================
# 18. FINAL ANOMALY TYPE DETECTION
# ============================================================

print("\n")
print("=" * 70)
print("FINAL ANOMALY TYPE DETECTION")
print("=" * 70)


type_results = []

actual_types = sorted(
    test_df[
        test_df[
            "anomaly"
        ] == 1
    ][
        "anomaly_type"
    ].unique()
)


for anomaly_type in actual_types:

    type_df = test_df[
        test_df[
            "anomaly_type"
        ]
        == anomaly_type
    ]

    actual_count = len(
        type_df
    )

    detected_count = int(
        type_df[
            "final_anomaly"
        ].sum()
    )

    missed_count = (
        actual_count
        - detected_count
    )

    detection_rate = (
        detected_count
        / actual_count
        if actual_count > 0
        else 0
    )

    type_results.append(
        {
            "anomaly_type":
                anomaly_type,

            "actual_count":
                actual_count,

            "detected_count":
                detected_count,

            "missed_count":
                missed_count,

            "detection_rate":
                detection_rate
        }
    )


type_results_df = pd.DataFrame(
    type_results
)


print(
    type_results_df.to_string(
        index=False
    )
)


# ============================================================
# 19. COMPONENT SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("DETECTION COMPONENT SUMMARY")
print("=" * 70)

print(
    "\nIsolation Forest alerts:",
    int(
        test_df[
            "if_anomaly"
        ].sum()
    )
)

print(
    "Sensor-rule alerts:",
    int(
        test_df[
            "rule_anomaly"
        ].sum()
    )
)

print(
    "Final alerts:",
    int(
        test_df[
            "final_anomaly"
        ].sum()
    )
)


# ============================================================
# 20. MISSED ANOMALIES
# ============================================================

print("\n")
print("=" * 70)
print("MISSED ANOMALIES")
print("=" * 70)


missed = test_df[
    (
        test_df[
            "anomaly"
        ] == 1
    )
    &
    (
        test_df[
            "final_anomaly"
        ] == 0
    )
][
    [
        "timestamp",
        "temperature",
        "humidity",
        "pressure",
        "anomaly_type",
        "if_anomaly",
        "rule_anomaly",
        "final_anomaly"
    ]
]


if len(missed) == 0:

    print(
        "No anomalies were missed."
    )

else:

    print(
        missed.to_string(
            index=False
        )
    )


# ============================================================
# 21. SAVE RESULTS
# ============================================================

result_columns = [
    "timestamp",
    "temperature",
    "humidity",
    "pressure",

    "anomaly",
    "anomaly_type",

    "if_anomaly",
    "if_score",

    "rule_missing_temperature",
    "rule_temp_frozen",
    "rule_temp_drift",

    "rule_humidity_spike",
    "rule_humidity_drop",
    "rule_humidity_anomaly",

    "rule_anomaly",
    "rule_anomaly_type",

    "final_anomaly",
    "final_anomaly_type"
]


test_df[
    result_columns
].to_csv(
    RESULT_FILE,
    index=False
)


# ============================================================
# 22. SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    [
        {
            "detector":
                "Isolation Forest V2 + Temporal Rules + Humidity Rule",

            "test_year":
                2024,

            "humidity_threshold":
                HUMIDITY_JUMP_THRESHOLD,

            "test_rows":
                len(test_df),

            "actual_anomalies":
                int(y_true.sum()),

            "predicted_anomalies":
                int(y_pred.sum()),

            "accuracy":
                accuracy,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "fpr":
                fpr,

            "TP":
                tp,

            "FP":
                fp,

            "FN":
                fn,

            "TN":
                tn
        }
    ]
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# 23. SAVE ANOMALY TYPE RESULTS
# ============================================================

type_results_df.to_csv(
    TYPE_RESULT_FILE,
    index=False
)


# ============================================================
# 24. SUCCESS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL DETECTOR V2 COMPLETE")
print("=" * 70)

print("\nResults saved:")
print(
    RESULT_FILE
)

print("\nSummary saved:")
print(
    SUMMARY_FILE
)

print("\nAnomaly type results saved:")
print(
    TYPE_RESULT_FILE
)