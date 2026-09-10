import numpy as np
import pandas as pd

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
    / "day6_sensor_rule_2024_results.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "day6_sensor_rule_summary.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("VAYU PRAHARI - TEMPORAL SENSOR RULES")
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
# 4. RULE 1 - MISSING TEMPERATURE
# ============================================================

def detect_missing_temperature(data):

    return (
        data["temperature"]
        .isna()
    )


missing_rule = (
    detect_missing_temperature(df)
)


# ============================================================
# 5. RULE 2 - FROZEN / STUCK TEMPERATURE SENSOR
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

        # ----------------------------------------------------
        # Continue checking current sequence
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # End of current sequence
        # ----------------------------------------------------

        segment = temperature[
            run_start:i
        ]

        segment_length = len(
            segment
        )

        # ----------------------------------------------------
        # Check if sequence is long enough
        # ----------------------------------------------------

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

            # ------------------------------------------------
            # Nearly identical readings over many hours
            # ------------------------------------------------

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
        df,
        min_consecutive=5,
        tolerance=0.05
    )
)


# ============================================================
# 6. RULE 3 - TEMPERATURE DRIFT
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

    # ========================================================
    # Examine rolling windows
    # ========================================================

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

        # ----------------------------------------------------
        # Missing values invalidate the candidate sequence
        # ----------------------------------------------------

        if np.isnan(
            segment
        ).any():

            continue

        # ----------------------------------------------------
        # Readings must be exactly one hour apart
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Temperature differences
        # ----------------------------------------------------

        diffs = np.diff(
            segment
        )

        if len(diffs) == 0:

            continue

        # ----------------------------------------------------
        # Must consistently move in one direction
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Size of each temperature change
        # ----------------------------------------------------

        step_sizes = np.abs(
            diffs
        )

        # ----------------------------------------------------
        # Every step must be in a narrow range
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Sensor drift should be highly consistent
        # ----------------------------------------------------

        step_variation = (
            step_sizes.max()
            - step_sizes.min()
        )

        if (
            step_variation
            > max_step_variation
        ):

            continue

        # ----------------------------------------------------
        # Total temperature change
        # ----------------------------------------------------

        total_change = abs(
            segment[-1]
            - segment[0]
        )

        if (
            total_change
            < min_total_change
        ):

            continue

        # ----------------------------------------------------
        # Linear-fit check
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Sequence must be almost perfectly linear
        # ----------------------------------------------------

        if (
            r2
            < min_r2
        ):

            continue

        # ----------------------------------------------------
        # Candidate is considered sensor drift
        # ----------------------------------------------------

        result[
            start:end
        ] = True

    return result


drift_rule = (
    detect_temperature_drift(
        df,

        min_consecutive=8,

        min_step=0.6,

        max_step=1.1,

        min_total_change=5.5,

        max_step_variation=0.25,

        min_r2=0.995
    )
)


# ============================================================
# 7. COMBINE TEMPORAL RULES
# ============================================================

df["rule_missing_temperature"] = (
    missing_rule.astype(int)
)

df["rule_temp_frozen"] = (
    frozen_rule.astype(int)
)

df["rule_temp_drift"] = (
    drift_rule.astype(int)
)


df["rule_anomaly"] = (
    missing_rule
    | frozen_rule
    | drift_rule
).astype(int)


df["rule_anomaly_type"] = np.select(
    [
        missing_rule,
        frozen_rule,
        drift_rule
    ],
    [
        "missing_temperature",
        "temp_frozen",
        "temp_drift"
    ],
    default="normal"
)


# ============================================================
# 8. TEST 2024
# ============================================================

test_df = df[
    df["year"] == 2024
].copy()


print("\n")
print("=" * 70)
print("2024 TEST DATA")
print("=" * 70)

print("\nRows:")
print(
    len(test_df)
)


print("\nActual anomaly counts:")

print(
    test_df[
        "anomaly"
    ].value_counts()
)


print("\nActual anomaly types:")

print(
    test_df[
        "anomaly_type"
    ].value_counts()
)


# ============================================================
# 9. CREATE PREDICTIONS
# ============================================================

y_true = (
    test_df[
        "anomaly"
    ]
    .astype(int)
)

y_pred = (
    test_df[
        "rule_anomaly"
    ]
    .astype(int)
)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

tn, fp, fn, tp = (
    cm.ravel()
)


# ============================================================
# 11. METRICS
# ============================================================

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
# 12. PRINT OVERALL RULE PERFORMANCE
# ============================================================

print("\n")
print("=" * 70)
print("RULE LAYER - ALL ANOMALIES")
print("=" * 70)

print(
    f"\nAccuracy : {accuracy:.6f}"
)

print(
    f"Precision: {precision:.6f}"
)

print(
    f"Recall   : {recall:.6f}"
)

print(
    f"F1 Score : {f1:.6f}"
)

print(
    f"FPR      : {fpr:.6f}"
)

print("\nConfusion matrix:")

print(cm)

print("\nTrue Positives:")
print(tp)

print("\nFalse Positives:")
print(fp)

print("\nFalse Negatives:")
print(fn)

print("\nTrue Negatives:")
print(tn)


# ============================================================
# 13. TEMPORAL / DATA QUALITY EVALUATION
# ============================================================

target_types = [
    "temp_frozen",
    "temp_drift",
    "missing_temperature"
]

temporal_target = (
    test_df[
        "anomaly_type"
    ]
    .isin(
        target_types
    )
    .astype(int)
)

temporal_prediction = (
    y_pred
)


temporal_precision = precision_score(
    temporal_target,
    temporal_prediction,
    zero_division=0
)

temporal_recall = recall_score(
    temporal_target,
    temporal_prediction,
    zero_division=0
)

temporal_f1 = f1_score(
    temporal_target,
    temporal_prediction,
    zero_division=0
)


print("\n")
print("=" * 70)
print("TEMPORAL / DATA-QUALITY RULE PERFORMANCE")
print("=" * 70)

print(
    f"\nPrecision: {temporal_precision:.6f}"
)

print(
    f"Recall   : {temporal_recall:.6f}"
)

print(
    f"F1 Score : {temporal_f1:.6f}"
)


# ============================================================
# 14. DETECTION BY ANOMALY TYPE
# ============================================================

print("\n")
print("=" * 70)
print("ANOMALY TYPE DETECTION")
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
            "rule_anomaly"
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
# 15. RULE-SPECIFIC DETECTION
# ============================================================

print("\n")
print("=" * 70)
print("RULE-SPECIFIC DETECTION")
print("=" * 70)


# ------------------------------------------------------------
# Missing temperature
# ------------------------------------------------------------

actual = test_df[
    test_df[
        "anomaly_type"
    ]
    == "missing_temperature"
]

detected = actual[
    actual[
        "rule_missing_temperature"
    ]
    == 1
]

print(
    "\nMissing temperature rule"
)

print(
    "Actual:",
    len(actual)
)

print(
    "Detected:",
    len(detected)
)

if len(actual) > 0:

    print(
        "Detection rate:",
        f"{len(detected) / len(actual):.2%}"
    )


# ------------------------------------------------------------
# Frozen temperature
# ------------------------------------------------------------

actual = test_df[
    test_df[
        "anomaly_type"
    ]
    == "temp_frozen"
]

detected = actual[
    actual[
        "rule_temp_frozen"
    ]
    == 1
]

print(
    "\nFrozen temperature rule"
)

print(
    "Actual:",
    len(actual)
)

print(
    "Detected:",
    len(detected)
)

if len(actual) > 0:

    print(
        "Detection rate:",
        f"{len(detected) / len(actual):.2%}"
    )


# ------------------------------------------------------------
# Temperature drift
# ------------------------------------------------------------

actual = test_df[
    test_df[
        "anomaly_type"
    ]
    == "temp_drift"
]

detected = actual[
    actual[
        "rule_temp_drift"
    ]
    == 1
]

print(
    "\nTemperature drift rule"
)

print(
    "Actual:",
    len(actual)
)

print(
    "Detected:",
    len(detected)
)

if len(actual) > 0:

    print(
        "Detection rate:",
        f"{len(detected) / len(actual):.2%}"
    )


# ============================================================
# 16. SHOW ONLY DETECTED RECORDS
# ============================================================

print("\n")
print("=" * 70)
print("RULE-DETECTED 2024 ANOMALIES")
print("=" * 70)


detected = test_df[
    test_df[
        "rule_anomaly"
    ] == 1
][
    [
        "timestamp",
        "temperature",
        "humidity",
        "pressure",
        "anomaly",
        "anomaly_type",
        "rule_anomaly_type"
    ]
]


if len(detected) == 0:

    print(
        "No anomalies detected."
    )

else:

    print(
        detected.to_string(
            index=False
        )
    )


# ============================================================
# 17. SAVE 2024 RESULTS
# ============================================================

test_output = test_df[
    [
        "timestamp",
        "temperature",
        "humidity",
        "pressure",
        "anomaly",
        "anomaly_type",
        "rule_missing_temperature",
        "rule_temp_frozen",
        "rule_temp_drift",
        "rule_anomaly",
        "rule_anomaly_type"
    ]
].copy()


test_output.to_csv(
    RESULT_FILE,
    index=False
)


# ============================================================
# 18. SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    [
        {
            "detector":
                "Temporal Sensor Rules",

            "test_year":
                2024,

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
                tn,

            "temporal_precision":
                temporal_precision,

            "temporal_recall":
                temporal_recall,

            "temporal_f1":
                temporal_f1
        }
    ]
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# 19. SUCCESS
# ============================================================

print("\n")
print("=" * 70)
print("SENSOR RULE EVALUATION COMPLETE")
print("=" * 70)

print("\nResults saved:")
print(
    RESULT_FILE
)

print("\nSummary saved:")
print(
    SUMMARY_FILE
)