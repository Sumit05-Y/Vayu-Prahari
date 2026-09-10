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
    / "day6_final_detector_2024_results.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "day6_final_detector_summary.csv"
)

TYPE_RESULT_FILE = (
    REPORT_DIR
    / "day6_final_detector_anomaly_types.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("VAYU PRAHARI - FINAL ANOMALY DETECTOR")
print("=" * 70)

print("\nDataset:")
print(DATA_FILE)

print("\nIsolation Forest model:")
print(MODEL_FILE)

print("\nFeature list:")
print(FEATURE_FILE)


if not DATA_FILE.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_FILE}"
    )


if not MODEL_FILE.exists():

    raise FileNotFoundError(
        f"\nIsolation Forest model not found:\n{MODEL_FILE}"
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

print("\nNumber of model features:")
print(len(feature_columns))

print("\nFeatures:")

for i, feature in enumerate(
    feature_columns,
    start=1
):

    print(
        f"{i:2d}. {feature}"
    )


# ============================================================
# 5. SELECT 2024 TEST DATA
# ============================================================

test_df = df[
    df["year"] == 2024
].copy()

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
        "anomaly_type"
    ].value_counts()
)


# ============================================================
# 6. INITIALIZE ISOLATION FOREST RESULTS
# ============================================================

test_df["if_anomaly"] = 0

test_df["if_score"] = np.nan


# ============================================================
# 7. PREPARE VALID ROWS FOR ISOLATION FOREST
# ============================================================

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


# ============================================================
# 8. RUN ISOLATION FOREST
# ============================================================

print("\n")
print("=" * 70)
print("RUNNING ISOLATION FOREST")
print("=" * 70)

if len(X_test_if) > 0:

    raw_if_prediction = model.predict(
        X_test_if
    )

    if_prediction = (
        raw_if_prediction == -1
    ).astype(int)

    if_scores = model.decision_function(
        X_test_if
    )

    test_df.loc[
        valid_if_rows,
        "if_anomaly"
    ] = if_prediction

    test_df.loc[
        valid_if_rows,
        "if_score"
    ] = if_scores


# ============================================================
# 9. TEMPORAL RULE - MISSING TEMPERATURE
# ============================================================

missing_rule = (
    test_df[
        "temperature"
    ]
    .isna()
)


# ============================================================
# 10. TEMPORAL RULE - FROZEN TEMPERATURE
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
# 11. TEMPORAL RULE - TEMPERATURE DRIFT
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
# 12. STORE INDIVIDUAL RULE RESULTS
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


# ============================================================
# 13. COMBINE TEMPORAL RULES
# ============================================================

test_df["rule_anomaly"] = (
    missing_rule
    | frozen_rule
    | drift_rule
).astype(int)


# ============================================================
# 14. DETERMINE RULE ANOMALY TYPE
# ============================================================

test_df["rule_anomaly_type"] = np.select(
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
# 15. FINAL COMBINATION
# ============================================================

# Isolation Forest handles point anomalies.
# Temporal rules handle frozen, drift and missing readings.

test_df["final_anomaly"] = (
    (
        test_df["if_anomaly"]
        == 1
    )
    |
    (
        test_df["rule_anomaly"]
        == 1
    )
).astype(int)


# ============================================================
# 16. DETERMINE FINAL ANOMALY TYPE
# ============================================================

test_df["final_anomaly_type"] = np.select(
    [
        test_df["rule_missing_temperature"] == 1,
        test_df["rule_temp_frozen"] == 1,
        test_df["rule_temp_drift"] == 1,
        test_df["if_anomaly"] == 1
    ],
    [
        "missing_temperature",
        "temp_frozen",
        "temp_drift",
        "isolation_forest_anomaly"
    ],
    default="normal"
)


# ============================================================
# 17. OVERALL EVALUATION
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
# 18. PRINT ISOLATION FOREST RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("ISOLATION FOREST V2 RESULTS")
print("=" * 70)

print(
    "\nPredicted anomalies:",
    int(test_df["if_anomaly"].sum())
)

print(
    "Actual point anomalies detected:",
    int(
        test_df[
            (
                test_df["anomaly"] == 1
            )
            &
            (
                test_df["if_anomaly"] == 1
            )
        ].shape[0]
    )
)


# ============================================================
# 19. PRINT RULE RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("TEMPORAL RULE RESULTS")
print("=" * 70)

print(
    "\nRule anomalies:",
    int(test_df["rule_anomaly"].sum())
)


# ============================================================
# 20. PRINT FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL COMBINED DETECTOR")
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
# 21. ANOMALY TYPE DETECTION
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
# 22. DETECTION BY COMPONENT
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
    "Temporal rule alerts:",
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
# 23. SHOW MISSED ANOMALIES
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
# 24. SAVE FINAL RESULTS
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
# 25. SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    [
        {
            "detector":
                "Isolation Forest V2 + Temporal Rules",

            "test_year":
                2024,

            "test_rows":
                len(test_df),

            "actual_anomalies":
                int(y_true.sum()),

            "final_predicted_anomalies":
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
# 26. SAVE ANOMALY TYPE RESULTS
# ============================================================

type_results_df.to_csv(
    TYPE_RESULT_FILE,
    index=False
)


# ============================================================
# 27. SUCCESS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL DETECTOR COMPLETE")
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