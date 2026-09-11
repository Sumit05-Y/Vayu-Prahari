import numpy as np
import pandas as pd
import joblib

from pathlib import Path

from sklearn.ensemble import IsolationForest
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
    / "day7"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_FILE = (
    REPORT_DIR
    / "multivariate_rule_tuning_2023.csv"
)

BEST_FILE = (
    REPORT_DIR
    / "multivariate_rule_best_config.txt"
)


# ============================================================
# 2. FINAL FROZEN ISOLATION FOREST PARAMETERS
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
print("VAYU PRAHARI - MULTIVARIATE RULE TUNING USING 2023")
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
# 5. EXACT 24 MODEL FEATURES
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
# 6. TRAIN VALIDATION ISOLATION FOREST
#
# Train:
# 2020-2022 NORMAL ONLY
#
# Validate:
# 2023
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING 2023 VALIDATION MODEL")
print("=" * 70)


train_df = df[
    (df["year"] >= 2020)
    &
    (df["year"] <= 2022)
    &
    (df["anomaly"] == 0)
].copy()


train_df = train_df.dropna(
    subset=feature_columns
).copy()


X_train = train_df[
    feature_columns
]


print(
    "\nTraining rows:",
    len(X_train)
)


print(
    "Training shape:",
    X_train.shape
)


validation_df = df[
    df["year"] == 2023
].copy()


validation_df = validation_df.reset_index(
    drop=True
)


print("\n2023 rows:")
print(
    len(validation_df)
)


print("\n2023 actual anomalies:")
print(
    validation_df[
        "anomaly"
    ].value_counts()
)


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


# ============================================================
# 7. ISOLATION FOREST PREDICTION
# ============================================================

valid_rows = (
    validation_df[
        feature_columns
    ]
    .notna()
    .all(
        axis=1
    )
)


X_validation = (
    validation_df.loc[
        valid_rows,
        feature_columns
    ]
)


raw_prediction = model.predict(
    X_validation
)


if_prediction = (
    raw_prediction == -1
).astype(int)


if_full = np.zeros(
    len(validation_df),
    dtype=int
)


if_full[
    valid_rows.to_numpy()
] = if_prediction


print("\nIsolation Forest alerts:")
print(
    int(if_full.sum())
)


# ============================================================
# 8. MISSING TEMPERATURE RULE
# ============================================================

missing_rule = (
    validation_df[
        "temperature"
    ].isna()
)


# ============================================================
# 9. FROZEN TEMPERATURE RULE
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
        validation_df
    )
)


# ============================================================
# 10. TEMPERATURE DRIFT RULE
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
            -
            step_sizes.min()
        )

        if (
            step_variation
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
            + intercept
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
        validation_df
    )
)


# ============================================================
# 11. HUMIDITY RULE
# ============================================================

validation_df["humidity_prev_raw"] = (
    validation_df[
        "humidity"
    ].shift(1)
)

validation_df["humidity_next_raw"] = (
    validation_df[
        "humidity"
    ].shift(-1)
)

validation_df["timestamp_prev_raw"] = (
    validation_df[
        "timestamp"
    ].shift(1)
)

validation_df["timestamp_next_raw"] = (
    validation_df[
        "timestamp"
    ].shift(-1)
)


prev_gap = (
    (
        validation_df[
            "timestamp"
        ]
        -
        validation_df[
            "timestamp_prev_raw"
        ]
    )
    .dt.total_seconds()
    / 3600
)


next_gap = (
    (
        validation_df[
            "timestamp_next_raw"
        ]
        -
        validation_df[
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
    validation_df[
        "humidity"
    ]
)

previous_humidity = (
    validation_df[
        "humidity_prev_raw"
    ]
)

next_humidity = (
    validation_df[
        "humidity_next_raw"
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
# 12. BASELINE DETECTOR
#
# This is our CURRENT detector WITHOUT
# the new multivariate rule.
# ============================================================

baseline_rule_prediction = (
    missing_rule
    |
    frozen_rule
    |
    drift_rule
    |
    humidity_rule
).astype(int)


baseline_prediction = (
    (
        if_full == 1
    )
    |
    (
        baseline_rule_prediction == 1
    )
).astype(int)


actual = (
    validation_df[
        "anomaly"
    ]
    .astype(int)
    .to_numpy()
)


# ============================================================
# 13. BASELINE METRICS
# ============================================================

baseline_accuracy = accuracy_score(
    actual,
    baseline_prediction
)

baseline_precision = precision_score(
    actual,
    baseline_prediction,
    zero_division=0
)

baseline_recall = recall_score(
    actual,
    baseline_prediction,
    zero_division=0
)

baseline_f1 = f1_score(
    actual,
    baseline_prediction,
    zero_division=0
)

baseline_tn, baseline_fp, baseline_fn, baseline_tp = (
    confusion_matrix(
        actual,
        baseline_prediction
    ).ravel()
)

baseline_fpr = (
    baseline_fp
    /
    (
        baseline_fp
        +
        baseline_tn
    )
)


# ============================================================
# 14. PREPARE MULTIVARIATE RESIDUALS
#
# We compare the CURRENT value with the
# average of the previous and next values.
#
# Example:
#
# Previous = 25
# Current  = 35
# Next     = 25
#
# Expected current ≈ 25
# Residual = 10
#
# ============================================================

validation_df["temperature_prev_raw"] = (
    validation_df[
        "temperature"
    ].shift(1)
)

validation_df["temperature_next_raw"] = (
    validation_df[
        "temperature"
    ].shift(-1)
)

validation_df["pressure_prev_raw"] = (
    validation_df[
        "pressure"
    ].shift(1)
)

validation_df["pressure_next_raw"] = (
    validation_df[
        "pressure"
    ].shift(-1)
)


temperature_expected = (
    (
        validation_df[
            "temperature_prev_raw"
        ]
        +
        validation_df[
            "temperature_next_raw"
        ]
    )
    / 2
)


humidity_expected = (
    (
        validation_df[
            "humidity_prev_raw"
        ]
        +
        validation_df[
            "humidity_next_raw"
        ]
    )
    / 2
)


pressure_expected = (
    (
        validation_df[
            "pressure_prev_raw"
        ]
        +
        validation_df[
            "pressure_next_raw"
        ]
    )
    / 2
)


temperature_residual = (
    (
        validation_df[
            "temperature"
        ]
        -
        temperature_expected
    )
    .abs()
)


humidity_residual = (
    (
        validation_df[
            "humidity"
        ]
        -
        humidity_expected
    )
    .abs()
)


pressure_residual = (
    (
        validation_df[
            "pressure"
        ]
        -
        pressure_expected
    )
    .abs()
)


# ============================================================
# 15. VALID MULTIVARIATE WINDOWS
# ============================================================

temperature_valid = (
    validation_df[
        [
            "temperature_prev_raw",
            "temperature",
            "temperature_next_raw"
        ]
    ]
    .notna()
    .all(
        axis=1
    )
)


humidity_valid = (
    validation_df[
        [
            "humidity_prev_raw",
            "humidity",
            "humidity_next_raw"
        ]
    ]
    .notna()
    .all(
        axis=1
    )
)


pressure_valid = (
    validation_df[
        [
            "pressure_prev_raw",
            "pressure",
            "pressure_next_raw"
        ]
    ]
    .notna()
    .all(
        axis=1
    )
)


multivariate_valid = (
    valid_neighbors
    &
    temperature_valid
    &
    humidity_valid
    &
    pressure_valid
)


# ============================================================
# 16. PARAMETERS TO TUNE
# ============================================================

temperature_thresholds = [
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10
]


humidity_max_residuals = [
    2,
    5,
    8,
    10,
    15
]


pressure_max_residuals = [
    0.5,
    1,
    2,
    3,
    5
]


# ============================================================
# 17. BASELINE SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("CURRENT BASELINE - 2023")
print("=" * 70)

print(
    "\nAccuracy:",
    f"{baseline_accuracy:.6f}"
)

print(
    "Precision:",
    f"{baseline_precision:.6f}"
)

print(
    "Recall:",
    f"{baseline_recall:.6f}"
)

print(
    "F1:",
    f"{baseline_f1:.6f}"
)

print(
    "FPR:",
    f"{baseline_fpr:.6f}"
)

print(
    "\nTP:",
    baseline_tp
)

print(
    "FP:",
    baseline_fp
)

print(
    "FN:",
    baseline_fn
)

print(
    "TN:",
    baseline_tn
)


# ============================================================
# 18. TUNE MULTIVARIATE RULE
# ============================================================

results = []


for temperature_threshold in (
    temperature_thresholds
):

    for humidity_max in (
        humidity_max_residuals
    ):

        for pressure_max in (
            pressure_max_residuals
        ):

            multivariate_rule = (
                multivariate_valid

                & (
                    temperature_residual
                    >= temperature_threshold
                )

                & (
                    humidity_residual
                    <= humidity_max
                )

                & (
                    pressure_residual
                    <= pressure_max
                )
            )


            multivariate_rule = (
                multivariate_rule
                .fillna(False)
                .to_numpy()
            )


            combined_prediction = (
                (
                    baseline_prediction == 1
                )
                |
                (
                    multivariate_rule
                )
            ).astype(int)


            accuracy = accuracy_score(
                actual,
                combined_prediction
            )

            precision = precision_score(
                actual,
                combined_prediction,
                zero_division=0
            )

            recall = recall_score(
                actual,
                combined_prediction,
                zero_division=0
            )

            f1 = f1_score(
                actual,
                combined_prediction,
                zero_division=0
            )


            tn, fp, fn, tp = (
                confusion_matrix(
                    actual,
                    combined_prediction
                ).ravel()
            )


            fpr = (
                fp
                /
                (
                    fp
                    +
                    tn
                )
                if (
                    fp
                    +
                    tn
                ) > 0
                else 0
            )


            multivariate_actual = (
                validation_df[
                    "anomaly_type"
                ]
                ==
                "multivariate_inconsistency"
            ).to_numpy()
            

            multivariate_detected = int(
                (
                    multivariate_rule
                    &
                    multivariate_actual
                ).sum()
            )


            multivariate_total = int(
                multivariate_actual.sum()
            )


            results.append(
                {
                    "temperature_threshold":
                        temperature_threshold,

                    "humidity_max_residual":
                        humidity_max,

                    "pressure_max_residual":
                        pressure_max,

                    "multivariate_detected":
                        multivariate_detected,

                    "multivariate_total":
                        multivariate_total,

                    "multivariate_recall":
                        (
                            multivariate_detected
                            /
                            multivariate_total
                        )
                        if multivariate_total > 0
                        else 0,

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
                        int(tp),

                    "FP":
                        int(fp),

                    "FN":
                        int(fn),

                    "TN":
                        int(tn),

                    "added_alerts":
                        int(
                            (
                                multivariate_rule
                                &
                                (
                                    baseline_prediction == 0
                                )
                            ).sum()
                        ),

                    "baseline_f1":
                        baseline_f1,

                    "baseline_recall":
                        baseline_recall,

                    "baseline_fpr":
                        baseline_fpr
                }
            )


# ============================================================
# 19. RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# 20. SORT RESULTS
#
# Priority:
# 1. F1
# 2. Recall
# 3. FPR
# 4. Precision
# ============================================================

results_df = results_df.sort_values(
    [
        "f1",
        "recall",
        "fpr",
        "precision"
    ],
    ascending=[
        False,
        False,
        True,
        False
    ]
).reset_index(
    drop=True
)


# ============================================================
# 21. PRINT TOP RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("TOP 20 MULTIVARIATE CONFIGURATIONS")
print("=" * 70)


print(
    results_df.head(
        20
    ).to_string(
        index=False
    )
)


# ============================================================
# 22. BEST CONFIGURATION
# ============================================================

best = results_df.iloc[0]


print("\n")
print("=" * 70)
print("BEST MULTIVARIATE CONFIGURATION")
print("=" * 70)


print(
    "\nTemperature residual threshold:",
    best[
        "temperature_threshold"
    ]
)

print(
    "Maximum humidity residual:",
    best[
        "humidity_max_residual"
    ]
)

print(
    "Maximum pressure residual:",
    best[
        "pressure_max_residual"
    ]
)


print(
    "\nMultivariate detection:",
    int(
        best[
            "multivariate_detected"
        ]
    ),
    "/",
    int(
        best[
            "multivariate_total"
        ]
    )
)


print(
    "\nAccuracy:",
    f"{best['accuracy']:.6f}"
)

print(
    "Precision:",
    f"{best['precision']:.6f}"
)

print(
    "Recall:",
    f"{best['recall']:.6f}"
)

print(
    "F1:",
    f"{best['f1']:.6f}"
)

print(
    "FPR:",
    f"{best['fpr']:.6f}"
)


print(
    "\nTP:",
    int(
        best["TP"]
    )
)

print(
    "FP:",
    int(
        best["FP"]
    )
)

print(
    "FN:",
    int(
        best["FN"]
    )
)

print(
    "TN:",
    int(
        best["TN"]
    )
)


print(
    "\nAdditional alerts caused by the rule:",
    int(
        best["added_alerts"]
    )
)


# ============================================================
# 23. COMPARE AGAINST BASELINE
# ============================================================

print("\n")
print("=" * 70)
print("BASELINE VS MULTIVARIATE RULE")
print("=" * 70)


print(
    "\nMetric              Baseline       With Rule"
)

print(
    f"Accuracy            "
    f"{baseline_accuracy:.6f}       "
    f"{best['accuracy']:.6f}"
)

print(
    f"Precision           "
    f"{baseline_precision:.6f}       "
    f"{best['precision']:.6f}"
)

print(
    f"Recall              "
    f"{baseline_recall:.6f}       "
    f"{best['recall']:.6f}"
)

print(
    f"F1                  "
    f"{baseline_f1:.6f}       "
    f"{best['f1']:.6f}"
)

print(
    f"FPR                 "
    f"{baseline_fpr:.6f}       "
    f"{best['fpr']:.6f}"
)


# ============================================================
# 24. SAVE ALL RESULTS
# ============================================================

results_df.to_csv(
    RESULT_FILE,
    index=False
)


# ============================================================
# 25. SAVE BEST CONFIGURATION
# ============================================================

best_text = f"""
VAYU PRAHARI - BEST MULTIVARIATE RULE
=====================================

Validation year:
2023

Training years:
2020-2022

Isolation Forest:
n_estimators = {N_ESTIMATORS}
max_samples = {MAX_SAMPLES}
max_features = {MAX_FEATURES}
contamination = {CONTAMINATION}

Existing humidity threshold:
{HUMIDITY_THRESHOLD}

Best multivariate parameters:

temperature residual threshold:
{best["temperature_threshold"]}

maximum humidity residual:
{best["humidity_max_residual"]}

maximum pressure residual:
{best["pressure_max_residual"]}

2023 performance with new rule:

Accuracy:
{best["accuracy"]:.6f}

Precision:
{best["precision"]:.6f}

Recall:
{best["recall"]:.6f}

F1:
{best["f1"]:.6f}

False Positive Rate:
{best["fpr"]:.6f}

TP:
{int(best["TP"])}

FP:
{int(best["FP"])}

FN:
{int(best["FN"])}

TN:
{int(best["TN"])}

Multivariate anomalies detected:
{int(best["multivariate_detected"])} / {int(best["multivariate_total"])}

Additional alerts caused by rule:
{int(best["added_alerts"])}
"""


BEST_FILE.write_text(
    best_text.strip(),
    encoding="utf-8"
)


# ============================================================
# 26. COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("MULTIVARIATE TUNING COMPLETE")
print("=" * 70)

print(
    "\nAll tuning results saved:"
)

print(
    RESULT_FILE
)

print(
    "\nBest configuration saved:"
)

print(
    BEST_FILE
)

print("\n")
print("=" * 70)
print("DO NOT MODIFY THE FINAL 2024 MODEL YET")
print("=" * 70)