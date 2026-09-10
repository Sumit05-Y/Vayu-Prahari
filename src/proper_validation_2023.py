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
    / "isolation_forest_validation_2023.joblib"
)

FEATURE_FILE = (
    MODEL_DIR
    / "feature_columns_validation_2023.joblib"
)

RESULT_FILE = (
    REPORT_DIR
    / "proper_validation_2023_results.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "proper_validation_2023_summary.csv"
)

TYPE_RESULT_FILE = (
    REPORT_DIR
    / "proper_validation_2023_anomaly_types.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("VAYU PRAHARI - PROPER 2023 VALIDATION")
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
# 4. EXACT 24 MODEL FEATURES
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
# 5. CHECK FEATURES
# ============================================================

missing_columns = [
    column
    for column in (
        feature_columns
        + [
            "anomaly",
            "anomaly_type"
        ]
    )
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing columns:\n"
        + "\n".join(
            missing_columns
        )
    )


# ============================================================
# 6. CONVERT MODEL FEATURES TO NUMERIC
# ============================================================

for column in feature_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# 7. PREPARE TRUE TRAINING DATA
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING DATA")
print("=" * 70)

train_df = df[
    (
        df["year"] >= 2020
    )
    &
    (
        df["year"] <= 2022
    )
    &
    (
        df["anomaly"] == 0
    )
].copy()


print(
    "\nTraining rows before dropna:",
    len(train_df)
)


train_df = train_df.dropna(
    subset=feature_columns
).copy()


print(
    "Training rows after dropna:",
    len(train_df)
)


X_train = train_df[
    feature_columns
]


print(
    "\nX_train shape:",
    X_train.shape
)


# ============================================================
# 8. TRAIN VALIDATION MODEL
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING VALIDATION MODEL")
print("=" * 70)


model = IsolationForest(
    n_estimators=300,
    max_samples=0.8,
    max_features=1.0,
    contamination=0.01,
    random_state=42,
    n_jobs=-1
)


print("\nConfiguration:")
print("n_estimators  = 300")
print("max_samples   = 0.8")
print("max_features  = 1.0")
print("contamination = 0.01")
print("random_state  = 42")


model.fit(
    X_train
)


print("\nTraining completed.")


# ============================================================
# 9. SAVE VALIDATION MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)

joblib.dump(
    feature_columns,
    FEATURE_FILE
)


print("\nValidation model saved:")
print(MODEL_FILE)

print("\nFeature list saved:")
print(FEATURE_FILE)


# ============================================================
# 10. 2023 VALIDATION DATA
# ============================================================

print("\n")
print("=" * 70)
print("2023 VALIDATION DATA")
print("=" * 70)


test_df = df[
    df["year"] == 2023
].copy()


test_df = test_df.reset_index(
    drop=True
)


print(
    "\nRows:",
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
        test_df["anomaly"] == 1
    ]["anomaly_type"].value_counts()
)


# ============================================================
# 11. RUN ISOLATION FOREST
# ============================================================

test_df["if_anomaly"] = 0

test_df["if_score"] = np.nan


valid_rows = (
    test_df[
        feature_columns
    ]
    .notna()
    .all(
        axis=1
    )
)


X_test = test_df.loc[
    valid_rows,
    feature_columns
]


print(
    "\nRows usable by Isolation Forest:",
    len(X_test)
)


raw_prediction = model.predict(
    X_test
)

if_prediction = (
    raw_prediction == -1
).astype(int)

if_score = model.decision_function(
    X_test
)


test_df.loc[
    valid_rows,
    "if_anomaly"
] = if_prediction

test_df.loc[
    valid_rows,
    "if_score"
] = if_score


print(
    "\nIsolation Forest alerts:",
    int(
        test_df[
            "if_anomaly"
        ].sum()
    )
)


# ============================================================
# 12. MISSING TEMPERATURE RULE
# ============================================================

missing_rule = (
    test_df[
        "temperature"
    ]
    .isna()
)


# ============================================================
# 13. FROZEN TEMPERATURE RULE
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
        test_df
    )
)


# ============================================================
# 14. TEMPERATURE DRIFT RULE
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
# 15. HUMIDITY RULE
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
    &
    current_humidity.notna()
    &
    previous_humidity.notna()
    &
    next_humidity.notna()
    &
    (
        current_humidity
        >= 99
    )
    &
    (
        previous_humidity
        < 99
    )
    &
    (
        next_humidity
        < 99
    )
    &
    (
        current_humidity
        -
        np.maximum(
            previous_humidity,
            next_humidity
        )
        >= HUMIDITY_JUMP_THRESHOLD
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
    (
        current_humidity
        <= 6
    )
    &
    (
        previous_humidity
        > 6
    )
    &
    (
        next_humidity
        > 6
    )
    &
    (
        np.minimum(
            previous_humidity,
            next_humidity
        )
        -
        current_humidity
        >= HUMIDITY_JUMP_THRESHOLD
    )
)


humidity_rule = (
    humidity_spike_rule
    |
    humidity_drop_rule
)


# ============================================================
# 16. COMBINE SENSOR RULES
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


# ============================================================
# 17. FINAL COMBINATION
# ============================================================

test_df["final_anomaly"] = (
    (
        test_df[
            "if_anomaly"
        ]
        == 1
    )
    |
    (
        test_df[
            "rule_anomaly"
        ]
        == 1
    )
).astype(int)


# ============================================================
# 18. FINAL ANOMALY TYPE
# ============================================================

test_df["final_anomaly_type"] = np.select(
    [
        missing_rule,
        frozen_rule,
        drift_rule,
        humidity_spike_rule,
        humidity_drop_rule,
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
# 19. EVALUATION
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
        fp + tn
    )
    if (
        fp + tn
    ) > 0
    else 0
)


# ============================================================
# 20. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("PROPER 2023 VALIDATION RESULTS")
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
    int(
        y_true.sum()
    )
)

print(
    "Predicted anomalies:",
    int(
        y_pred.sum()
    )
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

print("\nConfusion matrix:")
print(cm)


# ============================================================
# 21. ANOMALY TYPE RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("PROPER 2023 ANOMALY TYPE DETECTION")
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
# 22. MISSED ANOMALIES
# ============================================================

print("\n")
print("=" * 70)
print("PROPER 2023 MISSED ANOMALIES")
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
# 23. SAVE RESULTS
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

    "rule_anomaly",

    "final_anomaly",
    "final_anomaly_type"
]


test_df[
    result_columns
].to_csv(
    RESULT_FILE,
    index=False
)


summary_df = pd.DataFrame(
    [
        {
            "detector":
                "Isolation Forest Validation Model + Sensor Rules",

            "training_years":
                "2020-2022",

            "validation_year":
                2023,

            "n_estimators":
                300,

            "max_samples":
                0.8,

            "max_features":
                1.0,

            "contamination":
                0.01,

            "humidity_threshold":
                HUMIDITY_JUMP_THRESHOLD,

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


type_results_df.to_csv(
    TYPE_RESULT_FILE,
    index=False
)


# ============================================================
# 24. SUCCESS
# ============================================================

print("\n")
print("=" * 70)
print("PROPER 2023 VALIDATION COMPLETE")
print("=" * 70)

print("\nModel saved:")
print(
    MODEL_FILE
)

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