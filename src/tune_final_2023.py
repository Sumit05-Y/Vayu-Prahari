import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.ensemble import IsolationForest
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

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "day6"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    REPORT_DIR
    / "2023_parameter_tuning_results.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("VAYU PRAHARI - FINAL PARAMETER TUNING USING 2023")
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
# 4. EXACT 24 FEATURES
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
# 5. TRAINING DATA = 2020-2022 NORMAL ONLY
# ============================================================

print("\n")
print("=" * 70)
print("PREPARING TRAINING DATA")
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


# ============================================================
# 6. 2023 VALIDATION DATA
# ============================================================

validation_df = df[
    df["year"] == 2023
].copy()

validation_df = validation_df.reset_index(
    drop=True
)


print("\n")
print("=" * 70)
print("2023 VALIDATION DATA")
print("=" * 70)

print(
    "\nRows:",
    len(validation_df)
)

print("\nActual anomalies:")

print(
    validation_df[
        "anomaly"
    ].value_counts()
)


# ============================================================
# 7. COMMON TEMPORAL RULES
# ============================================================

# ------------------------------------------------------------
# Missing temperature
# ------------------------------------------------------------

missing_rule = (
    validation_df[
        "temperature"
    ].isna()
)


# ============================================================
# 8. FROZEN TEMPERATURE RULE
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
        validation_df
    )
)


# ============================================================
# 9. TEMPERATURE DRIFT RULE
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
        validation_df
    )
)


# ============================================================
# 10. PREPARE HUMIDITY INFORMATION
# ============================================================

validation_df["humidity_prev"] = (
    validation_df[
        "humidity"
    ].shift(1)
)

validation_df["humidity_next"] = (
    validation_df[
        "humidity"
    ].shift(-1)
)

validation_df["timestamp_prev"] = (
    validation_df[
        "timestamp"
    ].shift(1)
)

validation_df["timestamp_next"] = (
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
            "timestamp_prev"
        ]
    )
    .dt.total_seconds()
    / 3600
)

next_gap = (
    (
        validation_df[
            "timestamp_next"
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
        "humidity_prev"
    ]
)

next_humidity = (
    validation_df[
        "humidity_next"
    ]
)


# ============================================================
# 11. PARAMETERS TO TEST
# ============================================================

max_samples_values = [
    0.6,
    0.8
]

contamination_values = [
    0.005,
    0.01,
    0.02
]

humidity_threshold_values = [
    5,
    10,
    15,
    20,
    25
]


# ============================================================
# 12. TUNING LOOP
# ============================================================

all_results = []


for max_samples in max_samples_values:

    for contamination in contamination_values:

        print("\n")
        print(
            "=" * 70
        )

        print(
            "Training configuration:"
        )

        print(
            "max_samples =",
            max_samples
        )

        print(
            "contamination =",
            contamination
        )


        # ----------------------------------------------------
        # Train Isolation Forest
        # ----------------------------------------------------

        model = IsolationForest(
            n_estimators=300,
            max_samples=max_samples,
            max_features=1.0,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X_train
        )


        # ----------------------------------------------------
        # Isolation Forest prediction
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Test humidity thresholds
        # ----------------------------------------------------

        for humidity_threshold in (
            humidity_threshold_values
        ):

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
                    >= humidity_threshold
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
                    >= humidity_threshold
                )
            )


            humidity_rule = (
                humidity_spike_rule
                |
                humidity_drop_rule
            )


            # ------------------------------------------------
            # Combine temporal rules
            # ------------------------------------------------

            rule_prediction = (
                missing_rule
                |
                frozen_rule
                |
                drift_rule
                |
                humidity_rule
            ).astype(int)


            # ------------------------------------------------
            # FINAL PREDICTION
            # ------------------------------------------------

            final_prediction = (
                (
                    if_full == 1
                )
                |
                (
                    rule_prediction == 1
                )
            ).astype(int)


            # ------------------------------------------------
            # TRUE LABELS
            # ------------------------------------------------

            actual = (
                validation_df[
                    "anomaly"
                ]
                .astype(int)
                .to_numpy()
            )


            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

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


            predicted_anomalies = int(
                final_prediction.sum()
            )


            # ------------------------------------------------
            # SAVE RESULT
            # ------------------------------------------------

            all_results.append(
                {
                    "n_estimators": 300,

                    "max_samples":
                        max_samples,

                    "max_features":
                        1.0,

                    "contamination":
                        contamination,

                    "humidity_threshold":
                        humidity_threshold,

                    "predicted_anomalies":
                        predicted_anomalies,

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
                        int(tn)
                }
            )


# ============================================================
# 13. CREATE RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    all_results
)


# ============================================================
# 14. SORT BEST CONFIGURATIONS
# ============================================================

results_df = results_df.sort_values(
    [
        "f1",
        "recall",
        "precision",
        "fpr"
    ],
    ascending=[
        False,
        False,
        False,
        True
    ]
).reset_index(
    drop=True
)


# ============================================================
# 15. PRINT ALL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("ALL 2023 PARAMETER RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 16. TOP 10 CONFIGURATIONS
# ============================================================

print("\n")
print("=" * 70)
print("TOP 10 CONFIGURATIONS")
print("=" * 70)

print(
    results_df.head(
        10
    ).to_string(
        index=False
    )
)


# ============================================================
# 17. BEST CONFIGURATION
# ============================================================

best = results_df.iloc[0]


print("\n")
print("=" * 70)
print("BEST 2023 CONFIGURATION")
print("=" * 70)

print(
    "\nn_estimators:",
    best["n_estimators"]
)

print(
    "max_samples:",
    best["max_samples"]
)

print(
    "max_features:",
    best["max_features"]
)

print(
    "contamination:",
    best["contamination"]
)

print(
    "humidity_threshold:",
    best["humidity_threshold"]
)

print(
    "\nPrecision:",
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
    int(best["TP"])
)

print(
    "FP:",
    int(best["FP"])
)

print(
    "FN:",
    int(best["FN"])
)

print(
    "TN:",
    int(best["TN"])
)


# ============================================================
# 18. SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 19. SUCCESS
# ============================================================

print("\n")
print("=" * 70)
print("2023 PARAMETER TUNING COMPLETE")
print("=" * 70)

print("\nResults saved:")
print(
    OUTPUT_FILE
)