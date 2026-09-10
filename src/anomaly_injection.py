import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 60)
print("VAYU PRAHARI - ANOMALY INJECTION")
print("=" * 60)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nInput directory:")
print(INPUT_DIR)

print("\nOutput directory:")
print(OUTPUT_DIR)


# ============================================================
# 2. LOAD ALL FIVE YEARS
# ============================================================

files = [
    INPUT_DIR / "2020CLEANED.csv",
    INPUT_DIR / "2021CLEANED.csv",
    INPUT_DIR / "2022CLEANED.csv",
    INPUT_DIR / "2023CLEANED.csv",
    INPUT_DIR / "2024CLEANED.csv"
]

dataframes = []

print("\n")
print("=" * 60)
print("LOADING DATASETS")
print("=" * 60)

for file in files:

    if not file.exists():

        raise FileNotFoundError(
            f"\nFile not found:\n{file}"
        )

    temp_df = pd.read_csv(file)

    print(
        f"{file.name}: "
        f"{len(temp_df)} rows"
    )

    dataframes.append(temp_df)


# ============================================================
# 3. COMBINE ALL DATASETS
# ============================================================

df = pd.concat(
    dataframes,
    ignore_index=True
)

print("\nAll datasets combined.")

print(
    "Combined shape:",
    df.shape
)


# ============================================================
# 4. STANDARDIZE COLUMN NAMES
# ============================================================

column_mapping = {}

if "temp" in df.columns:
    column_mapping["temp"] = "temperature"

if "rhum" in df.columns:
    column_mapping["rhum"] = "humidity"

if "pres" in df.columns:
    column_mapping["pres"] = "pressure"

df = df.rename(
    columns=column_mapping
)


# ============================================================
# 5. CREATE TIMESTAMP
# ============================================================

if "timestamp" not in df.columns:

    required_time_columns = [
        "year",
        "month",
        "day",
        "hour"
    ]

    for col in required_time_columns:

        if col not in df.columns:

            raise ValueError(
                f"Missing time column: {col}"
            )

    df["timestamp"] = pd.to_datetime(
        df[
            [
                "year",
                "month",
                "day",
                "hour"
            ]
        ],
        errors="coerce"
    )

else:

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )


# ============================================================
# 6. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "timestamp",
    "temperature",
    "humidity",
    "pressure"
]

for col in required_columns:

    if col not in df.columns:

        raise ValueError(
            f"Required column not found: {col}"
        )


# ============================================================
# 7. KEEP REQUIRED WEATHER DATA
# ============================================================

df = df[
    required_columns
].copy()


# ============================================================
# 8. CONVERT WEATHER VALUES TO NUMERIC
# ============================================================

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


# ============================================================
# 9. CHECK MISSING VALUES BEFORE ANOMALY INJECTION
# ============================================================

print("\n")
print("=" * 60)
print("MISSING VALUES BEFORE ANOMALY INJECTION")
print("=" * 60)

print(
    df[
        [
            "temperature",
            "humidity",
            "pressure"
        ]
    ].isna().sum()
)


# ============================================================
# 10. SORT BY TIMESTAMP
# ============================================================

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)

print("\n")
print("=" * 60)
print("CLEAN DATASET")
print("=" * 60)

print("\nShape:")
print(df.shape)

print("\nDate range:")
print(
    df["timestamp"].min(),
    "to",
    df["timestamp"].max()
)


# ============================================================
# 11. CREATE SYNTHETIC DATASET
# ============================================================

synthetic = df.copy()

synthetic["anomaly"] = 0

synthetic["anomaly_type"] = "normal"


# Fixed random seed
rng = np.random.default_rng(42)


# ============================================================
# 12. ANOMALY COUNTS
# ============================================================

POINT_ANOMALY_COUNT = 20

FROZEN_SEQUENCE_COUNT = 10
FROZEN_LENGTH = 5

DRIFT_SEQUENCE_COUNT = 10
DRIFT_LENGTH = 10

MISSING_SEQUENCE_COUNT = 10
MISSING_LENGTH = 5


# ============================================================
# 13. USED INDEX TRACKING
# ============================================================

used_indices = set()


def mark_point_anomaly(
    idx,
    column,
    value,
    anomaly_type
):

    synthetic.loc[
        idx,
        column
    ] = value

    synthetic.loc[
        idx,
        "anomaly"
    ] = 1

    synthetic.loc[
        idx,
        "anomaly_type"
    ] = anomaly_type

    used_indices.add(idx)


def get_available_window(length):

    max_start = len(synthetic) - length - 1

    if max_start <= 10:
        raise RuntimeError(
            "Dataset is too small for anomaly generation."
        )

    attempts = 0

    while attempts < 10000:

        start_idx = int(
            rng.integers(
                10,
                max_start
            )
        )

        window = set(
            range(
                start_idx,
                start_idx + length
            )
        )

        if not window.intersection(
            used_indices
        ):

            used_indices.update(window)

            return start_idx

        attempts += 1

    raise RuntimeError(
        "Could not find enough unused anomaly windows."
    )


# ============================================================
# 14. TEMPERATURE SPIKES
# ============================================================

for _ in range(POINT_ANOMALY_COUNT):

    idx = get_available_window(1)

    original = synthetic.loc[
        idx,
        "temperature"
    ]

    change = rng.uniform(
        15,
        25
    )

    mark_point_anomaly(
        idx,
        "temperature",
        original + change,
        "temp_spike"
    )


# ============================================================
# 15. TEMPERATURE DROPS
# ============================================================

for _ in range(POINT_ANOMALY_COUNT):

    idx = get_available_window(1)

    original = synthetic.loc[
        idx,
        "temperature"
    ]

    change = rng.uniform(
        15,
        25
    )

    mark_point_anomaly(
        idx,
        "temperature",
        original - change,
        "temp_drop"
    )


# ============================================================
# 16. HUMIDITY SPIKES
# ============================================================

for _ in range(POINT_ANOMALY_COUNT):

    idx = get_available_window(1)

    mark_point_anomaly(
        idx,
        "humidity",
        100,
        "humidity_spike"
    )


# ============================================================
# 17. HUMIDITY DROPS
# ============================================================

for _ in range(POINT_ANOMALY_COUNT):

    idx = get_available_window(1)

    mark_point_anomaly(
        idx,
        "humidity",
        5,
        "humidity_drop"
    )


# ============================================================
# 18. PRESSURE SPIKES
# ============================================================

for _ in range(POINT_ANOMALY_COUNT):

    idx = get_available_window(1)

    original = synthetic.loc[
        idx,
        "pressure"
    ]

    change = rng.uniform(
        15,
        25
    )

    mark_point_anomaly(
        idx,
        "pressure",
        original + change,
        "pressure_spike"
    )


# ============================================================
# 19. PRESSURE DROPS
# ============================================================

for _ in range(POINT_ANOMALY_COUNT):

    idx = get_available_window(1)

    original = synthetic.loc[
        idx,
        "pressure"
    ]

    change = rng.uniform(
        15,
        25
    )

    mark_point_anomaly(
        idx,
        "pressure",
        original - change,
        "pressure_drop"
    )


# ============================================================
# 20. FROZEN / STUCK TEMPERATURE SENSOR
# ============================================================

for _ in range(FROZEN_SEQUENCE_COUNT):

    start_idx = get_available_window(
        FROZEN_LENGTH
    )

    frozen_value = synthetic.loc[
        start_idx,
        "temperature"
    ]

    for i in range(FROZEN_LENGTH):

        idx = start_idx + i

        synthetic.loc[
            idx,
            "temperature"
        ] = frozen_value

        synthetic.loc[
            idx,
            "anomaly"
        ] = 1

        synthetic.loc[
            idx,
            "anomaly_type"
        ] = "temp_frozen"


# ============================================================
# 21. TEMPERATURE DRIFT
# ============================================================

for _ in range(DRIFT_SEQUENCE_COUNT):

    start_idx = get_available_window(
        DRIFT_LENGTH
    )

    drift_values = np.linspace(
        0,
        8,
        DRIFT_LENGTH
    )

    original = synthetic.loc[
        start_idx,
        "temperature"
    ]

    for i in range(DRIFT_LENGTH):

        idx = start_idx + i

        synthetic.loc[
            idx,
            "temperature"
        ] = (
            original
            + drift_values[i]
        )

        synthetic.loc[
            idx,
            "anomaly"
        ] = 1

        synthetic.loc[
            idx,
            "anomaly_type"
        ] = "temp_drift"


# ============================================================
# 22. MISSING TEMPERATURE
# ============================================================

for _ in range(MISSING_SEQUENCE_COUNT):

    start_idx = get_available_window(
        MISSING_LENGTH
    )

    for i in range(MISSING_LENGTH):

        idx = start_idx + i

        synthetic.loc[
            idx,
            "temperature"
        ] = np.nan

        synthetic.loc[
            idx,
            "anomaly"
        ] = 1

        synthetic.loc[
            idx,
            "anomaly_type"
        ] = "missing_temperature"


# ============================================================
# 23. MULTIVARIATE INCONSISTENCY
# ============================================================

for _ in range(POINT_ANOMALY_COUNT):

    idx = get_available_window(1)

    original = synthetic.loc[
        idx,
        "temperature"
    ]

    mark_point_anomaly(
        idx,
        "temperature",
        original + 10,
        "multivariate_inconsistency"
    )


# ============================================================
# 24. SORT DATA AGAIN
# ============================================================

synthetic = synthetic.sort_values(
    "timestamp"
).reset_index(drop=True)


# ============================================================
# 25. PREVIOUS VALUES
# ============================================================

synthetic["temp_prev"] = (
    synthetic["temperature"].shift(1)
)

synthetic["humidity_prev"] = (
    synthetic["humidity"].shift(1)
)

synthetic["pressure_prev"] = (
    synthetic["pressure"].shift(1)
)


# ============================================================
# 26. DIFFERENCE FEATURES
# ============================================================

synthetic["temp_diff"] = (
    synthetic["temperature"]
    - synthetic["temp_prev"]
)

synthetic["humidity_diff"] = (
    synthetic["humidity"]
    - synthetic["humidity_prev"]
)

synthetic["pressure_diff"] = (
    synthetic["pressure"]
    - synthetic["pressure_prev"]
)


# ============================================================
# 27. TIME GAP
# ============================================================

synthetic["time_gap_hours"] = (
    synthetic["timestamp"]
    .diff()
    .dt.total_seconds()
    / 3600
)


# ============================================================
# 28. RATE OF CHANGE
# ============================================================

synthetic["temp_rate"] = (
    synthetic["temp_diff"]
    / synthetic["time_gap_hours"]
)

synthetic["humidity_rate"] = (
    synthetic["humidity_diff"]
    / synthetic["time_gap_hours"]
)

synthetic["pressure_rate"] = (
    synthetic["pressure_diff"]
    / synthetic["time_gap_hours"]
)


# ============================================================
# 29. ROLLING MEAN
# ============================================================

synthetic["temp_roll_mean_3"] = (
    synthetic["temperature"]
    .rolling(3)
    .mean()
)

synthetic["humidity_roll_mean_3"] = (
    synthetic["humidity"]
    .rolling(3)
    .mean()
)

synthetic["pressure_roll_mean_3"] = (
    synthetic["pressure"]
    .rolling(3)
    .mean()
)


# ============================================================
# 30. ROLLING STANDARD DEVIATION
# ============================================================

synthetic["temp_roll_std_3"] = (
    synthetic["temperature"]
    .rolling(3)
    .std()
)

synthetic["humidity_roll_std_3"] = (
    synthetic["humidity"]
    .rolling(3)
    .std()
)

synthetic["pressure_roll_std_3"] = (
    synthetic["pressure"]
    .rolling(3)
    .std()
)


# ============================================================
# 31. TIME FEATURES
# ============================================================

synthetic["hour"] = (
    synthetic["timestamp"].dt.hour
)

synthetic["month"] = (
    synthetic["timestamp"].dt.month
)


# ============================================================
# 32. CYCLICAL TIME FEATURES
# ============================================================

synthetic["hour_sin"] = np.sin(
    2 * np.pi * synthetic["hour"] / 24
)

synthetic["hour_cos"] = np.cos(
    2 * np.pi * synthetic["hour"] / 24
)

synthetic["month_sin"] = np.sin(
    2 * np.pi * synthetic["month"] / 12
)

synthetic["month_cos"] = np.cos(
    2 * np.pi * synthetic["month"] / 12
)


# ============================================================
# 33. FINAL DATASET INFORMATION
# ============================================================

print("\n")
print("=" * 60)
print("SYNTHETIC DATASET RESULTS")
print("=" * 60)

print("\nDataset shape:")
print(synthetic.shape)

print("\nDate range:")
print(
    synthetic["timestamp"].min(),
    "to",
    synthetic["timestamp"].max()
)

print("\nNormal vs Anomaly:")
print(
    synthetic["anomaly"].value_counts()
)

print("\nAnomaly type counts:")
print(
    synthetic["anomaly_type"].value_counts()
)


# ============================================================
# 34. DISPLAY ANOMALY RECORDS
# ============================================================

print("\n")
print("=" * 60)
print("ANOMALY RECORD SAMPLE")
print("=" * 60)

anomaly_records = synthetic[
    synthetic["anomaly"] == 1
][
    [
        "timestamp",
        "temperature",
        "humidity",
        "pressure",
        "anomaly_type"
    ]
]

print(
    anomaly_records.head(50).to_string(
        index=False
    )
)


# ============================================================
# 35. MISSING VALUES
# ============================================================

print("\n")
print("=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(
    synthetic.isna().sum()
)


# ============================================================
# 36. SAVE FINAL DATASET
# ============================================================

output_file = (
    OUTPUT_DIR
    / "synthetic_anomalies_2020_2024.csv"
)

synthetic.to_csv(
    output_file,
    index=False
)


# ============================================================
# 37. SUCCESS
# ============================================================

print("\n")
print("=" * 60)
print("SUCCESS")
print("=" * 60)

print(
    "\nFinal dataset saved at:"
)

print(output_file)

print("\nFile exists:")
print(output_file.exists())