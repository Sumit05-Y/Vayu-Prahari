import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# Find the Vayu-Prahari project root automatically
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
# 12. GENERATE RANDOM INDICES
# ============================================================

valid_indices = np.arange(
    10,
    len(synthetic) - 20
)

rng.shuffle(valid_indices)

used_indices = set()


def get_index():

    for idx in valid_indices:

        if idx not in used_indices:

            used_indices.add(idx)

            return idx

    raise RuntimeError(
        "Not enough unused indices."
    )


# ============================================================
# 13. TEMPERATURE SPIKE
# ============================================================

idx = get_index()

synthetic.loc[
    idx,
    "temperature"
] += rng.uniform(
    15,
    25
)

synthetic.loc[
    idx,
    "anomaly"
] = 1

synthetic.loc[
    idx,
    "anomaly_type"
] = "temp_spike"


# ============================================================
# 14. TEMPERATURE DROP
# ============================================================

idx = get_index()

synthetic.loc[
    idx,
    "temperature"
] -= rng.uniform(
    15,
    25
)

synthetic.loc[
    idx,
    "anomaly"
] = 1

synthetic.loc[
    idx,
    "anomaly_type"
] = "temp_drop"


# ============================================================
# 15. HUMIDITY SPIKE
# ============================================================

idx = get_index()

synthetic.loc[
    idx,
    "humidity"
] = 100

synthetic.loc[
    idx,
    "anomaly"
] = 1

synthetic.loc[
    idx,
    "anomaly_type"
] = "humidity_spike"


# ============================================================
# 16. HUMIDITY DROP
# ============================================================

idx = get_index()

synthetic.loc[
    idx,
    "humidity"
] = 5

synthetic.loc[
    idx,
    "anomaly"
] = 1

synthetic.loc[
    idx,
    "anomaly_type"
] = "humidity_drop"


# ============================================================
# 17. PRESSURE SPIKE
# ============================================================

idx = get_index()

synthetic.loc[
    idx,
    "pressure"
] += rng.uniform(
    15,
    25
)

synthetic.loc[
    idx,
    "anomaly"
] = 1

synthetic.loc[
    idx,
    "anomaly_type"
] = "pressure_spike"


# ============================================================
# 18. PRESSURE DROP
# ============================================================

idx = get_index()

synthetic.loc[
    idx,
    "pressure"
] -= rng.uniform(
    15,
    25
)

synthetic.loc[
    idx,
    "anomaly"
] = 1

synthetic.loc[
    idx,
    "anomaly_type"
] = "pressure_drop"


# ============================================================
# 19. FROZEN / STUCK TEMPERATURE SENSOR
# ============================================================

start_idx = get_index()

frozen_value = synthetic.loc[
    start_idx,
    "temperature"
]


for i in range(5):

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
# 20. TEMPERATURE DRIFT
# ============================================================

start_idx = get_index()

drift_values = np.linspace(
    0,
    8,
    10
)


for i in range(10):

    idx = start_idx + i

    synthetic.loc[
        idx,
        "temperature"
    ] += drift_values[i]

    synthetic.loc[
        idx,
        "anomaly"
    ] = 1

    synthetic.loc[
        idx,
        "anomaly_type"
    ] = "temp_drift"


# ============================================================
# 21. MISSING TEMPERATURE
# ============================================================

start_idx = get_index()


for i in range(5):

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
# 22. MULTIVARIATE INCONSISTENCY
# ============================================================

idx = get_index()

synthetic.loc[
    idx,
    "temperature"
] += 10

synthetic.loc[
    idx,
    "anomaly"
] = 1

synthetic.loc[
    idx,
    "anomaly_type"
] = (
    "multivariate_inconsistency"
)


# ============================================================
# 23. SORT DATA AGAIN
# ============================================================

synthetic = synthetic.sort_values(
    "timestamp"
).reset_index(drop=True)


# ============================================================
# 24. PREVIOUS VALUES
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
# 25. DIFFERENCE FEATURES
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
# 26. TIME GAP
# ============================================================

synthetic["time_gap_hours"] = (
    synthetic["timestamp"]
    .diff()
    .dt.total_seconds()
    / 3600
)


# ============================================================
# 27. RATE OF CHANGE
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
# 28. ROLLING MEAN
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
# 29. ROLLING STANDARD DEVIATION
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
# 30. TIME FEATURES
# ============================================================

synthetic["hour"] = (
    synthetic["timestamp"].dt.hour
)

synthetic["month"] = (
    synthetic["timestamp"].dt.month
)


# ============================================================
# 31. CYCLICAL TIME FEATURES
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
# 32. FINAL DATASET INFORMATION
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
# 33. DISPLAY ANOMALY RECORDS
# ============================================================

print("\n")
print("=" * 60)
print("ANOMALY RECORDS")
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
    anomaly_records.to_string(
        index=False
    )
)


# ============================================================
# 34. MISSING VALUES
# ============================================================

print("\n")
print("=" * 60)
print("MISSING VALUES")
print("=" * 60)


print(
    synthetic.isna().sum()
)


# ============================================================
# 35. SAVE FINAL DATASET
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
# 36. SUCCESS
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