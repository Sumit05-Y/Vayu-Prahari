import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# 1. PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "synthetic_anomalies_2020_2024.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("VAYU PRAHARI - 2023 MULTIVARIATE ANOMALY ANALYSIS")
print("=" * 70)

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
# 3. SELECT 2023
# ============================================================

data = df[
    df["year"] == 2023
].copy()

data = data.reset_index(
    drop=True
)


# ============================================================
# 4. CREATE NEIGHBOR VALUES
# ============================================================

data["temp_prev_raw"] = (
    data["temperature"].shift(1)
)

data["temp_next_raw"] = (
    data["temperature"].shift(-1)
)

data["humidity_prev_raw"] = (
    data["humidity"].shift(1)
)

data["humidity_next_raw"] = (
    data["humidity"].shift(-1)
)

data["pressure_prev_raw"] = (
    data["pressure"].shift(1)
)

data["pressure_next_raw"] = (
    data["pressure"].shift(-1)
)


# ============================================================
# 5. EXPECTED VALUES
# ============================================================

data["temp_expected"] = (
    (
        data["temp_prev_raw"]
        +
        data["temp_next_raw"]
    )
    / 2
)

data["humidity_expected"] = (
    (
        data["humidity_prev_raw"]
        +
        data["humidity_next_raw"]
    )
    / 2
)

data["pressure_expected"] = (
    (
        data["pressure_prev_raw"]
        +
        data["pressure_next_raw"]
    )
    / 2
)


# ============================================================
# 6. RESIDUALS
# ============================================================

data["temp_residual"] = (
    data["temperature"]
    -
    data["temp_expected"]
).abs()

data["humidity_residual"] = (
    data["humidity"]
    -
    data["humidity_expected"]
).abs()

data["pressure_residual"] = (
    data["pressure"]
    -
    data["pressure_expected"]
).abs()


# ============================================================
# 7. VALID NEIGHBOR WINDOWS
# ============================================================

data["previous_timestamp"] = (
    data["timestamp"].shift(1)
)

data["next_timestamp"] = (
    data["timestamp"].shift(-1)
)

previous_gap = (
    (
        data["timestamp"]
        -
        data["previous_timestamp"]
    )
    .dt.total_seconds()
    / 3600
)

next_gap = (
    (
        data["next_timestamp"]
        -
        data["timestamp"]
    )
    .dt.total_seconds()
    / 3600
)

valid_neighbors = (
    (previous_gap == 1)
    &
    (next_gap == 1)
)


# ============================================================
# 8. APPLY CURRENT BEST RULE
# ============================================================

current_rule = (
    valid_neighbors

    & data[
        [
            "temp_prev_raw",
            "temperature",
            "temp_next_raw"
        ]
    ]
    .notna()
    .all(axis=1)

    & data[
        [
            "humidity_prev_raw",
            "humidity",
            "humidity_next_raw"
        ]
    ]
    .notna()
    .all(axis=1)

    & data[
        [
            "pressure_prev_raw",
            "pressure",
            "pressure_next_raw"
        ]
    ]
    .notna()
    .all(axis=1)

    & (
        data["temp_residual"]
        >= 8
    )

    & (
        data["humidity_residual"]
        <= 5
    )

    & (
        data["pressure_residual"]
        <= 1
    )
)


# ============================================================
# 9. NOW SELECT MULTIVARIATE ANOMALIES
# ============================================================

anomalies = data[
    data["anomaly_type"]
    == "multivariate_inconsistency"
].copy()


print("\n")
print("=" * 70)
print("MULTIVARIATE ANOMALIES IN 2023")
print("=" * 70)

print(
    "\nNumber of 2023 multivariate anomalies:",
    len(anomalies)
)


# ============================================================
# 10. PRINT EACH ANOMALY
# ============================================================

print("\n")
print("=" * 70)
print("INDIVIDUAL MULTIVARIATE ANOMALIES")
print("=" * 70)


for index, row in anomalies.iterrows():

    print("\n")
    print("-" * 70)

    print(
        "Timestamp:",
        row["timestamp"]
    )

    print(
        "Injected type:",
        row["anomaly_type"]
    )


    print("\nCURRENT READING")

    print(
        "Temperature:",
        row["temperature"]
    )

    print(
        "Humidity:",
        row["humidity"]
    )

    print(
        "Pressure:",
        row["pressure"]
    )


    print("\nPREVIOUS READING")

    print(
        "Temperature:",
        row["temp_prev_raw"]
    )

    print(
        "Humidity:",
        row["humidity_prev_raw"]
    )

    print(
        "Pressure:",
        row["pressure_prev_raw"]
    )


    print("\nNEXT READING")

    print(
        "Temperature:",
        row["temp_next_raw"]
    )

    print(
        "Humidity:",
        row["humidity_next_raw"]
    )

    print(
        "Pressure:",
        row["pressure_next_raw"]
    )


    print("\nEXPECTED FROM NEIGHBORS")

    print(
        "Temperature:",
        row["temp_expected"]
    )

    print(
        "Humidity:",
        row["humidity_expected"]
    )

    print(
        "Pressure:",
        row["pressure_expected"]
    )


    print("\nRESIDUALS")

    print(
        "Temperature residual:",
        row["temp_residual"]
    )

    print(
        "Humidity residual:",
        row["humidity_residual"]
    )

    print(
        "Pressure residual:",
        row["pressure_residual"]
    )


    detected = bool(
        current_rule.loc[index]
    )

    print("\nCURRENT RULE:")

    if detected:

        print(
            "DETECTED"
        )

    else:

        print(
            "MISSED"
        )


# ============================================================
# 11. SUMMARY TABLE
# ============================================================

summary_columns = [
    "timestamp",

    "temperature",
    "humidity",
    "pressure",

    "temp_prev_raw",
    "temp_next_raw",

    "humidity_prev_raw",
    "humidity_next_raw",

    "pressure_prev_raw",
    "pressure_next_raw",

    "temp_expected",
    "humidity_expected",
    "pressure_expected",

    "temp_residual",
    "humidity_residual",
    "pressure_residual"
]


summary = anomalies[
    summary_columns
].copy()


summary["rule_detected"] = (
    current_rule.loc[
        anomalies.index
    ].to_numpy()
)


print("\n")
print("=" * 70)
print("SUMMARY TABLE")
print("=" * 70)

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# 12. SAVE REPORT
# ============================================================

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "day7"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


OUTPUT_FILE = (
    REPORT_DIR
    / "multivariate_analysis_2023.csv"
)


summary.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n")
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print("\nSaved:")
print(
    OUTPUT_FILE
)