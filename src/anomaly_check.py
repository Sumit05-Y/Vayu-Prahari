import pandas as pd
from pathlib import Path


# Find the Vayu-Prahari project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

file = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "synthetic_anomalies_2020_2024.csv"
)

print("Reading file from:")
print(file)

df = pd.read_csv(file)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df["year"] = df["timestamp"].dt.year


print("=" * 60)
print("ANOMALIES BY YEAR")
print("=" * 60)

print(
    df[df["anomaly"] == 1]["year"]
    .value_counts()
    .sort_index()
)


print("\n" + "=" * 60)
print("ANOMALY TYPES BY YEAR")
print("=" * 60)

print(
    pd.crosstab(
        df["year"],
        df["anomaly_type"]
    )
)