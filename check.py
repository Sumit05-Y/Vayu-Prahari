import pandas as pd

# Load the 2021 weather dataset
df = pd.read_csv("DATASET/RAW/2021.csv")

# Show first 5 rows
print(df.head())

# Show number of rows and columns
print(df.shape)

# Show all column names
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)

print("\nSource counts:")
print(df["temp_source"].value_counts())

print("\nBasic statistics:")
print(df[["temp", "rhum", "pres"]].describe())

print("\nTemperature sources:")
print(df["temp_source"].value_counts())

print("\nHumidity sources:")
print(df["rhum_source"].value_counts())

print("\nPressure sources:")
print(df["pres_source"].value_counts())


print("\nRows by temperature source:")
print(df.groupby("temp_source")[["temp", "rhum", "pres"]].agg(
    ["count", "min", "mean", "max"]
))


print("\nFirst record:")
print(df.iloc[0][["year", "month", "day", "hour"]])

print("\nLast record:")
print(df.iloc[-1][["year", "month", "day", "hour"]])


print("\nDuplicate timestamps:")
print(
    df.duplicated(
        subset=["year", "month", "day", "hour"]
    ).sum()
)
print("hello world")