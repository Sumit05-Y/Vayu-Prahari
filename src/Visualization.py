import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

years = [2020, 2021, 2022, 2023, 2024]

data = {}
all_data_frames = [] # Added to collect data for the heatmap

for year in years:
    # Update this path if you moved your raw files to ../data/raw/
    df = pd.read_csv(f"../data/raw/{year}.csv")

    df['timestamp'] = pd.to_datetime(
        df[['year', 'month', 'day', 'hour']]
    )

    df = df[['timestamp', 'temp', 'rhum', 'pres']]

    df = df.rename(columns={
        'temp': 'temperature',
        'rhum': 'humidity',
        'pres': 'pressure'
    })

    df = df.sort_values('timestamp')
    df = df.drop_duplicates(subset=['timestamp'])

    # Save the cleaned dataframe for the overall heatmap
    all_data_frames.append(df)

    df['month'] = df['timestamp'].dt.month

    monthly = df.groupby('month')[[
        'temperature',
        'humidity',
        'pressure'
    ]].mean()

    data[year] = monthly

months = [
    'January', 'February', 'March', 'April',
    'May', 'June', 'July', 'August',
    'September', 'October', 'November', 'December'
]

# --- 1. Temperature Comparison ---
plt.figure(figsize=(16, 6))
for year in years:
    plt.plot(
        months,
        data[year]['temperature'],
        marker='o',
        label=str(year)
    )
plt.title('Monthly Average Temperature Comparison (2020–2024)')
plt.xlabel('Month')
plt.ylabel('Temperature (°C)')
plt.xticks(rotation=45)
plt.legend(title='Year')
plt.tight_layout()
plt.show()

# --- 2. Humidity Comparison ---
plt.figure(figsize=(16, 6))
for year in years:
    plt.plot(
        months,
        data[year]['humidity'],
        marker='o',
        label=str(year)
    )
plt.title('Monthly Average Humidity Comparison (2020–2024)')
plt.xlabel('Month')
plt.ylabel('Humidity (%)')
plt.xticks(rotation=45)
plt.legend(title='Year')
plt.tight_layout()
plt.show()

# --- 3. Pressure Comparison ---
plt.figure(figsize=(16, 6))
for year in years:
    plt.plot(
        months,
        data[year]['pressure'],
        marker='o',
        label=str(year)
    )
plt.title('Monthly Average Pressure Comparison (2020–2024)')
plt.xlabel('Month')
plt.ylabel('Pressure (hPa)')
plt.xticks(rotation=45)
plt.legend(title='Year')
plt.tight_layout()
plt.show()

# --- 4. NEW: Overall Correlation Heatmap ---
# Combine all 5 years of data into one master dataset
combined_df = pd.concat(all_data_frames, ignore_index=True)

plt.figure(figsize=(10, 8))
# Select only the numerical columns for the correlation matrix
numeric_cols = combined_df[['temperature', 'humidity', 'pressure']]

sns.heatmap(
    numeric_cols.corr(),
    annot=True,          # Shows the exact correlation numbers inside the squares
    cmap="coolwarm",     # Red for positive correlation, blue for negative
    center=0,            # Centers the color scale at 0
    vmin=-1, vmax=1      # Sets the bounds of the correlation scale
)

plt.title('Overall Variable Correlation Heatmap (2020–2024)')
plt.tight_layout()
plt.show()