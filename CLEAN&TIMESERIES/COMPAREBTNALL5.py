
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

years = [2020, 2021, 2022, 2023, 2024]

data = {}

for year in years:
    df = pd.read_csv(f"../DATASET/RAW/{year}.csv")

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
