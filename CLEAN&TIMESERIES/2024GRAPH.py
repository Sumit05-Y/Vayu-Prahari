
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

df = pd.read_csv("../DATASET/RAW/2024.csv")

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

months = [
    'January', 'February', 'March', 'April',
    'May', 'June', 'July', 'August',
    'September', 'October', 'November', 'December'
]

monthly.index = months

plt.figure(figsize=(16, 5))
plt.plot(monthly.index, monthly['temperature'], marker='o')
plt.title('Average Monthly Temperature — 2024')
plt.xlabel('Month')
plt.ylabel('Temperature (°C)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(16, 5))
plt.plot(monthly.index, monthly['humidity'], marker='o')
plt.title('Average Monthly Humidity — 2024')
plt.xlabel('Month')
plt.ylabel('Humidity (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(16, 5))
plt.plot(monthly.index, monthly['pressure'], marker='o')
plt.title('Average Monthly Pressure — 2024')
plt.xlabel('Month')
plt.ylabel('Pressure (hPa)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
