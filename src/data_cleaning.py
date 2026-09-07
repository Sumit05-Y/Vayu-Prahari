import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_monthly_averages(year):
    sns.set_theme(style="darkgrid")
    
    # 1. Dynamically load the dataset based on the year requested
    df = pd.read_csv(f"../data/raw/{year}.csv")
    
    # 2. Clean and format the data (your exact logic)
    df['timestamp'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df = df[['timestamp', 'temp', 'rhum', 'pres']]
    df = df.rename(columns={'temp': 'temperature', 'rhum': 'humidity', 'pres': 'pressure'})
    df = df.sort_values('timestamp')
    df = df.drop_duplicates(subset=['timestamp'])
    
    # 3. Calculate monthly averages
    df['month'] = df['timestamp'].dt.month
    monthly = df.groupby('month')[['temperature', 'humidity', 'pressure']].mean()
    
    months = [
        'January', 'February', 'March', 'April', 'May', 'June', 
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    monthly.index = months

    # 4. Plot Temperature
    plt.figure(figsize=(16, 5))
    plt.plot(monthly.index, monthly['temperature'], marker='o', color='#C13A3A')
    plt.title(f'Average Monthly Temperature — {year}') # Dynamic title
    plt.xlabel('Month')
    plt.ylabel('Temperature (°C)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 5. Plot Humidity
    plt.figure(figsize=(16, 5))
    plt.plot(monthly.index, monthly['humidity'], marker='o', color='#0E7C9E')
    plt.title(f'Average Monthly Humidity — {year}')
    plt.xlabel('Month')
    plt.ylabel('Humidity (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 6. Plot Pressure
    plt.figure(figsize=(16, 5))
    plt.plot(monthly.index, monthly['pressure'], marker='o', color='#2C4870')
    plt.title(f'Average Monthly Pressure — {year}')
    plt.xlabel('Month')
    plt.ylabel('Pressure (hPa)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


for y in range(2020, 2025):
    print(f"Generating charts for {y}...")
    plot_monthly_averages(y)