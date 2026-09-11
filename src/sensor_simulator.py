import time
import requests
from datetime import datetime, timedelta


API_URL = "http://127.0.0.1:8000/predict"


# ============================================================
# SIMULATED WEATHER DATA
# ============================================================

readings = [
    {
        "temperature": 25.0,
        "humidity": 65,
        "pressure": 1012.0
    },
    {
        "temperature": 25.2,
        "humidity": 65,
        "pressure": 1012.1
    },
    {
        "temperature": 25.4,
        "humidity": 64,
        "pressure": 1012.0
    },
    {
        "temperature": 25.6,
        "humidity": 64,
        "pressure": 1011.9
    },

    # Simulated abnormal reading
    {
        "temperature": 45.0,
        "humidity": 64,
        "pressure": 1012.0
    },

    {
        "temperature": 25.8,
        "humidity": 63,
        "pressure": 1011.8
    },
]


# ============================================================
# START TIME
# ============================================================

current_time = datetime(
    2024,
    1,
    1,
    10,
    0,
    0
)


# ============================================================
# SEND DATA
# ============================================================

for reading in readings:

    payload = {
        **reading,
        "timestamp": current_time.isoformat()
    }

    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=5
        )

        print(
            "\nSENT:"
        )

        print(
            payload
        )

        print(
            "STATUS:",
            response.status_code
        )

        print(
            "API RESPONSE:"
        )

        print(
            response.json()
        )

    except requests.exceptions.RequestException as e:

        print(
            "\nAPI ERROR:"
        )

        print(e)

    current_time += timedelta(
        hours=1
    )

    time.sleep(2)