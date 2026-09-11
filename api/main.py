from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.detector import predict, reset_history


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Vayu Prahari API",
    description="Real-time weather anomaly detection API",
    version="1.0.0"
)


# ============================================================
# INPUT MODEL
# ============================================================

class WeatherReading(BaseModel):

    temperature: float | None = Field(
        default=None,
        description="Temperature in Celsius"
    )

    humidity: float | None = Field(
        default=None,
        description="Relative humidity in percent"
    )

    pressure: float | None = Field(
        default=None,
        description="Atmospheric pressure in hPa"
    )

    timestamp: datetime | None = Field(
        default=None,
        description="Timestamp of the sensor reading"
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "project": "Vayu Prahari",
        "status": "running",
        "service": "weather anomaly detection"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# RESET SENSOR HISTORY
# ============================================================

@app.post("/reset")
def reset():

    reset_history()

    return {
        "status": "history_reset"
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict_weather(
    reading: WeatherReading
):

    timestamp = (
        reading.timestamp
        if reading.timestamp is not None
        else datetime.now()
    )

    result = predict(
        temperature=reading.temperature,
        humidity=reading.humidity,
        pressure=reading.pressure,
        timestamp=timestamp
    )

    return result