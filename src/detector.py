import numpy as np
import pandas as pd
import joblib

from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "isolation_forest_final.joblib"
)

FEATURE_FILE = (
    PROJECT_ROOT
    / "models"
    / "feature_columns_final.joblib"
)


# ============================================================
# LOAD FINAL MODEL
# ============================================================

model = joblib.load(
    MODEL_FILE
)

feature_columns = joblib.load(
    FEATURE_FILE
)


# ============================================================
# FROZEN PARAMETERS
# ============================================================

HUMIDITY_THRESHOLD = 10

MULTIVARIATE_TEMP_THRESHOLD = 8
MULTIVARIATE_HUMIDITY_MAX = 5
MULTIVARIATE_PRESSURE_MAX = 1


# ============================================================
# SENSOR HISTORY
# ============================================================

history = []


# ============================================================
# FEATURE CREATION
# ============================================================

def create_features(
    temperature,
    humidity,
    pressure,
    timestamp
):
    """
    Create the 24 features required by the
    Isolation Forest model.

    At least previous readings are required
    for time-based features.
    """

    current = {
        "timestamp": pd.Timestamp(timestamp),
        "temperature": temperature,
        "humidity": humidity,
        "pressure": pressure
    }

    data = history + [current]

    df = pd.DataFrame(
        data
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    # Previous values
    df["temp_prev"] = (
        df["temperature"].shift(1)
    )

    df["humidity_prev"] = (
        df["humidity"].shift(1)
    )

    df["pressure_prev"] = (
        df["pressure"].shift(1)
    )

    # Difference
    df["temp_diff"] = (
        df["temperature"]
        -
        df["temp_prev"]
    )

    df["humidity_diff"] = (
        df["humidity"]
        -
        df["humidity_prev"]
    )

    df["pressure_diff"] = (
        df["pressure"]
        -
        df["pressure_prev"]
    )

    # Time difference
    time_diff = (
        df["timestamp"].diff()
        .dt.total_seconds()
        / 3600
    )

    # Rate
    df["temp_rate"] = (
        df["temp_diff"]
        /
        time_diff
    )

    df["humidity_rate"] = (
        df["humidity_diff"]
        /
        time_diff
    )

    df["pressure_rate"] = (
        df["pressure_diff"]
        /
        time_diff
    )

    # Rolling mean
    df["temp_roll_mean_3"] = (
        df["temperature"]
        .rolling(3)
        .mean()
    )

    df["humidity_roll_mean_3"] = (
        df["humidity"]
        .rolling(3)
        .mean()
    )

    df["pressure_roll_mean_3"] = (
        df["pressure"]
        .rolling(3)
        .mean()
    )

    # Rolling std
    df["temp_roll_std_3"] = (
        df["temperature"]
        .rolling(3)
        .std()
    )

    df["humidity_roll_std_3"] = (
        df["humidity"]
        .rolling(3)
        .std()
    )

    df["pressure_roll_std_3"] = (
        df["pressure"]
        .rolling(3)
        .std()
    )

    # Time features
    df["hour"] = (
        df["timestamp"].dt.hour
    )

    df["month"] = (
        df["timestamp"].dt.month
    )

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    return df.iloc[-1]


# ============================================================
# FROZEN SENSOR DETECTION
# ============================================================

def detect_frozen_temperature():

    if len(history) < 5:
        return False

    recent = history[-5:]

    temperatures = [
        row["temperature"]
        for row in recent
    ]

    if any(
        pd.isna(value)
        for value in temperatures
    ):
        return False

    return (
        max(temperatures)
        -
        min(temperatures)
        <= 0.05
    )


# ============================================================
# TEMPERATURE DRIFT DETECTION
# ============================================================

def detect_temperature_drift():

    if len(history) < 8:
        return False

    recent = history[-8:]

    temperatures = np.array(
        [
            row["temperature"]
            for row in recent
        ],
        dtype=float
    )

    timestamps = pd.to_datetime(
        [
            row["timestamp"]
            for row in recent
        ]
    )

    if np.isnan(
        temperatures
    ).any():
        return False

    time_diffs = (
        np.diff(timestamps)
        /
        np.timedelta64(
            1,
            "h"
        )
    )

    if not np.all(
        time_diffs == 1
    ):
        return False

    diffs = np.diff(
        temperatures
    )

    if len(diffs) == 0:
        return False

    increasing = np.all(
        diffs > 0
    )

    decreasing = np.all(
        diffs < 0
    )

    if not (
        increasing
        or decreasing
    ):
        return False

    step_sizes = np.abs(
        diffs
    )

    if not np.all(
        (
            step_sizes >= 0.6
        )
        &
        (
            step_sizes <= 1.1
        )
    ):
        return False

    if (
        step_sizes.max()
        -
        step_sizes.min()
        > 0.25
    ):
        return False

    total_change = abs(
        temperatures[-1]
        -
        temperatures[0]
    )

    if total_change < 5.5:
        return False

    x = np.arange(
        len(temperatures)
    )

    slope, intercept = np.polyfit(
        x,
        temperatures,
        1
    )

    predicted = (
        slope * x
        +
        intercept
    )

    residual_sum = np.sum(
        (
            temperatures
            -
            predicted
        ) ** 2
    )

    total_sum = np.sum(
        (
            temperatures
            -
            temperatures.mean()
        ) ** 2
    )

    if total_sum == 0:
        return False

    r2 = (
        1
        -
        residual_sum / total_sum
    )

    return r2 >= 0.995


# ============================================================
# HUMIDITY ANOMALY
# ============================================================

def detect_humidity_anomaly():

    if len(history) < 3:
        return False

    previous = history[-3]
    current = history[-2]
    next_value = history[-1]

    previous_humidity = (
        previous["humidity"]
    )

    current_humidity = (
        current["humidity"]
    )

    next_humidity = (
        next_value["humidity"]
    )

    if any(
        pd.isna(value)
        for value in [
            previous_humidity,
            current_humidity,
            next_humidity
        ]
    ):
        return False

    previous_time = pd.Timestamp(
        previous["timestamp"]
    )

    current_time = pd.Timestamp(
        current["timestamp"]
    )

    next_time = pd.Timestamp(
        next_value["timestamp"]
    )

    previous_gap = (
        current_time
        -
        previous_time
    ).total_seconds() / 3600

    next_gap = (
        next_time
        -
        current_time
    ).total_seconds() / 3600

    if (
        previous_gap != 1
        or next_gap != 1
    ):
        return False

    spike = (
        current_humidity >= 99
        and previous_humidity < 99
        and next_humidity < 99
        and (
            current_humidity
            -
            max(
                previous_humidity,
                next_humidity
            )
            >= HUMIDITY_THRESHOLD
        )
    )

    drop = (
        current_humidity <= 6
        and previous_humidity > 6
        and next_humidity > 6
        and (
            min(
                previous_humidity,
                next_humidity
            )
            -
            current_humidity
            >= HUMIDITY_THRESHOLD
        )
    )

    return bool(
        spike or drop
    )


# ============================================================
# MULTIVARIATE INCONSISTENCY
# ============================================================

def detect_multivariate_inconsistency():

    # Need at least 3 previous readings
    if len(history) < 4:
        return False

    current = history[-1]
    previous_readings = history[-4:-1]

    current_values = [
        current["temperature"],
        current["humidity"],
        current["pressure"]
    ]

    previous_values = []

    for reading in previous_readings:

        previous_values.extend(
            [
                reading["temperature"],
                reading["humidity"],
                reading["pressure"]
            ]
        )

    # Make sure required values exist
    if any(
        pd.isna(value)
        for value in current_values
    ):
        return False

    if any(
        pd.isna(value)
        for value in previous_values
    ):
        return False

    # --------------------------------------------------------
    # Check timestamps are consecutive
    # --------------------------------------------------------

    timestamps = pd.to_datetime(
        [
            reading["timestamp"]
            for reading in previous_readings
        ]
        +
        [
            current["timestamp"]
        ]
    )

    gaps = (
        np.diff(timestamps)
        /
        np.timedelta64(
            1,
            "h"
        )
    )

    if not np.all(
        gaps == 1
    ):
        return False

    # --------------------------------------------------------
    # Previous 3 readings
    # --------------------------------------------------------

    previous_temperatures = np.array(
        [
            reading["temperature"]
            for reading in previous_readings
        ],
        dtype=float
    )

    previous_humidity = np.array(
        [
            reading["humidity"]
            for reading in previous_readings
        ],
        dtype=float
    )

    previous_pressure = np.array(
        [
            reading["pressure"]
            for reading in previous_readings
        ],
        dtype=float
    )

    # --------------------------------------------------------
    # Expected current values
    #
    # Median is more robust than mean because one previous
    # anomaly should not distort the expected value too much.
    # --------------------------------------------------------

    expected_temperature = np.median(
        previous_temperatures
    )

    expected_humidity = np.median(
        previous_humidity
    )

    expected_pressure = np.median(
        previous_pressure
    )

    # --------------------------------------------------------
    # Residuals
    # --------------------------------------------------------

    temperature_residual = abs(
        current["temperature"]
        -
        expected_temperature
    )

    humidity_residual = abs(
        current["humidity"]
        -
        expected_humidity
    )

    pressure_residual = abs(
        current["pressure"]
        -
        expected_pressure
    )

    # --------------------------------------------------------
    # Multivariate anomaly
    # --------------------------------------------------------

    return bool(

        temperature_residual
        >= MULTIVARIATE_TEMP_THRESHOLD

        and humidity_residual
        <= MULTIVARIATE_HUMIDITY_MAX

        and pressure_residual
        <= MULTIVARIATE_PRESSURE_MAX
    )


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def predict(
    temperature,
    humidity,
    pressure,
    timestamp
):
    """
    Process one weather observation.

    Returns:
        dictionary containing prediction,
        anomaly type and detector information.
    """

    global history

    timestamp = pd.Timestamp(
        timestamp
    )

    reading = {
        "timestamp": timestamp,
        "temperature": temperature,
        "humidity": humidity,
        "pressure": pressure
    }

    history.append(
        reading
    )

    # Keep only recent history
    if len(history) > 20:
        history = history[-20:]

    anomaly = False
    anomaly_type = None
    detectors = []

    # --------------------------------------------------------
    # Missing temperature
    # --------------------------------------------------------

    if pd.isna(
        temperature
    ):

        anomaly = True

        anomaly_type = (
            "missing_temperature"
        )

        detectors.append(
            "missing_temperature_rule"
        )

    # --------------------------------------------------------
    # Isolation Forest
    # --------------------------------------------------------

    try:

        feature_row = (
            create_features(
                temperature,
                humidity,
                pressure,
                timestamp
            )
        )

        model_input = (
            pd.DataFrame(
                [feature_row]
            )[
                feature_columns
            ]
        )

        if (
            not model_input
            .isna()
            .any()
            .any()
        ):

            prediction = (
                model.predict(
                    model_input
                )[0]
            )

            if prediction == -1:

                anomaly = True

                detectors.append(
                    "isolation_forest"
                )

    except Exception:

        pass

    # --------------------------------------------------------
    # Frozen sensor
    # --------------------------------------------------------

    if detect_frozen_temperature():

        anomaly = True

        detectors.append(
            "frozen_temperature_rule"
        )

        if anomaly_type is None:
            anomaly_type = (
                "temp_frozen"
            )

    # --------------------------------------------------------
    # Temperature drift
    # --------------------------------------------------------

    if detect_temperature_drift():

        anomaly = True

        detectors.append(
            "temperature_drift_rule"
        )

        if anomaly_type is None:
            anomaly_type = (
                "temp_drift"
            )

    # --------------------------------------------------------
    # Humidity anomaly
    # --------------------------------------------------------

    if detect_humidity_anomaly():

        anomaly = True

        detectors.append(
            "humidity_rule"
        )

        if anomaly_type is None:
            anomaly_type = (
                "humidity_anomaly"
            )

    # --------------------------------------------------------
    # Multivariate inconsistency
    # --------------------------------------------------------

    if detect_multivariate_inconsistency():

        anomaly = True

        detectors.append(
            "multivariate_rule"
        )

        if anomaly_type is None:
            anomaly_type = (
                "multivariate_inconsistency"
            )

    # --------------------------------------------------------
    # Generic IF anomaly without specific rule
    # --------------------------------------------------------

    if (
        anomaly
        and anomaly_type is None
    ):

        anomaly_type = (
            "general_anomaly"
        )

    return {
        "timestamp": str(
            timestamp
        ),
        "temperature": temperature,
        "humidity": humidity,
        "pressure": pressure,
        "anomaly": bool(anomaly),
        "anomaly_type": anomaly_type,
        "detectors": detectors
    }


# ============================================================
# RESET HISTORY
# ============================================================

def reset_history():

    global history

    history = []