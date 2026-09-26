import requests
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# ==========================================
# 1. CITY INPUT
# ==========================================

city = input("Enter city name: ")


# ==========================================
# 2. FIND CITY COORDINATES
# ==========================================

geo_url = "https://geocoding-api.open-meteo.com/v1/search"

geo_params = {
    "name": city,
    "count": 1,
    "language": "en",
    "format": "json"
}

geo_response = requests.get(
    geo_url,
    params=geo_params,
    timeout=10
)

geo_data = geo_response.json()

if "results" not in geo_data:
    print("City not found!")
    exit()

latitude = geo_data["results"][0]["latitude"]
longitude = geo_data["results"][0]["longitude"]
city_name = geo_data["results"][0]["name"]

print("\nCity:", city_name)
print("Latitude:", latitude)
print("Longitude:", longitude)


# ==========================================
# 3. GET REAL WEATHER DATA
# ==========================================

weather_url = "https://api.open-meteo.com/v1/forecast"

weather_params = {
    "latitude": latitude,
    "longitude": longitude,

    "hourly": (
        "temperature_2m,"
        "relative_humidity_2m,"
        "pressure_msl,"
        "wind_speed_10m,"
        "precipitation,"
        "weather_code"
    ),

    "past_days": 7,
    "forecast_days": 1,
    "timezone": "auto"
}

weather_response = requests.get(
    weather_url,
    params=weather_params,
    timeout=10
)

weather_data = weather_response.json()


# ==========================================
# 4. CREATE DATAFRAME
# ==========================================

data = pd.DataFrame({
    "Time": weather_data["hourly"]["time"],
    "Temperature": weather_data["hourly"]["temperature_2m"],
    "Humidity": weather_data["hourly"]["relative_humidity_2m"],
    "Pressure": weather_data["hourly"]["pressure_msl"],
    "Wind_Speed": weather_data["hourly"]["wind_speed_10m"],
    "Rainfall": weather_data["hourly"]["precipitation"],
    "Weather_Code": weather_data["hourly"]["weather_code"]
})


# ==========================================
# 5. SAVE DATA
# ==========================================

data.to_csv(
    "real_weather_data.csv",
    index=False
)

print("\nReal weather data saved!")
print("Total records:", len(data))


# ==========================================
# 6. WEATHER CONDITION FUNCTION
# ==========================================

def weather_condition(code):

    if code == 0:
        return "Clear Sky"

    elif code in [1, 2, 3]:
        return "Cloudy"

    elif code in [45, 48]:
        return "Fog"

    elif code in [51, 53, 55, 56, 57]:
        return "Drizzle"

    elif code in [61, 63, 65, 66, 67]:
        return "Rain"

    elif code in [71, 73, 75, 77]:
        return "Snow"

    elif code in [80, 81, 82]:
        return "Rain Showers"

    elif code in [95, 96, 99]:
        return "Thunderstorm"

    else:
        return "Unknown"


# ==========================================
# 7. CURRENT WEATHER
# ==========================================

current = data.iloc[-1]

current_condition = weather_condition(
    int(current["Weather_Code"])
)


print("\n================================")
print("       CURRENT WEATHER")
print("================================")

print("City:", city_name)

print(
    "Temperature:",
    round(current["Temperature"], 2),
    "°C"
)

print(
    "Humidity:",
    round(current["Humidity"], 2),
    "%"
)

print(
    "Pressure:",
    round(current["Pressure"], 2),
    "hPa"
)

print(
    "Wind Speed:",
    round(current["Wind_Speed"], 2),
    "km/h"
)

print(
    "Rainfall:",
    round(current["Rainfall"], 2),
    "mm"
)

print(
    "Weather Condition:",
    current_condition
)

print("================================")


# ==========================================
# 8. TEMPERATURE GRAPH
# ==========================================

plt.figure(figsize=(12, 5))

plt.plot(
    data["Time"],
    data["Temperature"]
)

plt.xlabel("Date / Time")
plt.ylabel("Temperature (°C)")
plt.title(
    f"Temperature Analysis - {city_name}"
)

# Show fewer labels
plt.xticks(
    data["Time"][::12],
    rotation=45
)

plt.tight_layout()
plt.show()


# ==========================================
# 9. HUMIDITY GRAPH
# ==========================================

plt.figure(figsize=(12, 5))

plt.plot(
    data["Time"],
    data["Humidity"]
)

plt.xlabel("Date / Time")
plt.ylabel("Humidity (%)")
plt.title(
    f"Humidity Analysis - {city_name}"
)

plt.xticks(
    data["Time"][::12],
    rotation=45
)

plt.tight_layout()
plt.show()


# ==========================================
# 10. RAINFALL GRAPH
# ==========================================

plt.figure(figsize=(12, 5))

plt.plot(
    data["Time"],
    data["Rainfall"]
)

plt.xlabel("Date / Time")
plt.ylabel("Rainfall (mm)")
plt.title(
    f"Rainfall Analysis - {city_name}"
)

plt.xticks(
    data["Time"][::12],
    rotation=45
)

plt.tight_layout()
plt.show()


# ==========================================
# 11. MACHINE LEARNING
# ==========================================

# Remove missing values
ml_data = data.dropna().copy()


# Previous hour temperature
ml_data["Previous_Temperature"] = (
    ml_data["Temperature"].shift(1)
)


# Remove first row
ml_data = ml_data.dropna()


# Input features
X = ml_data[
    [
        "Previous_Temperature",
        "Humidity",
        "Pressure",
        "Wind_Speed",
        "Rainfall"
    ]
]


# Target
y = ml_data["Temperature"]


# ==========================================
# 12. TRAIN MODEL
# ==========================================

model = LinearRegression()

model.fit(
    X,
    y
)

print("\nMachine Learning model trained successfully!")


# ==========================================
# 13. MODEL EVALUATION
# ==========================================

predictions = model.predict(X)

mae = mean_absolute_error(
    y,
    predictions
)

r2 = r2_score(
    y,
    predictions
)

print("\n================================")
print("       MODEL PERFORMANCE")
print("================================")

print(
    "Mean Absolute Error:",
    round(mae, 2),
    "°C"
)

print(
    "R2 Score:",
    round(r2, 2)
)

print("================================")


# ==========================================
# 14. NEXT-HOUR TEMPERATURE PREDICTION
# ==========================================

latest = ml_data.iloc[-1]


next_hour_input = pd.DataFrame({

    "Previous_Temperature": [
        latest["Temperature"]
    ],

    "Humidity": [
        latest["Humidity"]
    ],

    "Pressure": [
        latest["Pressure"]
    ],

    "Wind_Speed": [
        latest["Wind_Speed"]
    ],

    "Rainfall": [
        latest["Rainfall"]
    ]
})


next_hour_temperature = model.predict(
    next_hour_input
)[0]


# ==========================================
# 15. FINAL WEATHER PREDICTION
# ==========================================

print("\n================================")
print("       WEATHER PREDICTION")
print("================================")

print("City:", city_name)

print(
    "Current Temperature:",
    round(current["Temperature"], 2),
    "°C"
)

print(
    "Current Humidity:",
    round(current["Humidity"], 2),
    "%"
)

print(
    "Current Wind Speed:",
    round(current["Wind_Speed"], 2),
    "km/h"
)

print(
    "Current Rainfall:",
    round(current["Rainfall"], 2),
    "mm"
)

print(
    "Current Weather:",
    current_condition
)

print(
    "\nPredicted Next-Hour Temperature:",
    round(next_hour_temperature, 2),
    "°C"
)

print("================================")