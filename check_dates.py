
import pandas as pd

file_path = r"C:\Users\91990\OneDrive\Documents\Nikhitha B.Tech\B.Tech 4th year\MAJOR PROJECT\dataset\electricity_consumption_based_weather_dataset.csv"

df = pd.read_csv(file_path)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

print("===== DATE AND FORECAST CHECK =====")

# Check whether dates are consecutive
date_gaps = df["date"].diff().dt.days
print("\nNumber of gaps greater than one day:", (date_gaps > 1).sum())

# Show the first few dates and consumption values
print("\nFirst 10 records:")
print(df[["date", "daily_consumption"]].head(10).to_string(index=False))

# Show the last few dates
print("\nLast 5 records:")
print(df[["date", "daily_consumption"]].tail(5).to_string(index=False))

# Check the weather features used by the current model
weather_columns = ["AWND", "PRCP", "TMAX", "TMIN"]
print("\nWeather features:")
print(weather_columns)

print("\nImportant:")
print("For a day-ahead forecast, actual weather on the target day")
print("must be replaced by weather forecasts available at prediction time.")
