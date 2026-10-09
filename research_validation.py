
import pandas as pd
import numpy as np
from pathlib import Path

dataset_path = Path(
    r"C:\Users\91990\OneDrive\Documents\Nikhitha B.Tech\B.Tech 4th year\MAJOR PROJECT\dataset\electricity_consumption_based_weather_dataset.csv"
)

results_folder = Path("ieee_results")
predictions_path = results_folder / "next_day_predictions.csv"
metrics_path = results_folder / "next_day_model_results.csv"

# 1. Check original dataset
df = pd.read_csv(dataset_path)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date")

print("\n===== RESEARCH VALIDATION REPORT =====")
print("\nOriginal dataset rows:", len(df))
print("Duplicate dates:", df["date"].duplicated().sum())

columns = ["AWND", "PRCP", "TMAX", "TMIN", "daily_consumption"]
print("\nMissing values by column:")
print(df[columns].isna().sum())

dates = pd.DatetimeIndex(df["date"].dt.normalize().unique()).sort_values()
expected = pd.date_range(dates.min(), dates.max(), freq="D")
missing_dates = expected.difference(dates)

print("\nMissing calendar dates:", len(missing_dates))
print("Missing dates:", list(missing_dates))

# 2. Check saved test predictions
pred = pd.read_csv(predictions_path, parse_dates=["date"])
print("\nPrediction rows:", len(pred))
print("Duplicate prediction dates:", pred["date"].duplicated().sum())
print("Missing prediction values:")
print(pred.isna().sum())

# 3. Independently recalculate key metrics
actual = pred["actual_consumption"].to_numpy()

print("\nRecalculated test metrics:")
for name, col in [
    ("Persistence", "persistence_prediction"),
    ("Random Forest", "random_forest_prediction")
]:
    forecast = pred[col].to_numpy()
    valid = np.isfinite(actual) & np.isfinite(forecast)

    mae = np.mean(np.abs(actual[valid] - forecast[valid]))
    rmse = np.sqrt(np.mean((actual[valid] - forecast[valid]) ** 2))

    print(f"{name}: MAE={mae:.4f}, RMSE={rmse:.4f}")

# 4. Check saved metric table
if metrics_path.exists():
    print("\nSaved model metrics:")
    print(pd.read_csv(metrics_path).round(4).to_string(index=False))

print("\nCheck complete.")
