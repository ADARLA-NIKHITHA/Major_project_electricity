
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# STEP 1: Load your dataset
file_path = r"C:\Users\91990\OneDrive\Documents\Nikhitha B.Tech\B.Tech 4th year\MAJOR PROJECT\dataset\electricity_consumption_based_weather_dataset.csv"

df = pd.read_csv(file_path)

# STEP 2: Prepare the data
df["date"] = pd.to_datetime(df["date"], errors="coerce")

for col in ["AWND", "PRCP", "TMAX", "TMIN", "daily_consumption"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["date", "daily_consumption"])
df = df.sort_values("date")
df = df.drop_duplicates(subset=["date"])
df = df.set_index("date")

# STEP 3: Create features for prediction
data = df.copy()

data["previous_day"] = data["daily_consumption"].shift(1)
data["previous_week"] = data["daily_consumption"].shift(7)

data["average_last_7_days"] = (
    data["daily_consumption"].shift(1).rolling(7).mean()
)

data["day_of_week"] = data.index.dayofweek
data["month"] = data.index.month

data = data.dropna()

features = [
    "AWND",
    "PRCP",
    "TMAX",
    "TMIN",
    "previous_day",
    "previous_week",
    "average_last_7_days",
    "day_of_week",
    "month"
]

X = data[features]
y = data["daily_consumption"]

# STEP 4: Split data by date (no random shuffling)
split = int(len(data) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# STEP 5: Train Random Forest
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# STEP 6: Predict electricity consumption
predictions = model.predict(X_test)

# STEP 7: Calculate evaluation metrics
actual = y_test.to_numpy()

mae = mean_absolute_error(actual, predictions)
mse = mean_squared_error(actual, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(actual, predictions)

# MAPE excludes zero or near-zero actual values
valid = np.abs(actual) > 1e-8

mape = (
    np.mean(
        np.abs((actual[valid] - predictions[valid]) / actual[valid])
    ) * 100
    if valid.any() else np.nan
)

# Symmetric mean absolute percentage error
denominator = np.abs(actual) + np.abs(predictions)
valid_s = denominator > 1e-8

smape = (
    np.mean(
        2 * np.abs(actual[valid_s] - predictions[valid_s])
        / denominator[valid_s]
    ) * 100
    if valid_s.any() else np.nan
)

print("\n===== MODEL RESULTS =====")
print("MAE:", round(mae, 4))
print("MSE:", round(mse, 4))
print("RMSE:", round(rmse, 4))
print("MAPE (%):", round(mape, 4) if np.isfinite(mape) else "N/A")
print("sMAPE (%):", round(smape, 4) if np.isfinite(smape) else "N/A")
print("R-squared:", round(r2, 4))

# STEP 8: Save results
output_folder = "ieee_results"
os.makedirs(output_folder, exist_ok=True)

metrics = pd.DataFrame([{
    "Model": "Random Forest",
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse,
    "MAPE_percent": mape,
    "sMAPE_percent": smape,
    "R2": r2
}])

metrics.to_csv(
    os.path.join(output_folder, "evaluation_metrics.csv"),
    index=False
)

# STEP 9: Plot actual vs predicted consumption
plt.figure(figsize=(12, 5))

plt.plot(
    y_test.index,
    actual,
    label="Actual consumption"
)

plt.plot(
    y_test.index,
    predictions,
    label="Predicted consumption"
)

plt.title("Actual vs Predicted Daily Electricity Consumption")
plt.xlabel("Date")
plt.ylabel("Daily consumption (dataset units)")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "actual_vs_predicted.png"),
    dpi=300
)

plt.show()

# STEP 10: Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values()

importance.to_csv(
    os.path.join(output_folder, "feature_importance.csv")
)

importance.plot(kind="barh", figsize=(9, 6))

plt.title("Factors Affecting Electricity Consumption Predictions")
plt.xlabel("Feature importance")
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "feature_importance.png"),
    dpi=300
)

plt.show()

# STEP 11: Save actual and predicted values
prediction_results = pd.DataFrame({
    "date": y_test.index,
    "actual_consumption": actual,
    "predicted_consumption": predictions
})

prediction_results.to_csv(
    os.path.join(output_folder, "predictions.csv"),
    index=False
)

print("\nAll results saved in the ieee_results folder.")
