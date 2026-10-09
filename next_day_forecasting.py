
import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

file_path = r"C:\Users\91990\OneDrive\Documents\Nikhitha B.Tech\B.Tech 4th year\MAJOR PROJECT\dataset\electricity_consumption_based_weather_dataset.csv"

# 1. Load and prepare data
df = pd.read_csv(file_path)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])
df = df.sort_values("date")

df = df.drop_duplicates(subset="date", keep="last")
df = df.set_index("date")

columns = [
    "AWND", "PRCP", "TMAX", "TMIN", "daily_consumption"
]

# Create a complete daily calendar.
# Missing dates become NaN, not invented observations.
df = df[columns].asfreq("D")

# 2. Create features using past data only
data = pd.DataFrame(index=df.index)

data["consumption_previous_day"] = df["daily_consumption"].shift(1)
data["consumption_previous_week"] = df["daily_consumption"].shift(7)

data["consumption_7day_average"] = (
    df["daily_consumption"].shift(1)
    .rolling(window=7, min_periods=7)
    .mean()
)

# Use previous day's observed weather, not target-day actual weather
for col in ["AWND", "PRCP", "TMAX", "TMIN"]:
    data[col + "_previous_day"] = df[col].shift(1)

# Calendar features are known in advance
data["day_of_week"] = data.index.dayofweek
data["month"] = data.index.month

# Predict today's consumption
data["target"] = df["daily_consumption"]

# Remove rows with missing target or unavailable historical features
data = data.replace([np.inf, -np.inf], np.nan).dropna()

if len(data) < 30:
    raise ValueError(
        "Too few usable rows after removing missing historical data."
    )

X = data.drop(columns=["target"])
y = data["target"]

# 3. Chronological train/test split (no random shuffling)
split = int(len(data) * 0.80)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# 4. Persistence baseline: yesterday's consumption
baseline_predictions = X_test["consumption_previous_day"].to_numpy()

# 5. Train Random Forest
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    min_samples_leaf=2,
    n_jobs=-1
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# 6. Evaluate both models
def evaluate(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    nonzero = np.abs(actual) > 1e-8

    mape = (
        np.mean(
            np.abs(
                (actual[nonzero] - predicted[nonzero])
                / actual[nonzero]
            )
        ) * 100
        if nonzero.any() else np.nan
    )

    smape_denominator = np.abs(actual) + np.abs(predicted)
    valid = smape_denominator > 1e-8

    smape = (
        np.mean(
            2 * np.abs(actual[valid] - predicted[valid])
            / smape_denominator[valid]
        ) * 100
        if valid.any() else np.nan
    )

    return {
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE_percent": mape,
        "sMAPE_percent": smape,
        "R2": r2_score(actual, predicted)
    }

results = pd.DataFrame([
    evaluate("Persistence", y_test.to_numpy(), baseline_predictions),
    evaluate("Random Forest", y_test.to_numpy(), predictions)
])

# 7. Save results and predictions
os.makedirs("ieee_results", exist_ok=True)

results.to_csv(
    "ieee_results/next_day_model_results.csv",
    index=False
)

prediction_output = pd.DataFrame({
    "date": y_test.index,
    "actual_consumption": y_test.to_numpy(),
    "persistence_prediction": baseline_predictions,
    "random_forest_prediction": predictions
})

prediction_output.to_csv(
    "ieee_results/next_day_predictions.csv",
    index=False
)

print("\n===== NEXT-DAY FORECASTING RESULTS =====")
print("Usable daily observations:", len(data))
print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))
print("Training ends:", X_train.index[-1].date())
print("Testing starts:", X_test.index[0].date())

print("\n", results.round(4).to_string(index=False))

print("\nFiles saved:")
print("ieee_results/next_day_model_results.csv")
print("ieee_results/next_day_predictions.csv")
