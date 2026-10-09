
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. Load your dataset
file_path = r"C:\Users\91990\OneDrive\Documents\Nikhitha B.Tech\B.Tech 4th year\MAJOR PROJECT\dataset\electricity_consumption_based_weather_dataset.csv"

df = pd.read_csv(file_path)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

for col in ["AWND", "PRCP", "TMAX", "TMIN", "daily_consumption"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["date", "daily_consumption"])
df = df.sort_values("date")
df = df.drop_duplicates(subset=["date"])
df = df.set_index("date")

# 2. Create prediction features
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
    "AWND", "PRCP", "TMAX", "TMIN",
    "previous_day", "previous_week",
    "average_last_7_days", "day_of_week", "month"
]

X = data[features]
y = data["daily_consumption"]

# 3. Split chronologically, without shuffling
split = int(len(data) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# 4. Define models
models = {
    "Ridge Regression": make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0)
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        random_state=42
    )
}

# 5. Calculate metrics
def evaluate(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    valid = np.abs(actual) > 1e-8
    mape = (
        np.mean(
            np.abs(
                (actual[valid] - predicted[valid]) / actual[valid]
            )
        ) * 100
        if valid.any() else np.nan
    )

    denominator = np.abs(actual) + np.abs(predicted)
    valid_s = denominator > 1e-8
    smape = (
        np.mean(
            2 * np.abs(actual[valid_s] - predicted[valid_s])
            / denominator[valid_s]
        ) * 100
        if valid_s.any() else np.nan
    )

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mean_squared_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "MAPE_percent": mape,
        "sMAPE_percent": smape,
        "R2": r2_score(actual, predicted)
    }

results = []

# 6. Persistence baseline
persistence_predictions = X_test["previous_day"].to_numpy()

row = {"Model": "Persistence"}
row.update(evaluate(y_test, persistence_predictions))
results.append(row)

# 7. Train and evaluate the other models
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    row = {"Model": name}
    row.update(evaluate(y_test, predictions))
    results.append(row)

# 8. Display and save comparison table
results_df = pd.DataFrame(results).sort_values("RMSE")

output_folder = "ieee_results"
os.makedirs(output_folder, exist_ok=True)

results_df.to_csv(
    os.path.join(output_folder, "model_comparison.csv"),
    index=False
)

print("\n===== MODEL COMPARISON =====")
print(results_df.round(4).to_string(index=False))

# 9. Plot RMSE comparison
plt.figure(figsize=(9, 5))
plt.bar(results_df["Model"], results_df["RMSE"])
plt.title("Comparison of Electricity Forecasting Models")
plt.xlabel("Model")
plt.ylabel("RMSE (dataset units)")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "model_comparison_rmse.png"),
    dpi=300
)
plt.show()

print("\nComparison table and graph saved in:", output_folder)
