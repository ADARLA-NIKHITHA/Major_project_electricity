
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

dataset_path = Path(
    r"C:\Users\91990\OneDrive\Documents\Nikhitha B.Tech\B.Tech 4th year\MAJOR PROJECT\dataset\electricity_consumption_based_weather_dataset.csv"
)

df = pd.read_csv(dataset_path)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date")
df = df.drop_duplicates(subset="date", keep="last")
df = df.set_index("date").asfreq("D")

# Construct features using only information available before each target date
data = pd.DataFrame(index=df.index)
data["previous_day"] = df["daily_consumption"].shift(1)
data["previous_week"] = df["daily_consumption"].shift(7)
data["previous_7day_average"] = (
    df["daily_consumption"].shift(1).rolling(7).mean()
)

for col in ["AWND", "PRCP", "TMAX", "TMIN"]:
    data[col + "_previous_day"] = df[col].shift(1)

data["day_of_week"] = data.index.dayofweek
data["month"] = data.index.month
data["target"] = df["daily_consumption"]

data = data.replace([np.inf, -np.inf], np.nan).dropna()

# Reserve the final 20% as an untouched test set
test_start = int(len(data) * 0.80)
train = data.iloc[:test_start]
test = data.iloc[test_start:]

X_train = train.drop(columns="target")
y_train = train["target"]
X_test = test.drop(columns="target")
y_test = test["target"]

# Evaluate multiple chronological windows within the final test period
results = []
window_size = max(30, len(test) // 3)

for fold in range(3):
    start = fold * window_size
    end = min(start + window_size, len(test))

    if start >= len(test) or end - start < 10:
        continue

    fold_X = X_test.iloc[start:end]
    fold_y = y_test.iloc[start:end]

    model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    pred = model.predict(fold_X)
    baseline = fold_X["previous_day"].to_numpy()
    actual = fold_y.to_numpy()

    results.append({
        "Test_period_start": fold_y.index.min().date(),
        "Test_period_end": fold_y.index.max().date(),
        "Observations": len(fold_y),
        "RF_MAE": mean_absolute_error(actual, pred),
        "RF_RMSE": np.sqrt(mean_squared_error(actual, pred)),
        "Persistence_MAE": mean_absolute_error(actual, baseline),
        "Persistence_RMSE": np.sqrt(
            mean_squared_error(actual, baseline)
        ),
        "RF_R2": r2_score(actual, pred)
    })

results_df = pd.DataFrame(results)
Path("ieee_results").mkdir(exist_ok=True)
results_df.to_csv(
    "ieee_results/time_series_evaluation.csv",
    index=False
)

print("\n===== CHRONOLOGICAL TEST-WINDOW RESULTS =====")
print(results_df.round(4).to_string(index=False))
print("\nSaved to ieee_results/time_series_evaluation.csv")
