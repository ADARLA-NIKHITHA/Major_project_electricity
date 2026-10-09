
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error

# 1. Load your dataset
file_path = r"C:\Users\91990\OneDrive\Documents\Nikhitha B.Tech\B.Tech 4th year\MAJOR PROJECT\dataset\electricity_consumption_based_weather_dataset.csv"

df = pd.read_csv(file_path)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

columns = [
    "AWND", "PRCP", "TMAX", "TMIN", "daily_consumption"
]

for col in columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["date", "daily_consumption"])
df = df.sort_values("date")
df = df.drop_duplicates(subset=["date"])
df = df.set_index("date")

# 2. Create features
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

# 3. Chronological train/test split
split = int(len(data) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# 4. Train Random Forest
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("Test RMSE:", round(rmse, 4))

# 5. Calculate permutation importance on test data
importance_result = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="neg_root_mean_squared_error",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

importance = pd.DataFrame({
    "Feature": features,
    "Importance": importance_result.importances_mean,
    "Std": importance_result.importances_std
}).sort_values("Importance", ascending=False)

print("\nPermutation Feature Importance:")
print(importance.round(4).to_string(index=False))

# 6. Save results
output_folder = "ieee_results"
os.makedirs(output_folder, exist_ok=True)

importance.to_csv(
    os.path.join(output_folder, "permutation_importance.csv"),
    index=False
)

# 7. Generate publication-quality graph
plot_data = importance.sort_values("Importance")

plt.figure(figsize=(9, 6))
plt.barh(
    plot_data["Feature"],
    plot_data["Importance"],
    xerr=plot_data["Std"],
    capsize=3
)

plt.axvline(0, linestyle="--")
plt.xlabel("Increase in RMSE after shuffling")
plt.ylabel("Input feature")
plt.title("Permutation Feature Importance - Random Forest")
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "xai_feature_importance.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nXAI results saved in:", output_folder)
