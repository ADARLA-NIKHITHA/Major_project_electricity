
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "ieee_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = Path(
    r"C:\Users\91990\OneDrive\Documents\Nikhitha B.Tech\B.Tech 4th year\MAJOR PROJECT\dataset\electricity_consumption_based_weather_dataset.csv"
)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_FILE}"
    )


# If your dataset is one folder above the project,
# replace DATA_FILE with its actual path.


# --------------------------------------------------
# 2. Load and prepare dataset
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)

df["date"] = pd.to_datetime(df["date"], errors="coerce")

df["daily_consumption"] = pd.to_numeric(
    df["daily_consumption"], errors="coerce"
)

df = df.dropna(subset=["date", "daily_consumption"])
df = df.sort_values("date").drop_duplicates("date")
df = df.reset_index(drop=True)

# Convert weather columns to numeric
weather_columns = ["AWND", "PRCP", "TMAX", "TMIN"]

for col in weather_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

feature_columns = [
    col for col in weather_columns if col in df.columns
] + ["daily_consumption"]

df[feature_columns] = df[feature_columns].replace(
    [np.inf, -np.inf], np.nan
)

# Only interpolate short gaps in weather measurements.
# Do not interpolate the target consumption.
for col in weather_columns:
    if col in df.columns:
        df[col] = df[col].interpolate(
            method="linear", limit=1, limit_area="inside"
        )

df = df.dropna(subset=feature_columns).reset_index(drop=True)

if len(df) < 100:
    raise ValueError(
        "Not enough valid observations after data preparation."
    )

print("Dataset shape:", df.shape)
print("Date range:", df["date"].min(), "to", df["date"].max())


# --------------------------------------------------
# 3. Create sequences for next-day forecasting
# --------------------------------------------------

WINDOW = 14  # Use the previous 14 consecutive days

features = df[feature_columns].to_numpy(dtype=float)
target = df["daily_consumption"].to_numpy(dtype=float)
dates = df["date"].to_numpy()

X_all = []
y_all = []
target_dates = []
target_indices = []

for i in range(WINDOW, len(df)):

    # Require every day in the input window and target
    # to be consecutive calendar days.
    window_dates = pd.DatetimeIndex(
        dates[i - WINDOW:i + 1]
    )

    if len(window_dates) != WINDOW + 1:
        continue

    day_differences = np.diff(window_dates.values)

    if not np.all(
        day_differences == np.timedelta64(1, "D")
    ):
        continue

    X_all.append(features[i - WINDOW:i])
    y_all.append(target[i])
    target_dates.append(dates[i])
    target_indices.append(i)

X_all = np.asarray(X_all, dtype=float)
y_all = np.asarray(y_all, dtype=float)
target_dates = pd.to_datetime(target_dates)
target_indices = np.asarray(target_indices)

if len(X_all) < 50:
    raise ValueError(
        "Too few consecutive daily sequences. "
        "Check the dataset date gaps."
    )

# Chronological 80%/20% split.
split = int(len(X_all) * 0.8)

X_train_raw = X_all[:split]
X_test_raw = X_all[split:]

y_train = y_all[:split]
y_test = y_all[split:]

test_dates = target_dates[split:]
test_indices = target_indices[split:]

# Fit scalers on training data only.
x_scaler = MinMaxScaler()

x_scaler.fit(
    X_train_raw.reshape(
        -1, X_train_raw.shape[-1]
    )
)

def scale_sequences(data):
    shape = data.shape

    transformed = x_scaler.transform(
        data.reshape(-1, shape[-1])
    )

    return transformed.reshape(shape)

X_train = scale_sequences(X_train_raw)
X_test = scale_sequences(X_test_raw)

y_scaler = MinMaxScaler()

y_scaler.fit(y_train.reshape(-1, 1))

y_train_scaled = y_scaler.transform(
    y_train.reshape(-1, 1)
)

print("Training sequences:", len(X_train))
print("Testing sequences:", len(X_test))


# --------------------------------------------------
# 4. Build the LSTM model
# --------------------------------------------------

model = Sequential([
    LSTM(
        64,
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        ),
        return_sequences=True
    ),
    Dropout(0.2),

    LSTM(32),
    Dropout(0.2),

    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse"
)

model.summary()


# --------------------------------------------------
# 5. Train the model
# --------------------------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train_scaled,
    epochs=50,
    batch_size=32,
    validation_split=0.15,
    shuffle=False,
    callbacks=[early_stopping],
    verbose=1
)


# --------------------------------------------------
# 6. Predict electricity consumption
# --------------------------------------------------

predictions_scaled = model.predict(
    X_test, verbose=0
)

predictions = y_scaler.inverse_transform(
    predictions_scaled
).flatten()


# --------------------------------------------------
# 7. Evaluate model performance
# --------------------------------------------------

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

# Avoid unstable percentage errors when actual values
# are zero or very close to zero.
nonzero = np.abs(y_test) > 1e-8

if nonzero.any():
    mape = np.mean(
        np.abs(
            (y_test[nonzero] - predictions[nonzero])
            / y_test[nonzero]
        )
    ) * 100
else:
    mape = np.nan

denominator = np.abs(y_test) + np.abs(predictions)

valid = denominator > 1e-8

if valid.any():
    smape = np.mean(
        2 * np.abs(y_test[valid] - predictions[valid])
        / denominator[valid]
    ) * 100
else:
    smape = np.nan

r2 = r2_score(y_test, predictions)

print("\nLSTM TEST RESULTS")
print("-----------------")
print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"sMAPE: {smape:.2f}%")
print(f"R2:   {r2:.4f}")


# --------------------------------------------------
# 8. Save predictions and metrics
# --------------------------------------------------

prediction_results = pd.DataFrame({
    "date": test_dates,
    "actual_consumption": y_test,
    "lstm_prediction": predictions
})

prediction_results.to_csv(
    RESULTS_DIR / "lstm_predictions.csv",
    index=False
)

metrics_results = pd.DataFrame([{
    "Model": "LSTM",
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse,
    "MAPE_percent": mape,
    "sMAPE_percent": smape,
    "R2": r2
}])

metrics_results.to_csv(
    RESULTS_DIR / "lstm_model_results.csv",
    index=False
)

# Save training history
pd.DataFrame(history.history).to_csv(
    RESULTS_DIR / "lstm_training_history.csv",
    index=False
)


# --------------------------------------------------
# 9. Plot actual vs predicted consumption
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    test_dates,
    y_test,
    label="Actual consumption",
    linewidth=2
)

plt.plot(
    test_dates,
    predictions,
    label="LSTM prediction",
    alpha=0.85
)

plt.title("LSTM: Actual vs Predicted Electricity Consumption")
plt.xlabel("Date")
plt.ylabel("Daily consumption (dataset units)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "lstm_actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nFiles saved in:", RESULTS_DIR)
print("LSTM forecasting and testing completed.")
