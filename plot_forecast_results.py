
import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = "ieee_results/next_day_predictions.csv"
df = pd.read_csv(file_path, parse_dates=["date"])

os.makedirs("ieee_results", exist_ok=True)

# Plot actual vs predicted consumption
plt.figure(figsize=(12, 6))

plt.plot(
    df["date"], df["actual_consumption"],
    label="Actual Consumption", linewidth=2
)

plt.plot(
    df["date"], df["persistence_prediction"],
    label="Persistence Baseline", alpha=0.8
)

plt.plot(
    df["date"], df["random_forest_prediction"],
    label="Random Forest", alpha=0.8
)

plt.title("Next-Day Electricity Consumption Forecast")
plt.xlabel("Date")
plt.ylabel("Daily Consumption (dataset units)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "ieee_results/next_day_forecast_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Graph saved to ieee_results/next_day_forecast_comparison.png")
