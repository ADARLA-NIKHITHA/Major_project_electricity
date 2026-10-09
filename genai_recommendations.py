
import pandas as pd
import os

# Load the actual forecasting predictions
file_path = "ieee_results/next_day_predictions.csv"
df = pd.read_csv(file_path, parse_dates=["date"])

# Create output folder
os.makedirs("ieee_results", exist_ok=True)

recommendations = []

for _, row in df.iterrows():
    actual = row["actual_consumption"]
    predicted = row["random_forest_prediction"]
    previous = row["persistence_prediction"]

    # Explain the forecast using model output
    if predicted > previous * 1.10:
        explanation = (
            "The forecast is more than 10% higher than "
            "the previous day's consumption."
        )
        action = (
            "Review planned appliance use and avoid running "
            "non-essential high-energy devices simultaneously."
        )

    elif predicted < previous * 0.90:
        explanation = (
            "The forecast is more than 10% lower than "
            "the previous day's consumption."
        )
        action = (
            "Continue monitoring consumption and switch off "
            "unused lights and appliances."
        )

    else:
        explanation = (
            "The forecast is within 10% of the previous "
            "day's consumption."
        )
        action = (
            "Maintain energy-saving habits and monitor "
            "daily electricity usage."
        )

    recommendations.append({
        "date": row["date"].strftime("%Y-%m-%d"),
        "previous_consumption": round(previous, 2),
        "predicted_consumption": round(predicted, 2),
        "actual_consumption_for_evaluation": round(actual, 2),
        "forecast_explanation": explanation,
        "energy_saving_recommendation": action
    })

# Save explanations and recommendations
output = pd.DataFrame(recommendations)

output.to_csv(
    "ieee_results/genai_recommendations.csv",
    index=False
)

print("\n===== FORECAST EXPLANATIONS AND RECOMMENDATIONS =====")
print(output.head(10).to_string(index=False))

print("\nSaved to ieee_results/genai_recommendations.csv")
