
import pandas as pd
import requests
from pathlib import Path
import json

input_file = Path("ieee_results/next_day_predictions.csv")
output_file = Path("ieee_results/llm_explanations.csv")

if not input_file.exists():
    raise FileNotFoundError(
        f"Cannot find {input_file}. Run next_day_forecasting.py first."
    )

df = pd.read_csv(input_file)

# Generate explanations for a small sample first.
sample = df.head(20).copy()
results = []

for _, row in sample.iterrows():
    actual = float(row["actual_consumption"])
    predicted = float(row["random_forest_prediction"])
    previous = float(row["persistence_prediction"])

    prompt = f"""
You are an assistant for an electricity demand forecasting project.

Use only these supplied values:
Date: {row["date"]}
Random Forest forecast: {predicted:.2f}
Previous-day consumption: {previous:.2f}

Write:
1. A simple explanation comparing the forecast with previous-day consumption.
2. Two practical electricity-saving suggestions.

Rules:
- Do not invent measurements, appliances, prices, or causes.
- Do not claim weather caused a change.
- Explain that the forecast is an estimate, not a guarantee.
- Keep the response under 100 words.
- Do not present suggestions as proven savings.
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2}
            },
            timeout=180
        )
        response.raise_for_status()
        explanation = response.json().get("response", "").strip()

        results.append({
            "date": row["date"],
            "actual_consumption": actual,
            "forecast": predicted,
            "previous_day_consumption": previous,
            "llm_explanation": explanation,
            "generation_status": "success"
        })

        print(f"Generated explanation for {row['date']}")

    except requests.RequestException as error:
        results.append({
            "date": row["date"],
            "actual_consumption": actual,
            "forecast": predicted,
            "previous_day_consumption": previous,
            "llm_explanation": "",
            "generation_status": f"error: {error}"
        })
        print(f"Could not process {row['date']}: {error}")

output_file.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(results).to_csv(output_file, index=False)

print("\nGenerative AI experiment completed.")
print(f"Results saved to: {output_file}")
print(f"Successful generations: {sum(r['generation_status'] == 'success' for r in results)}")
print(f"Total attempted: {len(results)}")
