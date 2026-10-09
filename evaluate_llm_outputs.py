
import pandas as pd
from pathlib import Path

input_file = Path("ieee_results/llm_explanations.csv")
output_file = Path("ieee_results/llm_quality_audit.csv")

if not input_file.exists():
    raise FileNotFoundError(
        "Run llm_explanations.py first."
    )

df = pd.read_csv(input_file)
audit = []

for _, row in df.iterrows():
    text = str(row.get("llm_explanation", "")).strip()
    words = text.split()
    lower_text = text.lower()

    checks = {
        "generation_success": (
            row.get("generation_status") == "success"
            and bool(text)
        ),
        "within_100_words": len(words) <= 100,
        "mentions_forecast": (
            "forecast" in lower_text
            or "predicted" in lower_text
            or "estimate" in lower_text
        ),
        "avoids_unsupported_weather_claim": not any(
            phrase in lower_text
            for phrase in [
                "weather caused",
                "caused by the weather",
                "due to weather"
            ]
        ),
        "includes_action_language": any(
            phrase in lower_text
            for phrase in [
                "turn off",
                "switch off",
                "unplug",
                "efficient",
                "reduce",
                "avoid",
                "use less",
                "when not in use"
            ]
        )
    }

    audit.append({
        "date": row["date"],
        "word_count": len(words),
        **checks,
        "all_automatic_checks_pass": all(checks.values())
    })

audit_df = pd.DataFrame(audit)
output_file.parent.mkdir(parents=True, exist_ok=True)
audit_df.to_csv(output_file, index=False)

print("\n===== LLM QUALITY AUDIT =====")
print(f"Total explanations: {len(audit_df)}")

for column in [
    "generation_success",
    "within_100_words",
    "mentions_forecast",
    "avoids_unsupported_weather_claim",
    "includes_action_language",
    "all_automatic_checks_pass"
]:
    passed = int(audit_df[column].sum())
    print(f"{column}: {passed}/{len(audit_df)}")

print(f"\nSaved audit to: {output_file}")
print(
    "\nNote: Automatic checks are preliminary. "
    "Manually review the explanations for factual accuracy "
    "and useful recommendations."
)
