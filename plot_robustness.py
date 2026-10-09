
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

file_path = Path("ieee_results/time_series_evaluation.csv")
df = pd.read_csv(file_path)

labels = [
    f"{start}\nto\n{end}"
    for start, end in zip(
        df["Test_period_start"],
        df["Test_period_end"]
    )
]

x = range(len(df))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(
    [i - width / 2 for i in x],
    df["RF_RMSE"],
    width,
    label="Random Forest"
)

ax.bar(
    [i + width / 2 for i in x],
    df["Persistence_RMSE"],
    width,
    label="Persistence"
)

ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_ylabel("RMSE (dataset units)")
ax.set_title("Forecasting Robustness Across Test Periods")
ax.legend()
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()

Path("ieee_results").mkdir(exist_ok=True)
fig.savefig(
    "ieee_results/robustness_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print("Saved: ieee_results/robustness_comparison.png")
