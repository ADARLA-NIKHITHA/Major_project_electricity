
import pandas as pd
import matplotlib.pyplot as plt
import os

results = pd.read_csv("ieee_results/next_day_model_results.csv")

os.makedirs("ieee_results", exist_ok=True)

metrics = ["MAE", "RMSE", "R2"]

for metric in metrics:
    plt.figure(figsize=(7, 5))

    plt.bar(results["Model"], results[metric])
    plt.title(f"Model Comparison - {metric}")
    plt.ylabel(metric)
    plt.xlabel("Model")
    plt.tight_layout()

    filename = f"ieee_results/{metric.lower()}_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

print("Saved:")
print("ieee_results/mae_comparison.png")
print("ieee_results/rmse_comparison.png")
print("ieee_results/r2_comparison.png")
