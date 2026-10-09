
import pandas as pd
import os

file_path = r"C:\Users\91990\OneDrive\Documents\Nikhitha B.Tech\B.Tech 4th year\MAJOR PROJECT\dataset\electricity_consumption_based_weather_dataset.csv"

# Load dataset
df = pd.read_csv(file_path)

# Convert date column
df["date"] = pd.to_datetime(df["date"], errors="coerce")

print("\n===== DATASET VALIDATION =====")

# 1. Dataset size
print("\nDataset shape:", df.shape)

# 2. Column names and data types
print("\nColumn data types:")
print(df.dtypes)

# 3. Missing values
print("\nMissing values:")
print(df.isnull().sum())

# 4. Duplicate rows
print("\nDuplicate rows:", df.duplicated().sum())

# 5. Invalid dates
print("\nInvalid dates:", df["date"].isnull().sum())

# 6. Duplicate dates
print("Duplicate dates:", df["date"].duplicated().sum())

# 7. Date range
print("\nDate range:")
print("Start:", df["date"].min())
print("End:", df["date"].max())

# 8. Missing calendar dates
valid_dates = df["date"].dropna().dt.normalize()
if not valid_dates.empty:
    expected_dates = pd.date_range(
        valid_dates.min(), valid_dates.max(), freq="D"
    )
    missing_dates = expected_dates.difference(valid_dates.unique())
    print("Missing calendar dates:", len(missing_dates))
    print("First missing dates:", list(missing_dates[:10]))

# 9. Check consumption values
print("\nConsumption summary:")
print(df["daily_consumption"].describe())

print(
    "\nNegative consumption values:",
    (df["daily_consumption"] < 0).sum()
)

# Save a report
os.makedirs("ieee_results", exist_ok=True)

report = pd.DataFrame({
    "missing_values": df.isnull().sum(),
    "data_type": df.dtypes.astype(str)
})
report.to_csv("ieee_results/dataset_validation_report.csv")

print("\nValidation report saved to ieee_results/dataset_validation_report.csv")
