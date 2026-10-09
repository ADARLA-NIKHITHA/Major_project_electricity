
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RESULTS = BASE_DIR / "ieee_results"

st.set_page_config(
    page_title="Electricity Demand Forecasting",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ GenAI-Enhanced Electricity Demand Forecasting")
st.write(
    "Explore electricity demand predictions, model performance, "
    "LSTM forecasts, explainable AI, GenAI explanations, "
    "and energy-saving recommendations."
)

# --------------------------------------------------
# 2. OUTPUT FILES
# --------------------------------------------------

prediction_file = RESULTS / "next_day_predictions.csv"
metrics_file = RESULTS / "next_day_model_results.csv"
recommendation_file = RESULTS / "genai_recommendations.csv"

llm_file = RESULTS / "llm_explanations.csv"
audit_file = RESULTS / "llm_quality_audit.csv"

lstm_prediction_file = RESULTS / "lstm_predictions.csv"
lstm_metrics_file = RESULTS / "lstm_model_results.csv"
lstm_history_file = RESULTS / "lstm_training_history.csv"

# --------------------------------------------------
# 3. CHECK REQUIRED FILES
# --------------------------------------------------

required_files = [
    prediction_file,
    metrics_file,
    recommendation_file
]

missing_files = [
    str(file) for file in required_files
    if not file.exists()
]

if missing_files:
    st.error("Required files are missing:")

    for file in missing_files:
        st.write(f"- {file}")

    st.info(
        "Run next_day_forecasting.py and "
        "genai_recommendations.py first."
    )
    st.stop()

# --------------------------------------------------
# 4. LOAD EXISTING RESULTS
# --------------------------------------------------

predictions = pd.read_csv(
    prediction_file,
    parse_dates=["date"]
)

metrics = pd.read_csv(metrics_file)
recommendations = pd.read_csv(recommendation_file)

llm_outputs = (
    pd.read_csv(llm_file)
    if llm_file.exists()
    else pd.DataFrame()
)

quality_audit = (
    pd.read_csv(audit_file)
    if audit_file.exists()
    else pd.DataFrame()
)

# Load LSTM results when available
lstm_predictions = (
    pd.read_csv(lstm_prediction_file)
    if lstm_prediction_file.exists()
    else pd.DataFrame()
)

lstm_metrics = (
    pd.read_csv(lstm_metrics_file)
    if lstm_metrics_file.exists()
    else pd.DataFrame()
)

lstm_history = (
    pd.read_csv(lstm_history_file)
    if lstm_history_file.exists()
    else pd.DataFrame()
)

# --------------------------------------------------
# 5. COMBINE MODEL METRICS
# --------------------------------------------------

all_metrics = metrics.copy()

if not lstm_metrics.empty:
    all_metrics = pd.concat(
        [all_metrics, lstm_metrics],
        ignore_index=True
    )

# Keep one row per model if duplicate entries exist
all_metrics = all_metrics.drop_duplicates(
    subset=["Model"],
    keep="last"
)

# --------------------------------------------------
# 6. MODEL SELECTION
# --------------------------------------------------

st.sidebar.header("Dashboard Controls")

model_names = all_metrics["Model"].astype(str).tolist()

selected_model = st.sidebar.selectbox(
    "Select forecasting model",
    model_names
)

selected_metrics = all_metrics[
    all_metrics["Model"].astype(str) == selected_model
].iloc[0]

st.sidebar.caption(
    "Choose a model to inspect its available results."
)

# --------------------------------------------------
# 7. MODEL PERFORMANCE
# --------------------------------------------------

st.header("📊 Model Performance")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "MAE",
    f"{selected_metrics['MAE']:.2f}"
)

c2.metric(
    "RMSE",
    f"{selected_metrics['RMSE']:.2f}"
)

c3.metric(
    "MAPE",
    f"{selected_metrics['MAPE_percent']:.2f}%"
)

c4.metric(
    "R²",
    f"{selected_metrics['R2']:.3f}"
)

st.caption(
    "Lower MAE, RMSE and MAPE are generally better. "
    "Higher R² is generally better. Compare models only "
    "when their test periods and evaluation methods match."
)

# --------------------------------------------------
# 8. MODEL COMPARISON
# --------------------------------------------------

st.header("🏆 Model Comparison")

comparison_columns = [
    "Model",
    "MAE",
    "MSE",
    "RMSE",
    "MAPE_percent",
    "sMAPE_percent",
    "R2"
]

available_columns = [
    column for column in comparison_columns
    if column in all_metrics.columns
]

st.dataframe(
    all_metrics[available_columns],
    use_container_width=True,
    hide_index=True
)

# Comparison chart
if {"Model", "MAE", "RMSE"}.issubset(all_metrics.columns):

    chart_metrics = all_metrics[
        ["Model", "MAE", "RMSE"]
    ].set_index("Model")

    st.subheader("MAE and RMSE Comparison")

    fig, ax = plt.subplots(figsize=(10, 5))

    chart_metrics.plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Error (dataset units)")
    ax.set_xlabel("Model")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.xticks(rotation=0)
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

# --------------------------------------------------
# 9. ACTUAL VS PREDICTED
# --------------------------------------------------

st.header("📈 Actual vs Predicted Consumption")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    predictions["date"],
    predictions["actual_consumption"],
    label="Actual",
    linewidth=2
)

if selected_model == "Random Forest":
    prediction_column = "random_forest_prediction"

    if prediction_column in predictions.columns:
        ax.plot(
            predictions["date"],
            predictions[prediction_column],
            label="Random Forest",
            alpha=0.85
        )
    else:
        st.warning(
            "Random Forest prediction column is missing."
        )

elif selected_model == "Persistence":
    prediction_column = "persistence_prediction"

    if prediction_column in predictions.columns:
        ax.plot(
            predictions["date"],
            predictions[prediction_column],
            label="Persistence",
            alpha=0.85
        )
    else:
        st.warning(
            "Persistence prediction column is missing."
        )

elif selected_model == "LSTM":

    if not lstm_predictions.empty:

        # Detect common column names in the LSTM output.
        actual_candidates = [
            "actual_consumption",
            "actual",
            "y_true"
        ]

        predicted_candidates = [
            "lstm_prediction",
            "predicted_consumption",
            "prediction",
            "predicted",
            "y_pred"
        ]

        actual_column = next(
            (
                column for column in actual_candidates
                if column in lstm_predictions.columns
            ),
            None
        )

        predicted_column = next(
            (
                column for column in predicted_candidates
                if column in lstm_predictions.columns
            ),
            None
        )

        date_column = (
            "date"
            if "date" in lstm_predictions.columns
            else None
        )

        if actual_column and predicted_column:

            x_values = (
                pd.to_datetime(lstm_predictions[date_column])
                if date_column
                else range(len(lstm_predictions))
            )

            ax.plot(
                x_values,
                lstm_predictions[actual_column],
                label="LSTM actual",
                linewidth=2
            )

            ax.plot(
                x_values,
                lstm_predictions[predicted_column],
                label="LSTM predicted",
                alpha=0.85
            )

        else:
            st.warning(
                "Could not identify the actual and predicted "
                "columns in lstm_predictions.csv. Check its "
                "column names."
            )

    else:
        st.warning(
            "LSTM predictions are not available yet. "
            "Run lstm_forecasting.py first."
        )

else:
    st.info(
        "The selected model does not have a recognized "
        "prediction column in the current dashboard."
    )

ax.set_xlabel("Date")
ax.set_ylabel("Daily consumption (dataset units)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------
# 10. LSTM TRAINING HISTORY
# --------------------------------------------------

st.header("🧠 LSTM Training History")

if not lstm_history.empty:

    st.dataframe(
        lstm_history,
        use_container_width=True,
        hide_index=True
    )

    # Plot training and validation loss when available
    train_column = next(
        (
            column for column in ["loss", "train_loss"]
            if column in lstm_history.columns
        ),
        None
    )

    validation_column = next(
        (
            column for column in [
                "val_loss",
                "validation_loss"
            ]
            if column in lstm_history.columns
        ),
        None
    )

    if train_column or validation_column:

        fig, ax = plt.subplots(figsize=(10, 4))

        if train_column:
            ax.plot(
                lstm_history[train_column],
                label="Training loss"
            )

        if validation_column:
            ax.plot(
                lstm_history[validation_column],
                label="Validation loss"
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("LSTM Training and Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

else:
    st.info(
        "LSTM training history is not available. "
        "Run lstm_forecasting.py to generate it."
    )

# --------------------------------------------------
# 11. FORECAST DATA
# --------------------------------------------------

st.header("🔮 Forecast Data")

st.subheader("Random Forest and Persistence")

st.dataframe(
    predictions,
    use_container_width=True,
    hide_index=True
)

if not lstm_predictions.empty:
    st.subheader("LSTM Predictions")

    st.dataframe(
        lstm_predictions,
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# 12. ENERGY-SAVING RECOMMENDATIONS
# --------------------------------------------------

st.header("💡 Energy-Saving Recommendations")

if recommendations.empty:
    st.info(
        "The recommendations file contains no rows."
    )
else:
    st.dataframe(
        recommendations,
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# 13. GENAI FORECAST EXPLANATIONS
# --------------------------------------------------

st.header("🤖 GenAI Forecast Explanations")

if llm_outputs.empty:
    st.warning(
        "LLM explanations are not available. "
        "Check llm_explanations.csv."
    )
else:
    st.dataframe(
        llm_outputs,
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# 14. GENAI OUTPUT QUALITY AUDIT
# --------------------------------------------------

st.header("🔍 GenAI Output Quality Audit")

if quality_audit.empty:
    st.info(
        "No GenAI quality-audit results are available."
    )
else:
    st.dataframe(
        quality_audit,
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# 15. DOWNLOAD RESULTS
# --------------------------------------------------

st.header("⬇️ Download Results")

st.download_button(
    "Download combined model metrics",
    data=all_metrics.to_csv(index=False),
    file_name="combined_model_metrics.csv",
    mime="text/csv"
)

st.download_button(
    "Download Random Forest and Persistence forecasts",
    data=predictions.to_csv(index=False),
    file_name="forecast_predictions.csv",
    mime="text/csv"
)

st.download_button(
    "Download recommendations",
    data=recommendations.to_csv(index=False),
    file_name="forecast_recommendations.csv",
    mime="text/csv"
)

if not lstm_predictions.empty:
    st.download_button(
        "Download LSTM predictions",
        data=lstm_predictions.to_csv(index=False),
        file_name="lstm_predictions.csv",
        mime="text/csv"
    )

if not lstm_history.empty:
    st.download_button(
        "Download LSTM training history",
        data=lstm_history.to_csv(index=False),
        file_name="lstm_training_history.csv",
        mime="text/csv"
    )

if not llm_outputs.empty:
    st.download_button(
        "Download GenAI explanations",
        data=llm_outputs.to_csv(index=False),
        file_name="llm_explanations.csv",
        mime="text/csv"
    )

if not quality_audit.empty:
    st.download_button(
        "Download GenAI quality audit",
        data=quality_audit.to_csv(index=False),
        file_name="llm_quality_audit.csv",
        mime="text/csv"
    )

st.divider()

st.caption(
    "GenAI-Enhanced Electricity Demand Forecasting | "
    "Research prototype. Forecasts and recommendations "
    "should be reviewed before practical use."
)
