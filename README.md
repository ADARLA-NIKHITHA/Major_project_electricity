⚡ GenAI-Enhanced Electricity Demand Forecasting Using Deep Learning

📌 Project Overview

The GenAI-Enhanced Electricity Demand Forecasting System is a machine learning and deep learning project that predicts future electricity consumption using historical electricity demand and weather-related data.

The project combines Long Short-Term Memory (LSTM), Random Forest, and Persistence forecasting models to predict next-day electricity consumption. It evaluates model performance using standard regression metrics and uses Explainable AI (XAI) and Generative AI (GenAI) to provide understandable forecast explanations and practical energy-saving recommendations.

An interactive Streamlit dashboard allows users to explore forecasts, compare model performance, view explanations, and download results.

🎯 Objectives

Forecast next-day electricity consumption.

Implement and evaluate an LSTM deep learning model.

Compare LSTM with Random Forest and Persistence baseline models.

Evaluate predictions using MAE, MSE, RMSE, MAPE, sMAPE, and R².

Use Explainable AI techniques to understand feature importance.

Generate natural-language forecast explanations and energy-saving recommendations.

Develop an interactive dashboard using Streamlit.

🗂️ Dataset

The project uses the Electricity Consumption Based on Weather Data dataset.

The dataset contains daily electricity consumption observations and weather-related variables, including:

Date

Average wind speed (AWND)

Precipitation (PRCP)

Maximum temperature (TMAX)

Minimum temperature (TMIN)

Daily electricity consumption

The data is processed and prepared for chronological next-day forecasting. Missing dates and data quality are handled during preprocessing.

Note: Electricity consumption units should be specified only after confirming them from the dataset documentation.

🧠 Models Implemented

1. Persistence Model

Uses the previous day's electricity consumption as the prediction for the next day. It serves as a simple baseline for evaluating forecasting performance.

2. Random Forest

Uses an ensemble of decision trees with historical consumption, lagged features, weather information, and calendar features to forecast next-day electricity demand.

3. LSTM (Long Short-Term Memory)

Uses a recurrent neural network to learn temporal patterns from sequences of historical observations and forecast next-day electricity consumption.

📊 Evaluation Metrics

The models are evaluated using:

Metric

Purpose

MAE

Measures average absolute prediction error

MSE

Measures the average squared prediction error

RMSE

Measures prediction error while penalizing larger errors

MAPE

Measures average absolute percentage error

sMAPE

Measures symmetric percentage error

R²

Measures how well predictions explain variation in actual consumption

Lower error values generally indicate better predictions, while a higher R² generally indicates a better fit.

🔍 Explainable AI (XAI)

The project includes feature-importance analysis to help explain which input features contribute to forecasting performance.

The analysis uses saved feature-importance and permutation-importance results to support interpretation of the forecasting models.

Feature importance indicates predictive usefulness, not necessarily a causal relationship.

🤖 Generative AI Integration

The project includes a local LLM-based explanation module using Ollama and Llama 3.2:3b.

The module generates natural-language explanations for forecast results and suggests practical energy-saving actions.

The generated outputs are saved for review, and an automated quality-audit module checks selected output characteristics. Automated checks are only a preliminary screening and do not replace manual review.

📈 Streamlit Dashboard

The interactive dashboard provides:

Model selection and performance metrics

MAE and RMSE comparison charts

Actual-versus-predicted consumption plots

LSTM predictions and training history, when available

Forecast data tables

Energy-saving recommendations

GenAI-generated explanations

GenAI output quality-audit results

CSV hine Learning: Scikit-learn, Random Forest

Deep Learning: TensorFlow, Keras, LSTM

Data Visualization: Matplotlib

Dashboard: Streamlit

Generative AI: Ollama, Llama 3.2:3b

Development Tools: VS Code, Git, GitHubdownloads for metrics, forecasts, and recommendations


 Future Enhancements

Improve LSTM architecture and hyperparameter tuning.

Evaluate additional forecasting models.

Perform rolling-origin time-series evaluation.

Incorporate additional weather and calendar features.

Improve GenAI explanation quality and factual consistency.

Add interactive forecast horizons and historical date selection.

Explore automated model retraining and updated forecasts.
