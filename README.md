# Overrun Advisor

AI-Driven Forecasting and Mitigation for Software Project Effort Overruns

Overrun Advisor is a machine learning system designed to help project managers predict the risk of software project effort overruns and take early action. It integrates predictive modeling, explainable AI, and generative language models into a single decision-support tool.

This project was developed as part of the BUDT 751 "Harnessing AI for Business" course at the University of Maryland.

## Features

- Predicts project effort overrun percentage using a tuned XGBoost Regressor
- Provides transparent model explainability using SHAP (SHapley Additive Explanations)
- Offers actionable mitigation strategies generated via DeepSeek-R1 LLM
- Allows users to upload custom project data via a Streamlit-based user interface
- Supports dynamic visualizations for global and instance-level SHAP insights
- Enables AI-driven project risk management without requiring deep ML expertise

## Tech Stack

- **Python**: Core language for data analysis and app development
- **Pandas, NumPy**: Data manipulation and feature engineering
- **Scikit-learn**: Data preprocessing and model tuning tools
- **XGBoost**: Core regression model
- **SHAP**: Model explainability framework
- **Streamlit**: Front-end web application interface
- **Pyngrok**: Public deployment of local app for demo
- **DeepSeek-R1 (LLM)**: Generative AI for translating SHAP insights into mitigation tips
- **Matplotlib/Seaborn**: Visualizations and SHAP plots

## Dataset and Augmentation

The original dataset consisted of 499 software development projects (china.csv), each described by 20 features including size, effort, scope changes, and team characteristics.

To address the small sample size, a two-stage data augmentation strategy was implemented:

- **Stage 1: Gaussian Mixture Model (GMM)** — generated 10,000 synthetic samples with noise and bounded sampling
- **Stage 2: Empirical Sampling** — added 5,000 more samples based on jittered GMM inputs and empirical distribution of target variables

## Feature Engineering

In addition to the 16 original raw features, four domain-specific features were engineered to enhance predictive power:

- `churn_rate`: Captures team turnover and potential instability
- `pdr_ratio`: Ratio of pre-delivery rework to final effort
- `prod_nominal`: Nominal productivity, derived from effort and adjusted size
- `team_intensity`: Ratio of peak team size to project duration

These engineered features, combined with the original variables such as `AFP`, `Deleted`, `N_effort`, and others, formed the 20-feature input for the model.

## Model Details

- **Model Type**: XGBoost Regressor
- **Training Details**:
  - Stratified 80/20 train-test split by overrun bins
  - Hyperparameter tuning using RandomizedSearchCV
  - Sample weighting applied to synthetic data (weight = 5.0)

## Model Evaluation Metrics

The Overrun Advisor model was evaluated using three key regression metrics: **R² (coefficient of determination)**, **RMSE (Root Mean Squared Error)**, and **MAE (Mean Absolute Error)**. Each provides a different perspective on the model's performance.

### R² (Coefficient of Determination): 0.6257

R² measures how much of the variance in the target variable (effort overrun percentage) is explained by the model. An R² of 0.6257 means approximately 62.6% of the variation in project overruns is captured by the model.

In project management contexts—where outcomes are affected by many intangible and unmeasured factors such as team dynamics, client behavior, and organizational politics—this is a meaningful result. While not capturing all variance, the model provides strong directional guidance, especially when compared to traditional heuristic methods.

### RMSE (Root Mean Squared Error): 3.9876

RMSE quantifies the model’s prediction error in the same unit as the target variable (overrun percentage). It penalizes larger errors more heavily than smaller ones.

An RMSE of approximately 3.99 indicates that, on average, predictions deviate from actual overrun values by about ±4 percentage points. This level of accuracy is acceptable for decision support in early risk detection and planning.

### MAE (Mean Absolute Error): 1.1141

MAE provides a more interpretable view of average prediction error by calculating the mean of absolute differences between predicted and actual values.

An MAE of around 1.11 implies that, on average, the model’s predictions are off by just over one percentage point, which demonstrates its practical value for estimating risk levels and informing project decisions.

### Summary

The combination of these metrics reflects a balanced model that generalizes well and provides consistent, interpretable results. While the R² may appear modest, it is appropriate for the high-variance, real-world nature of software project management. The relatively low RMSE and MAE values confirm the model's reliability for use in operational contexts where early identification of risk is critical.


## Application Overview

The system includes a Streamlit-based web application with the following capabilities:

- Upload project datasets (CSV format)
- Automatic feature engineering on raw input
- Real-time effort overrun prediction
- SHAP visualizations (bar chart and beeswarm plot)
- LLM-based recommendations derived from SHAP outputs

## How to Run

1. Clone this repository:
git clone https://github.com/sdv1708/project-overrun-advisor.git
cd project-overrun-advisor

2. Install dependencies:
pip install -r requirements.txt

3. Launch the Streamlit app:
streamlit run app/app.py

## File Structure

- `notebooks/`: Project notebook with full modeling pipeline
- `app/`: Streamlit application
- `data/`: Sample or sanitized input CSV file
- `reports/`: Final project report
- `requirements.txt`: Python dependencies
- `.gitignore`: Standard exclusions for Python projects

## Limitations and Future Work

- The model was trained on data from a specific context and may require re-training for generalizability
- Future enhancements include integration with tools like Jira, live data ingestion, time-series deep learning, AutoML, and a full MLOps pipeline
- Additional bias detection and formal fairness audits are recommended before production deployment

