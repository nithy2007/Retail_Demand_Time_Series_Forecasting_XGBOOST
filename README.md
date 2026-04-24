# Retail_Demand_Time_Series_Forecasting_XGBOOST

# Store Sales - Time Series Forecasting 

## Project Overview
This project focuses on predicting daily sales for thousands of product families across various stores in the "Store Sales - Time Series Forecasting" Kaggle competition. The final model achieved a **Validation RMSLE of 0.426** and a **Public Leaderboard Score of 0.447**, placing it in the **Top 200 (Top 5%)** of global participants.

The primary challenge was building a model capable of a **16-day forecast horizon** without data leakage, ensuring the solution is viable for real-world retail supply chain planning.

## Key Features & Results
* **Ranked #200** on Global Leaderboard.
* **Model:** XGBoost Regressor.
* **Validation Strategy:** Time-based split (last 15 days of training data) to ensure stability.
* **Forecast Horizon:** 16-day lead time (No "future" data leakage).

## Technical Implementation

### 1. Feature Engineering (The "Shift-16" Strategy)
To ensure the model is practically applicable, all lag and rolling features were shifted by **16 days**. This allows the model to predict the next two weeks of sales without needing the "yesterday's sales" data, which would be unavailable in a real-world deployment.
* **Seasonal Lags:** 16-day, 21-day (3 weeks), and 28-day (4 weeks) lags to capture weekly and monthly cycles.
* **Rolling Statistics:** 7, 14, and 30-day moving averages of sales to capture local trends.
* **Temporal Features:** Year, Month, Day of Week, and Weekend flags.
* **Economic Indicators:** Integration of daily oil prices (`dcoilwtico`) with forward-fill/backward-fill gap handling.

### 2. Data Preprocessing & Pipeline
* **Log Transformation:** Target variable (`sales`) transformed using `np.log1p()` to stabilize variance and minimize the RMSLE metric.
* **Missing Value Imputation:** Handled ~94% missing values in lag features for the test set by re-aligning the forecast horizon.
* **Categorical Encoding:** Label encoding for high-cardinality features like `store_nbr`, `prod_category`, and `city`.

### 3. Model Optimization
* Utilized **XGBoost** with optimized hyperparameters.
* Analyzed **Feature Importance** which identified `rolling_mean_7` and `sales_lag_21` as the primary drivers of model performance.

## Performance Summary
| Metric | Score |
| :--- | :--- |
| **Validation RMSLE** | 0.426 |
| **Kaggle Public Score** | 0.447 |
| **Leaderboard Rank** | Top 200 |

## Repository Structure
* `notebooks/`: Jupyter notebooks containing EDA and model iterations.
* `data/`: (Not included due to Kaggle terms) Raw and processed data.
* `src/`: Feature engineering scripts and model training logic.
* `submission.csv`: Final predictions for the competition.

## Future Improvements
* Implement **Target Encoding** by using the average sales of product families and store numbers combination as one feature 
