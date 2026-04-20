# Time Series Forecasting App

Final project for Data 5360 -- Deep Forecasting (Utah State University, Spring 2026).

A Streamlit web application for end-to-end time series forecasting with automatic
model optimization.

https://jsmith-data-5630.streamlit.app/

## Features

- CSV file upload with user-selectable date and target columns
- Automatic missing value handling (linear interpolation)
- Automatic frequency and seasonal period detection
- Four forecasting models with **auto-optimized parameters**:
  - **Holt-Winters**: tests all trend/seasonal/damping combinations, picks best AIC
  - **ARIMA / SARIMA**: stepwise search via pmdarima for optimal (p,d,q)(P,D,Q,s)
  - **Random Forest**: grid search + time series cross-validation
  - **XGBoost**: grid search + time series cross-validation
- Manual feature engineering for ML models (lag features, rolling means)
- Recursive multi-step forecasting for ML models
- Interactive Plotly visualizations
- Model comparison table with RMSE, MAE, and MAPE
- Forecast CSV export

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Usage

1. Upload a CSV file (or use the built-in Airline Passengers sample)
2. Select the date column and target variable
3. Check which models to run
4. Pick a forecast horizon (12, 24, or 36 steps)
5. Click "Run Forecast" -- all parameters are optimized automatically

## Project Structure

```
app.py              Main Streamlit application
forecasters.py      Model fitting, feature engineering, and evaluation
requirements.txt    Python dependencies
README.md           This file
sample_data/
    airline_passengers.csv
```

## How It Works

### Data Handling

- Dates are parsed automatically from the selected column
- Missing values are filled via linear interpolation
- Data frequency and seasonal period are inferred from the date index

### Train / Test Split

Data is split chronologically (80% train / 20% test), preserving temporal order
and preventing data leakage.

### Feature Engineering (ML Models)

ML models use a supervised learning approach with hand-coded features:

- **Lag features**: past values of the target (e.g., t-1 through t-12 for monthly data)
- **Rolling mean features**: moving averages over configurable windows (3, 6, 12 periods)

Future forecasts use a **recursive strategy**: each predicted value is fed back as
input for the next prediction step, avoiding the data leakage that would occur from
using actual future values.

### Model Optimization

- **Holt-Winters**: exhaustive search over trend (add/mul/none), seasonal (add/mul/none),
  and damping options. Best model selected by AIC.
- **ARIMA/SARIMA**: pmdarima stepwise algorithm searches (p,d,q) and seasonal (P,D,Q,s)
  orders, minimizing AIC.
- **Random Forest / XGBoost**: `GridSearchCV` with `TimeSeriesSplit` cross-validation
  tunes n_estimators, max_depth, and (for XGBoost) learning_rate.

### Evaluation Metrics

| Metric | Description                                                            |
| ------ | ---------------------------------------------------------------------- |
| RMSE   | Root Mean Squared Error -- penalizes large errors, same unit as target |
| MAE    | Mean Absolute Error -- average absolute deviation, same unit as target |
| MAPE   | Mean Absolute Percentage Error -- scale-independent percentage metric  |

Lower values indicate better model performance for all three metrics.

## Author

Jacob Smith -- Utah State University
