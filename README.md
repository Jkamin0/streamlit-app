# Time Series Forecasting App

Final project for Data 5360 -- Deep Forecasting (Utah State University, Spring 2026).

A Streamlit web application for end-to-end time series forecasting with automatic
model optimization.

https://jsmith-data-5630.streamlit.app/

## Features

- CSV file upload with user-selectable date and target columns
- Missing value handling (linear interpolation, forward fill, or drop)
- Optional date range filtering (last N periods)
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
- Missing values are handled via user-selected strategy: linear interpolation,
  forward fill, or dropping incomplete rows
- Users can optionally filter to the last N periods of data
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

## Design Decisions

**Why first-differencing for ML models** -- Tree-based models (Random Forest, XGBoost)
can only predict values within the range they've seen during training. A trending series
will have future values above that range. Training on period-over-period changes instead
of raw levels sidesteps this limitation, and cumulative summing reconstructs the forecast
back to the original scale.

**Why recursive forecasting instead of direct** -- A direct strategy would train a
separate model for each step in the horizon (one model for t+1, another for t+2, etc.),
which multiplies training time and can produce inconsistent step-to-step forecasts.
Recursive forecasting uses a single model and feeds each prediction back as input for
the next step, keeping the approach simple and coherent.

**Why AIC for econometric model selection** -- AIC balances goodness-of-fit against model
complexity (number of parameters). This avoids overfitting that would come from just
picking the model with the lowest training error, without requiring a separate validation
set that would further shrink an already small dataset.

**Why TimeSeriesSplit for ML cross-validation** -- Standard k-fold CV shuffles data
randomly, which would let the model train on future observations and test on past ones.
TimeSeriesSplit preserves temporal order in every fold, giving a realistic estimate of
how the model performs on unseen future data.

## Sample Dataset

The included `sample_data/airline_passengers.csv` contains the classic Box & Jenkins
airline passenger dataset: 144 monthly observations from January 1949 to December 1960.
The series exhibits a clear upward trend and strong yearly seasonality with summer peaks,
making it a good benchmark for testing trend and seasonal forecasting models.

## Author

Jacob Smith -- Utah State University
