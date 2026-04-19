import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import XGBRegressor
import pmdarima as pm
import warnings


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def create_lag_features(series, lags):
    """Create lag features from a pandas Series."""
    df = pd.DataFrame(index=series.index)
    for lag in lags:
        df[f"lag_{lag}"] = series.shift(lag)
    return df


def create_rolling_features(series, windows):
    """Create rolling mean features. Shifted by 1 to prevent target leakage."""
    df = pd.DataFrame(index=series.index)
    shifted = series.shift(1)
    for w in windows:
        df[f"rolling_mean_{w}"] = shifted.rolling(window=w).mean()
    return df


def build_supervised_df(series, lags, rolling_windows):
    """Build a supervised learning DataFrame from first-differenced data.

    Tree-based models cannot extrapolate beyond training range, so we train
    on period-over-period changes (diff) instead of raw levels. Predictions
    are later reconstructed back to levels via cumulative sum.

    The target column is the first difference of the original series.
    Features are lags and rolling means of the differenced series, plus
    a month-of-year feature if the index is a DatetimeIndex.

    Returns (X, y, diff_series) with NaN rows dropped.
    diff_series is the full differenced series (needed for reconstruction).
    """
    diff = series.diff()
    diff.name = "target"

    lag_df = create_lag_features(diff, lags)
    roll_df = create_rolling_features(diff, rolling_windows)
    features = pd.concat([lag_df, roll_df], axis=1)

    # Add calendar features when possible
    if hasattr(series.index, "month"):
        features["month"] = series.index.month

    combined = pd.concat([diff, features], axis=1).dropna()
    X = combined.drop(columns=["target"])
    y = combined["target"]
    return X, y, diff


def infer_seasonal_period(index):
    """Infer the seasonal period from a DatetimeIndex frequency."""
    if index.freq is None:
        freq = pd.infer_freq(index)
    else:
        freq = index.freq

    freq_str = str(freq).upper() if freq else ""

    if freq_str.startswith("Q"):
        return 4
    if freq_str.startswith("M") or freq_str.startswith("<MONTHT") or "MS" in freq_str:
        return 12
    if freq_str.startswith("W"):
        return 52
    if freq_str.startswith("D") or freq_str.startswith("<DAY"):
        return 7
    if freq_str.startswith("Y") or freq_str.startswith("A"):
        return 1
    # Default: try to detect from the data spacing
    return 12


# ---------------------------------------------------------------------------
# Econometric models -- auto-optimized
# ---------------------------------------------------------------------------

def auto_fit_exponential_smoothing(train, seasonal_period=None):
    """Try multiple Holt-Winters configurations and pick the best by AIC.

    Returns (fitted_model, model_description_string).
    """
    if seasonal_period is None:
        seasonal_period = infer_seasonal_period(train.index)

    # Don't use seasonal components if the series is too short or period is 1
    use_seasonal = seasonal_period > 1 and len(train) >= 2 * seasonal_period

    candidates = []
    configs = []

    trend_options = ["add", "mul", None]
    seasonal_options = ["add", "mul", None] if use_seasonal else [None]
    damped_options = [False, True]

    for trend in trend_options:
        for seasonal in seasonal_options:
            for damped in damped_options:
                if damped and trend is None:
                    continue
                configs.append((trend, seasonal, damped))

    best_aic = np.inf
    best_fit = None
    best_desc = ""

    for trend, seasonal, damped in configs:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sp = seasonal_period if seasonal is not None else None
                model = ExponentialSmoothing(
                    train, trend=trend, seasonal=seasonal,
                    seasonal_periods=sp, damped_trend=damped,
                )
                fit = model.fit(optimized=True)
                if fit.aic < best_aic:
                    best_aic = fit.aic
                    best_fit = fit
                    t_str = trend or "None"
                    s_str = seasonal or "None"
                    d_str = "damped" if damped else "not damped"
                    best_desc = (f"Holt-Winters (trend={t_str}, seasonal={s_str}, "
                                 f"{d_str}, sp={sp}, AIC={fit.aic:.1f})")
        except Exception:
            continue

    if best_fit is None:
        raise RuntimeError("All Exponential Smoothing configurations failed to fit.")

    return best_fit, best_desc


def auto_fit_arima(train, seasonal_period=None):
    """Use pmdarima auto_arima to find optimal ARIMA/SARIMA parameters.

    Returns (fitted_statsmodels_result, model_description_string).
    """
    if seasonal_period is None:
        seasonal_period = infer_seasonal_period(train.index)

    use_seasonal = seasonal_period > 1 and len(train) >= 2 * seasonal_period

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auto_model = pm.auto_arima(
            train,
            seasonal=use_seasonal,
            m=seasonal_period if use_seasonal else 1,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
        )

    order = auto_model.order
    seasonal_order = auto_model.seasonal_order

    # Refit with statsmodels for consistent API
    kwargs = {"endog": train, "order": order}
    if use_seasonal and seasonal_order[3] > 0:
        kwargs["seasonal_order"] = seasonal_order
    model = ARIMA(**kwargs)
    fit = model.fit()

    if use_seasonal and seasonal_order[3] > 0:
        desc = (f"SARIMA{order}x{seasonal_order} "
                f"(AIC={auto_model.aic():.1f})")
    else:
        desc = f"ARIMA{order} (AIC={auto_model.aic():.1f})"

    return fit, desc


# ---------------------------------------------------------------------------
# ML models -- auto-tuned via GridSearchCV + TimeSeriesSplit
# ---------------------------------------------------------------------------

def auto_fit_random_forest(X_train, y_train):
    """Grid search over RF hyperparameters using time series cross-validation.

    Returns (fitted_model, best_params_dict).
    """
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 20, None],
    }
    n_splits = min(3, max(2, len(X_train) // 20))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    grid = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def auto_fit_xgboost(X_train, y_train):
    """Grid search over XGBoost hyperparameters using time series cross-validation.

    Returns (fitted_model, best_params_dict).
    """
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 6, 10],
        "learning_rate": [0.01, 0.1, 0.2],
    }
    n_splits = min(3, max(2, len(X_train) // 20))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    grid = GridSearchCV(
        XGBRegressor(random_state=42, verbosity=0),
        param_grid,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def recursive_ml_forecast(model, diff_history, steps, lags, rolling_windows,
                          feature_names, last_level, start_date=None,
                          freq=None):
    """Recursive multi-step forecast for ML models trained on differenced data.

    Predicts diffs one step at a time, feeding each back into history,
    then reconstructs levels by cumulative sum from last_level.

    Returns (level_predictions, diff_predictions) as np.ndarrays.
    """
    diff_history = list(diff_history)
    diff_preds = []

    # Determine starting month if month feature is used
    use_month = "month" in feature_names
    if use_month and start_date is not None:
        current_month = start_date.month
    else:
        current_month = 1

    for step in range(steps):
        features = {}
        for lag in lags:
            features[f"lag_{lag}"] = diff_history[-lag]
        for w in rolling_windows:
            window_vals = diff_history[-(w + 1):-1]
            features[f"rolling_mean_{w}"] = (
                np.mean(window_vals) if len(window_vals) == w else np.nan
            )
        if use_month:
            features["month"] = current_month

        row = pd.DataFrame([features])[feature_names]
        pred_diff = model.predict(row)[0]
        diff_preds.append(pred_diff)
        diff_history.append(pred_diff)

        if use_month:
            current_month = current_month % 12 + 1

    diff_preds = np.array(diff_preds)
    level_preds = last_level + np.cumsum(diff_preds)
    return level_preds, diff_preds


def reconstruct_levels_from_diffs(diff_predictions, last_known_level):
    """Convert predicted diffs back to levels via cumulative sum."""
    return last_known_level + np.cumsum(diff_predictions)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_metrics(actual, predicted):
    """Compute RMSE, MAE, and MAPE."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    nonzero = actual != 0
    if nonzero.any():
        mape = np.mean(np.abs(
            (actual[nonzero] - predicted[nonzero]) / actual[nonzero]
        )) * 100
    else:
        mape = np.nan

    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}
