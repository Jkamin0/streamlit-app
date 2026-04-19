import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import time

from forecasters import (
    build_supervised_df,
    auto_fit_exponential_smoothing,
    auto_fit_arima,
    auto_fit_random_forest,
    auto_fit_xgboost,
    recursive_ml_forecast,
    reconstruct_levels_from_diffs,
    compute_metrics,
    infer_seasonal_period,
)

# ---------------------------------------------------------------------------
# Page config & session state init
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Time Series Forecasting", layout="wide")

PAGES = ["Data Explorer", "Forecast Results", "Model Comparison", "About"]

if "page" not in st.session_state:
    st.session_state.page = "Data Explorer"
if "should_run" not in st.session_state:
    st.session_state.should_run = False

# ---------------------------------------------------------------------------
# Sidebar -- minimal user inputs
# ---------------------------------------------------------------------------

st.sidebar.header("Data")

uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample dataset (Airline Passengers)")

df_raw = None
if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
elif use_sample:
    sample_path = os.path.join(os.path.dirname(__file__),
                               "sample_data", "airline_passengers.csv")
    df_raw = pd.read_csv(sample_path)

if df_raw is not None:
    all_columns = df_raw.columns.tolist()
    date_col = st.sidebar.selectbox("Date / time column", all_columns)

    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.sidebar.error("No numeric columns found in the data.")
        st.stop()
    target_col = st.sidebar.selectbox("Target variable", numeric_cols)

    missing_method = st.sidebar.selectbox(
        "Missing value handling",
        ["Interpolate (linear)", "Forward Fill", "Drop Rows"],
    )
    last_n = st.sidebar.number_input(
        "Use last N periods (0 = all)", min_value=0, value=0, step=1,
    )

    st.sidebar.header("Models")
    use_hw = st.sidebar.checkbox("Holt-Winters (Exponential Smoothing)", value=True)
    use_arima = st.sidebar.checkbox("ARIMA / SARIMA", value=True)
    use_rf = st.sidebar.checkbox("Random Forest", value=False)
    use_xgb = st.sidebar.checkbox("XGBoost", value=False)

    if not any([use_hw, use_arima, use_rf, use_xgb]):
        st.sidebar.warning("Select at least one model.")

    st.sidebar.header("Forecast")
    horizon = st.sidebar.selectbox("Forecast horizon (steps ahead)",
                                   [12, 24, 36], index=0)

    run_btn = st.sidebar.button("Run Forecast", type="primary")

    # When run is clicked, switch to Forecast Results and set the run flag
    if run_btn:
        st.session_state.page = "Forecast Results"
        st.session_state.should_run = True
        # Store model selections so they survive the rerun
        st.session_state.run_config = {
            "use_hw": use_hw, "use_arima": use_arima,
            "use_rf": use_rf, "use_xgb": use_xgb,
            "horizon": horizon,
        }
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "All model parameters are automatically optimized at runtime. "
        "Holt-Winters selects the best trend/seasonal config by AIC. "
        "ARIMA uses stepwise search for optimal (p,d,q)(P,D,Q,s). "
        "ML models are tuned via grid search with time series cross-validation."
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("Time Series Forecasting App")

if df_raw is None:
    st.info("Upload a CSV file or select the sample dataset to get started.")
    st.stop()

# --- Parse and clean data ---
df = df_raw.copy()
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col])
df = df.sort_values(date_col).set_index(date_col)

series_full = df[target_col].copy()

n_missing = series_full.isna().sum()
if n_missing > 0:
    if missing_method == "Interpolate (linear)":
        series_full = series_full.interpolate(method="linear").ffill().bfill()
    elif missing_method == "Forward Fill":
        series_full = series_full.ffill().bfill()
    else:
        series_full = series_full.dropna()
    st.sidebar.caption(f"{n_missing} missing value(s) handled via {missing_method}.")

inferred_freq = pd.infer_freq(series_full.index)
if inferred_freq:
    series_full.index.freq = inferred_freq
else:
    median_delta = series_full.index.to_series().diff().median()
    if median_delta <= pd.Timedelta(days=2):
        series_full = series_full.asfreq("D")
    elif median_delta <= pd.Timedelta(days=8):
        series_full = series_full.asfreq("W")
    elif median_delta <= pd.Timedelta(days=35):
        series_full = series_full.asfreq("MS")
    elif median_delta <= pd.Timedelta(days=100):
        series_full = series_full.asfreq("QS")
    else:
        series_full = series_full.asfreq("YS")
    if missing_method == "Interpolate (linear)":
        series_full = series_full.interpolate(method="linear").ffill().bfill()
    elif missing_method == "Forward Fill":
        series_full = series_full.ffill().bfill()
    else:
        series_full = series_full.dropna()

freq = series_full.index.freq

if last_n > 0:
    series_full = series_full.iloc[-last_n:]

if len(series_full) < 10:
    st.error("Not enough data points. Need at least 10 observations.")
    st.stop()

train_pct = 0.80
split_idx = int(len(series_full) * train_pct)
train_series = series_full.iloc[:split_idx]
test_series = series_full.iloc[split_idx:]

seasonal_period = infer_seasonal_period(series_full.index)

ml_lags = list(range(1, min(seasonal_period, 12) + 1))
ml_rolling = [w for w in [3, 6, 12] if w <= len(train_series) // 3]
if not ml_rolling:
    ml_rolling = [3]

# ---------------------------------------------------------------------------
# Page navigation (replaces st.tabs for programmatic control)
# ---------------------------------------------------------------------------

# Sync: if we programmatically changed the page (e.g. via Run Forecast),
# push that into the widget key so the radio renders on the correct page.
if "nav_radio" not in st.session_state:
    st.session_state.nav_radio = st.session_state.page
elif st.session_state.nav_radio != st.session_state.page:
    st.session_state.nav_radio = st.session_state.page

def _on_nav_change():
    st.session_state.page = st.session_state.nav_radio

current_page = st.radio(
    "Navigate",
    PAGES,
    horizontal=True,
    key="nav_radio",
    on_change=_on_nav_change,
    label_visibility="collapsed",
)
current_page = st.session_state.page

st.markdown("---")


# ===== PAGE: Data Explorer =====
if current_page == "Data Explorer":
    st.subheader("Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Observations", len(series_full))
    col2.metric("Start", str(series_full.index.min().date()))
    col3.metric("End", str(series_full.index.max().date()))
    col4.metric("Frequency", str(freq))

    st.subheader("Descriptive Statistics")
    st.dataframe(series_full.describe().to_frame().T, use_container_width=True)

    st.subheader("Time Series Plot")
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=train_series.index, y=train_series.values,
        mode="lines", name="Training",
        line=dict(color="#1f77b4"),
    ))
    fig_ts.add_trace(go.Scatter(
        x=test_series.index, y=test_series.values,
        mode="lines", name="Test",
        line=dict(color="#ff7f0e"),
    ))
    fig_ts.update_layout(
        xaxis_title="Date", yaxis_title=target_col,
        hovermode="x unified", height=400,
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    st.caption(f"Train: {len(train_series)} obs | Test: {len(test_series)} obs "
               f"(80/20 split) | Seasonal period: {seasonal_period}")

    with st.expander("Raw Data Preview"):
        st.dataframe(series_full.head(30).to_frame(), use_container_width=True)


# ===== PAGE: Forecast Results =====
elif current_page == "Forecast Results":

    # --- Run models if triggered ---
    if st.session_state.should_run:
        cfg = st.session_state.run_config
        use_hw = cfg["use_hw"]
        use_arima = cfg["use_arima"]
        use_rf = cfg["use_rf"]
        use_xgb = cfg["use_xgb"]
        horizon = cfg["horizon"]

        if not any([use_hw, use_arima, use_rf, use_xgb]):
            st.warning("Select at least one model in the sidebar.")
            st.session_state.should_run = False
            st.stop()

        results = {}
        model_info = {}

        # Count total steps for progress
        total_steps = sum([use_hw, use_arima, use_rf or use_xgb, use_rf, use_xgb])
        progress = {"done": 0}

        progress_bar = st.progress(0)
        status = st.status("Running forecasts...", expanded=True)

        with status:
            # --- Holt-Winters ---
            if use_hw:
                st.write("Optimizing Holt-Winters -- testing all trend/seasonal "
                         "combinations...")
                try:
                    fitted_hw, hw_desc = auto_fit_exponential_smoothing(
                        train_series, seasonal_period=seasonal_period,
                    )
                    hw_test_pred = fitted_hw.forecast(len(test_series))
                    hw_test_pred.index = test_series.index
                    hw_future = fitted_hw.forecast(len(test_series) + horizon)
                    hw_future = hw_future.iloc[len(test_series):]
                    metrics_hw = compute_metrics(test_series.values,
                                                 hw_test_pred.values)
                    results["Holt-Winters"] = {
                        "test_pred": hw_test_pred,
                        "future_pred": hw_future,
                        "metrics": metrics_hw,
                    }
                    model_info["Holt-Winters"] = hw_desc
                    st.write(f"  Holt-Winters complete -- "
                             f"RMSE: {metrics_hw['RMSE']:.4f}, "
                             f"MAE: {metrics_hw['MAE']:.4f}, "
                             f"MAPE: {metrics_hw['MAPE']:.2f}%")
                    st.write(f"  Selected: {hw_desc}")
                except Exception as e:
                    st.write(f"  Holt-Winters failed: {e}")
                progress["done"] += 1
                progress_bar.progress(progress["done"] / total_steps)

            # --- ARIMA ---
            if use_arima:
                st.write("Optimizing ARIMA -- stepwise parameter search via "
                         "pmdarima...")
                try:
                    fitted_ar, ar_desc = auto_fit_arima(
                        train_series, seasonal_period=seasonal_period,
                    )
                    ar_test_pred = fitted_ar.forecast(len(test_series))
                    ar_test_pred.index = test_series.index
                    ar_future = fitted_ar.forecast(len(test_series) + horizon)
                    ar_future = ar_future.iloc[len(test_series):]
                    metrics_ar = compute_metrics(test_series.values,
                                                 ar_test_pred.values)
                    model_label = ar_desc.split(" (")[0]
                    results[model_label] = {
                        "test_pred": ar_test_pred,
                        "future_pred": ar_future,
                        "metrics": metrics_ar,
                    }
                    model_info[model_label] = ar_desc
                    st.write(f"  ARIMA complete -- "
                             f"RMSE: {metrics_ar['RMSE']:.4f}, "
                             f"MAE: {metrics_ar['MAE']:.4f}, "
                             f"MAPE: {metrics_ar['MAPE']:.2f}%")
                    st.write(f"  Selected: {ar_desc}")
                except Exception as e:
                    st.write(f"  ARIMA failed: {e}")
                progress["done"] += 1
                progress_bar.progress(progress["done"] / total_steps)

            # --- ML shared setup ---
            ml_ready = False
            if use_rf or use_xgb:
                st.write("Building ML features -- differencing + lags + rolling "
                         "means...")
                try:
                    X_all, y_all, diff_series = build_supervised_df(
                        series_full, ml_lags, ml_rolling,
                    )
                    feature_names = X_all.columns.tolist()

                    split_date = train_series.index[-1]
                    X_train_ml = X_all.loc[X_all.index <= split_date]
                    y_train_ml = y_all.loc[y_all.index <= split_date]
                    X_test_ml = X_all.loc[X_all.index > split_date]
                    y_test_ml = y_all.loc[y_all.index > split_date]

                    test_level_index = y_test_ml.index
                    prev_levels = series_full.shift(1).loc[test_level_index]
                    diff_history_for_recursive = (
                        diff_series.dropna().values.tolist()
                    )
                    ml_ready = True
                    st.write(f"  Features: {feature_names}")
                except Exception as e:
                    st.write(f"  ML feature engineering failed: {e}")
                progress["done"] += 1
                progress_bar.progress(progress["done"] / total_steps)

            def _run_ml(name, fit_func):
                st.write(f"Optimizing {name} -- grid search with time series "
                         "cross-validation...")
                try:
                    fitted_model, best_params = fit_func(X_train_ml, y_train_ml)

                    test_diff_preds = fitted_model.predict(X_test_ml)
                    test_level_preds = prev_levels.values + test_diff_preds
                    test_pred_series = pd.Series(test_level_preds,
                                                 index=test_level_index)

                    last_level = series_full.iloc[-1]
                    last_date = series_full.index[-1]
                    future_idx = pd.date_range(
                        start=last_date, periods=horizon + 1, freq=freq,
                    )[1:]
                    future_levels, _ = recursive_ml_forecast(
                        fitted_model, diff_history_for_recursive, horizon,
                        ml_lags, ml_rolling, feature_names,
                        last_level=last_level,
                        start_date=future_idx[0], freq=freq,
                    )
                    future_series = pd.Series(future_levels, index=future_idx)

                    actual_test = series_full.loc[test_level_index].values
                    metrics = compute_metrics(actual_test, test_level_preds)

                    results[name] = {
                        "test_pred": test_pred_series,
                        "future_pred": future_series,
                        "metrics": metrics,
                    }
                    model_info[name] = (
                        f"{name} (trained on diffs, {best_params}, "
                        f"features: {feature_names})"
                    )
                    st.write(f"  {name} complete -- "
                             f"RMSE: {metrics['RMSE']:.4f}, "
                             f"MAE: {metrics['MAE']:.4f}, "
                             f"MAPE: {metrics['MAPE']:.2f}%")
                    st.write(f"  Best params: {best_params}")
                except Exception as e:
                    st.write(f"  {name} failed: {e}")
                progress["done"] += 1
                progress_bar.progress(progress["done"] / total_steps)

            if use_rf and ml_ready:
                _run_ml("Random Forest", auto_fit_random_forest)
            if use_xgb and ml_ready:
                _run_ml("XGBoost", auto_fit_xgboost)

        # Finalize
        status.update(label="All forecasts complete!", state="complete",
                      expanded=False)
        progress_bar.empty()

        st.session_state["results"] = results
        st.session_state["model_info"] = model_info
        st.session_state["train_series"] = train_series
        st.session_state["test_series"] = test_series
        st.session_state["target_col"] = target_col
        st.session_state["horizon"] = horizon
        st.session_state.should_run = False

    # --- Display results (always, from session state) ---
    if "results" not in st.session_state or not st.session_state["results"]:
        if not st.session_state.should_run:
            st.info("Select models in the sidebar and click 'Run Forecast'.")
    else:
        results = st.session_state["results"]
        m_info = st.session_state.get("model_info", {})
        train_s = st.session_state["train_series"]
        test_s = st.session_state["test_series"]
        tgt = st.session_state["target_col"]

        for model_name, res in results.items():
            st.subheader(model_name)

            if model_name in m_info:
                st.caption(f"Auto-selected: {m_info[model_name]}")

            m = res["metrics"]
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("RMSE", f"{m['RMSE']:.4f}")
            mc2.metric("MAE", f"{m['MAE']:.4f}")
            mc3.metric("MAPE", f"{m['MAPE']:.2f}%")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=train_s.index, y=train_s.values,
                mode="lines", name="Train",
                line=dict(color="#1f77b4"),
            ))
            fig.add_trace(go.Scatter(
                x=test_s.index, y=test_s.values,
                mode="lines", name="Actual (Test)",
                line=dict(color="#2ca02c"),
            ))
            fig.add_trace(go.Scatter(
                x=res["test_pred"].index, y=res["test_pred"].values,
                mode="lines", name="Predicted (Test)",
                line=dict(color="#ff7f0e", dash="dash"),
            ))
            fig.add_trace(go.Scatter(
                x=res["future_pred"].index, y=res["future_pred"].values,
                mode="lines", name="Future Forecast",
                line=dict(color="#d62728", width=2),
            ))
            fig.update_layout(
                xaxis_title="Date", yaxis_title=tgt,
                hovermode="x unified", height=400,
                title=f"{model_name} -- Actual vs Forecast",
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Predicted vs Actual (Test Set)"):
                if len(res["test_pred"]) < len(test_s):
                    detail = pd.DataFrame({
                        "Actual": [test_s.loc[d] if d in test_s.index else np.nan
                                   for d in res["test_pred"].index],
                        "Predicted": res["test_pred"].values,
                    }, index=res["test_pred"].index)
                else:
                    detail = pd.DataFrame({
                        "Actual": test_s.values,
                        "Predicted": res["test_pred"].values,
                    }, index=test_s.index)
                detail["Error"] = detail["Actual"] - detail["Predicted"]
                st.dataframe(detail, use_container_width=True)

            csv_data = res["future_pred"].to_frame(name="Forecast").to_csv()
            st.download_button(
                f"Download {model_name} future forecast CSV",
                data=csv_data,
                file_name=f"forecast_{model_name.replace(' ', '_').lower()}.csv",
                mime="text/csv",
                key=f"dl_{model_name}",
            )
            st.divider()


# ===== PAGE: Model Comparison =====
elif current_page == "Model Comparison":
    if "results" not in st.session_state or not st.session_state["results"]:
        st.info("Run at least one model to see comparisons.")
    else:
        results = st.session_state["results"]
        train_s = st.session_state["train_series"]
        test_s = st.session_state["test_series"]
        tgt = st.session_state["target_col"]

        st.subheader("Metrics Comparison")
        comp_rows = []
        for name, res in results.items():
            m = res["metrics"]
            comp_rows.append({"Model": name, "RMSE": m["RMSE"], "MAE": m["MAE"],
                              "MAPE (%)": m["MAPE"]})
        comp_df = pd.DataFrame(comp_rows).set_index("Model")

        def highlight_best(s):
            is_best = s == s.min()
            return ["background-color: #d4edda" if v else "" for v in is_best]

        st.dataframe(comp_df.style.apply(highlight_best), use_container_width=True)

        st.subheader("All Models -- Test Predictions vs Actual")
        fig_all = go.Figure()
        fig_all.add_trace(go.Scatter(
            x=test_s.index, y=test_s.values,
            mode="lines", name="Actual",
            line=dict(color="#2ca02c", width=2),
        ))
        colors = ["#ff7f0e", "#d62728", "#9467bd", "#8c564b"]
        for i, (name, res) in enumerate(results.items()):
            fig_all.add_trace(go.Scatter(
                x=res["test_pred"].index, y=res["test_pred"].values,
                mode="lines", name=name,
                line=dict(color=colors[i % len(colors)], dash="dash"),
            ))
        fig_all.update_layout(
            xaxis_title="Date", yaxis_title=tgt,
            hovermode="x unified", height=400,
        )
        st.plotly_chart(fig_all, use_container_width=True)

        st.subheader("Metrics Bar Chart")
        bar_df = comp_df.reset_index().melt(id_vars="Model", var_name="Metric",
                                            value_name="Value")
        fig_bar = px.bar(bar_df, x="Model", y="Value", color="Metric",
                         barmode="group")
        fig_bar.update_layout(height=350)
        st.plotly_chart(fig_bar, use_container_width=True)


# ===== PAGE: About =====
elif current_page == "About":
    st.subheader("About This App")
    st.markdown("""
This application is a **time series forecasting tool** built as a final project for
Data 5360 (Deep Forecasting) at Utah State University.

Upload any CSV with a date column and a numeric target, select your models, and the
app automatically optimizes all parameters and compares results.

---

### Models

**Econometric Models** (statsmodels):

- **Holt-Winters / Exponential Smoothing**: Decomposes the series into level, trend,
  and seasonal components. The app automatically tests all combinations of additive /
  multiplicative / no trend, additive / multiplicative / no seasonality, and damped vs
  undamped trend, then selects the configuration with the lowest AIC.
- **ARIMA / SARIMA**: AutoRegressive Integrated Moving Average. The app uses a stepwise
  algorithm (pmdarima) to automatically find the optimal (p,d,q) order and seasonal
  (P,D,Q,s) parameters, minimizing AIC.

**Machine Learning Models** (scikit-learn / XGBoost):

- **Random Forest Regressor**: An ensemble of decision trees. Hyperparameters
  (n_estimators, max_depth) are tuned via grid search with time series cross-validation.
- **XGBoost Regressor**: Gradient-boosted trees. Hyperparameters (n_estimators,
  max_depth, learning_rate) are tuned via grid search with time series cross-validation.

ML models use **manual feature engineering**: lag values and rolling means computed from
the target variable. To handle the extrapolation limitation of tree-based models (they
cannot predict beyond the range seen in training), the ML pipeline trains on
**first-differenced data** (period-over-period changes) rather than raw levels. Predicted
diffs are then reconstructed back to levels via cumulative sum. This is the same approach
used in the course's MLForecast pipeline (`Differences([1])` target transform) and in
HW5's manual differencing strategy. Future forecasts use a **recursive strategy** --
each predicted change is fed back as input for the next step.

---

### Data Handling

- **Missing values** are handled via user-selected method (interpolation, forward fill, or drop)
- **Frequency** is automatically inferred from the date index
- **Seasonal period** is inferred from frequency (e.g., 12 for monthly, 4 for quarterly)

### Train / Test Split

The data is split chronologically (80% train / 20% test). This preserves temporal
ordering and prevents data leakage.

---

### Metrics

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Squared Error. Penalizes large errors. Same unit as target. |
| **MAE** | Mean Absolute Error. Average absolute deviation. Same unit as target. |
| **MAPE** | Mean Absolute Percentage Error. Scale-independent percentage. |

Lower values indicate better performance for all three metrics.
    """)

    if df_raw is not None:
        st.subheader("Current Dataset Summary")
        st.markdown(f"""
- **Observations**: {len(series_full)}
- **Date range**: {series_full.index.min().date()} to {series_full.index.max().date()}
- **Inferred frequency**: {str(freq)}
- **Seasonal period**: {seasonal_period}
- **Target variable**: {target_col}
- **Mean**: {series_full.mean():.2f}
- **Std Dev**: {series_full.std():.2f}
- **Min**: {series_full.min():.2f}
- **Max**: {series_full.max():.2f}
        """)
