"""
Streamlit demo for Financial Forecasting Model Comparison project.

Models included:
1. LSTM: multivariate sequence model using prior LOOKBACK trading days.
2. XGBoost: engineered tabular features predicting next-day return, converted to close price.
3. N-BEATS: univariate sequence model using historical closing prices only.

Run locally:
    python -m streamlit run streamlit_stock_prediction.py
"""

from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Optional ML imports. These allow the app to explain missing packages clearly.
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from nbeats_pytorch.model import NBeatsNet
    NBEATS_AVAILABLE = True
except Exception:
    NBEATS_AVAILABLE = False


# -----------------------------
# CONSTANTS
# -----------------------------
LOOKBACK = 30
TRAIN_FRACTION = 0.95
RANDOM_SEED = 42

POPULAR_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "ORCL", "CRM", "INTC", "IBM",
    "JPM", "BAC", "GS", "MS", "V", "MA", "AXP", "BLK",
    "JNJ", "PFE", "MRK", "UNH", "ABBV", "LLY",
    "XOM", "CVX", "COP", "SLB",
    "WMT", "COST", "TGT", "HD", "MCD", "SBUX", "NKE", "KO", "PEP",
    "CAT", "BA", "GE", "HON", "UPS", "TSLA", "F", "GM", "DAL",
    "DIS", "NFLX", "T", "VZ", "SPY", "QQQ",
]

MODEL_OPTIONS = ["LSTM", "XGBoost", "N-BEATS"]


# -----------------------------
# SUPPORTING FUNCTIONS
# -----------------------------
def normalize_ticker(ticker: str) -> str: 
    """
    Standardize ticker format entered by user.
    """
    return ticker.strip().upper()

def safe_pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    """
    Safely calculate percent changes for reproducibility purposes.
    """
    return series.pct_change(periods=periods).replace([np.inf, -np.inf], np.nan)

def compute_metrics(actual: np.ndarray, predicted: np.ndarray, current_close: np.ndarray) -> Dict[str, float]:
    """
    Calculate the following metrics to evaluate model performance: RMSE, MAE, MAPE, directional accuracy.
    """
    actual = np.asarray(actual, dtype=float).flatten()
    predicted = np.asarray(predicted, dtype=float).flatten()
    current_close = np.asarray(current_close, dtype=float).flatten()

    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae = float(mean_absolute_error(actual, predicted))
    mape = float(np.mean(np.abs((actual - predicted) / actual)) * 100)

    actual_direction = np.sign(actual - current_close)
    predicted_direction = np.sign(predicted - current_close)
    directional_accuracy = float(np.mean(actual_direction == predicted_direction) * 100)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE%": mape,
        "Directional Accuracy%": directional_accuracy,
    }

def make_prediction_frame(model_name: str, dates: np.ndarray, actual: np.ndarray, predicted: np.ndarray, current_close: np.ndarray,) -> pd.DataFrame:
    """
    Store results for each model to cleanly compare in demo.
    """
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(dates),
            "Model": model_name,
            "Actual": np.asarray(actual, dtype=float).flatten(),
            "Predicted": np.asarray(predicted, dtype=float).flatten(),
            "Current_Close": np.asarray(current_close, dtype=float).flatten(),
        }
    )


# -----------------------------
# DATA DOWNLOAD & PREPARATION
# -----------------------------
@st.cache_data(show_spinner=False) # allows for streamlit to reuse previously downloaded data if user tries downloading results for the same ticker again
def download_market_data(ticker: str, start_year: int, end_year: int) -> pd.DataFrame: 
    """Download stock performance data for selected ticker and SPY/QQQ market benchmarks.

    end_year is treated as inclusive. For example, start_year = 2021 and end_year = 2026
    downloads from 2021-01-01 through 2027-01-01, or through the latest available date.
    """
    required_tickers = sorted(set([ticker, "SPY", "QQQ"]))

    raw = yf.download(
        required_tickers,
        start=f"{start_year}-01-01",
        end=f"{end_year + 1}-01-01",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        data = raw.stack(level=0).reset_index()
        if "Date" not in data.columns:
            if "level_0" in data.columns:
                data = data.rename(columns={"level_0": "Date"})
            elif "index" in data.columns:
                data = data.rename(columns={"index": "Date"})
        if "level_1" in data.columns:
            data = data.rename(columns={"level_1": "Ticker"})
        if "Ticker" not in data.columns:
            # Some yfinance versions name the stacked column differently.
            non_price_cols = [c for c in data.columns if c not in ["Date", "Open", "High", "Low", "Close", "Volume"]]
            if non_price_cols:
                data = data.rename(columns={non_price_cols[0]: "Ticker"})
    else:
        data = raw.reset_index()
        data["Ticker"] = ticker

    data.columns.name = None
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.dropna(subset=["Close"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Keep only the columns used by the demo.
    expected_cols = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
    available_cols = [col for col in expected_cols if col in data.columns]
    return data[available_cols].copy()

@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def get_company_profile(ticker: str) -> Dict[str, str]:
    """
    Retrieve basic company information from yfinance for the user's selected ticker.
    """
    try:
        info = yf.Ticker(ticker).get_info()
    except Exception:
        info = {}

    company_name = info.get("longName") or info.get("shortName") or ticker
    summary = info.get("longBusinessSummary") or "Company summary is unavailable for this ticker."
    sector = info.get("sector", "")
    industry = info.get("industry", "")
    website = info.get("website", "")

    return {
        "ticker": ticker,
        "company_name": company_name,
        "summary": summary,
        "sector": sector,
        "industry": industry,
        "website": website,
        "yahoo_quote_url": f"https://finance.yahoo.com/quote/{ticker}",
        "yahoo_news_url": f"https://finance.yahoo.com/quote/{ticker}/news",
    }

def get_stock_frame(data: pd.DataFrame, ticker: str) -> pd.DataFrame: 
    """
    Limits the downloaded data to just those rows corresponding to the user's ticker.
    """
    stock_df = data[data["Ticker"] == ticker].copy()
    stock_df = stock_df.sort_values("Date").reset_index(drop=True)
    return stock_df

def validate_inputs(stock_df: pd.DataFrame, start_year: int, end_year: int) -> None:
    """
    Addresses and raises possible errors for user inputs.
    """
    year_span = end_year - start_year

    if end_year <= start_year:
        raise ValueError("End year must be greater than start year.")
    if year_span < 2:
        raise ValueError("Please select at least 3 years of data for reliable model training.")
    if year_span > 10:
        raise ValueError("Please select no more than 10 years of data to keep training time reasonable.")
    if stock_df.empty:
        raise ValueError("No data was returned for this ticker. Please check the ticker symbol.")
    if len(stock_df) < LOOKBACK + 100:
        raise ValueError(
            f"Only {len(stock_df)} rows were returned. Select a longer date range.")


# -----------------------------
# LSTM MODEL
# -----------------------------
def prepare_lstm_data(stock_df: pd.DataFrame, full_data: pd.DataFrame,training_set: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, List[str]]: 
    """
    Replicate feature engineering and window preparation from notebook.
    """
    lstm_df = stock_df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    lstm_df = lstm_df.sort_values("Date").reset_index(drop=True)
    lstm_df["Original_Index"] = np.arange(len(lstm_df))

    lstm_df["Return_1d"] = safe_pct_change(lstm_df["Close"])
    lstm_df["High_Low_Range"] = (lstm_df["High"] - lstm_df["Low"]) / lstm_df["Close"]
    lstm_df["Volume_Change"] = safe_pct_change(lstm_df["Volume"])
    lstm_df["SMA_10"] = lstm_df["Close"].rolling(10).mean()
    lstm_df["Close_to_SMA_10"] = (lstm_df["Close"] / lstm_df["SMA_10"]) - 1

    spy_df = full_data[full_data["Ticker"] == "SPY"][["Date", "Close"]].copy()
    spy_df = spy_df.sort_values("Date").reset_index(drop=True)
    spy_df["SPY_Return_1d"] = safe_pct_change(spy_df["Close"])
    spy_df = spy_df[["Date", "SPY_Return_1d"]]

    lstm_df = lstm_df.merge(spy_df, on="Date", how="left")

    sequence_feature_cols = ["Close", "Return_1d", "High_Low_Range", "Volume_Change", "Close_to_SMA_10", "SPY_Return_1d"]

    lstm_required_cols = sequence_feature_cols + ["Date", "Close", "Original_Index"]
    lstm_df = lstm_df.replace([np.inf, -np.inf], np.nan).dropna(subset=lstm_required_cols).copy()
    lstm_df = lstm_df.reset_index(drop=True)

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler.fit(lstm_df.loc[lstm_df["Original_Index"] < training_set, sequence_feature_cols])
    lstm_features_scaled = feature_scaler.transform(lstm_df[sequence_feature_cols])

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler.fit(stock_df[["Close"]].values[:training_set])
    lstm_target_scaled = target_scaler.transform(lstm_df[["Close"]]).flatten()

    x_train, y_train = [], []
    x_test, y_test, test_dates, target_indices = [], [], [], []

    for pos in range(LOOKBACK, len(lstm_df)):
        target_original_idx = int(lstm_df.loc[pos, "Original_Index"])
        window_original_indices = lstm_df.loc[pos - LOOKBACK : pos - 1, "Original_Index"].values

        if len(window_original_indices) != LOOKBACK:
            continue
        if not np.all(np.diff(window_original_indices) == 1):
            continue
        if window_original_indices[-1] != target_original_idx - 1:
            continue

        x_window = lstm_features_scaled[pos - LOOKBACK : pos, :]
        y_value = lstm_target_scaled[pos]

        if target_original_idx < training_set:
            x_train.append(x_window)
            y_train.append(y_value)
        else:
            x_test.append(x_window)
            y_test.append(lstm_df.loc[pos, "Close"])
            test_dates.append(lstm_df.loc[pos, "Date"])
            target_indices.append(target_original_idx)

    return (
        np.array(x_train, dtype=np.float32),
        np.array(y_train, dtype=np.float32),
        np.array(x_test, dtype=np.float32),
        np.array(y_test, dtype=float),
        np.array(test_dates),
        np.array(target_indices),
        target_scaler,
        sequence_feature_cols,
    )

def run_lstm(stock_df: pd.DataFrame, full_data: pd.DataFrame, training_set: int, epochs: int) -> Tuple[pd.DataFrame, Dict[str, float], List[str]]:
    """
    Run and evaluate the LSTM model for the user's selected ticker.
    """
    if not TF_AVAILABLE: # Ensure TensorFlow is installed
        raise ImportError("TensorFlow is not installed. Install tensorflow to run the LSTM model.")

    tf.random.set_seed(RANDOM_SEED)

    x_train, y_train, x_test, y_test, test_dates, target_indices, target_scaler, feature_cols = prepare_lstm_data(
        stock_df, full_data, training_set
    )

    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError("Not enough valid rows were available to train and test the LSTM model.")

    model = keras.models.Sequential(
        [
            keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
            keras.layers.LSTM(units=64),
            keras.layers.Dropout(0.20),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1),
        ]
    )

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mean_squared_error")
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)

    model.fit(x_train, y_train, epochs=epochs, batch_size=32, validation_split=0.10, shuffle=False, callbacks=[early_stop], verbose=0)

    preds_scaled = model.predict(x_test, verbose=0)
    predicted = target_scaler.inverse_transform(preds_scaled).flatten()
    actual = y_test.flatten()
    current_close = stock_df["Close"].values[target_indices - 1]

    pred_df = make_prediction_frame("LSTM", test_dates, actual, predicted, current_close)
    metrics = compute_metrics(actual, predicted, current_close)
    return pred_df, metrics, feature_cols


# -----------------------------
# XGBOOST MODEL
# -----------------------------
def prepare_xgboost_data(stock_df: pd.DataFrame, full_data: pd.DataFrame, training_set: int) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Feature engineering to predict next-day return for ticker.
    """
    xgb_df = stock_df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    xgb_df = xgb_df.sort_values("Date").reset_index(drop=True)

    xgb_df["Return_1d"] = safe_pct_change(xgb_df["Close"])
    xgb_df["Open_Close_Return"] = (xgb_df["Close"] - xgb_df["Open"]) / xgb_df["Open"]
    xgb_df["High_Low_Range"] = (xgb_df["High"] - xgb_df["Low"]) / xgb_df["Close"]

    for lag in range(0, LOOKBACK):
        xgb_df[f"Return_lag_{lag}"] = xgb_df["Return_1d"].shift(lag)

    rolling_windows = [5, 10, 20, LOOKBACK]
    for window in rolling_windows:
        xgb_df[f"SMA_{window}"] = xgb_df["Close"].rolling(window).mean()
        xgb_df[f"Close_to_SMA_{window}"] = (xgb_df["Close"] / xgb_df[f"SMA_{window}"]) - 1
        xgb_df[f"Volatility_{window}"] = xgb_df["Return_1d"].rolling(window).std()
        xgb_df[f"Momentum_{window}"] = xgb_df["Close"].pct_change(window)

    xgb_df["Volume_Change"] = safe_pct_change(xgb_df["Volume"])
    for window in [5, 20]:
        xgb_df[f"Volume_SMA_{window}"] = xgb_df["Volume"].rolling(window).mean()
        xgb_df[f"Volume_to_SMA_{window}"] = xgb_df["Volume"] / xgb_df[f"Volume_SMA_{window}"]

    delta = xgb_df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    xgb_df["RSI_14"] = 100 - (100 / (1 + rs))

    xgb_df["EMA_12"] = xgb_df["Close"].ewm(span=12, adjust=False).mean()
    xgb_df["EMA_26"] = xgb_df["Close"].ewm(span=26, adjust=False).mean()
    xgb_df["MACD"] = xgb_df["EMA_12"] - xgb_df["EMA_26"]
    xgb_df["MACD_Signal"] = xgb_df["MACD"].ewm(span=9, adjust=False).mean()
    xgb_df["MACD_Hist"] = xgb_df["MACD"] - xgb_df["MACD_Signal"]

    xgb_df["DayOfWeek"] = pd.to_datetime(xgb_df["Date"]).dt.dayofweek
    xgb_df["Month"] = pd.to_datetime(xgb_df["Date"]).dt.month

    market_df = full_data[full_data["Ticker"].isin(["SPY", "QQQ"])][["Date", "Ticker", "Close"]].copy()
    market_df = market_df.pivot(index="Date", columns="Ticker", values="Close").reset_index()
    market_df["SPY_Return_1d"] = safe_pct_change(market_df["SPY"])
    market_df["QQQ_Return_1d"] = safe_pct_change(market_df["QQQ"])
    market_df["SPY_Return_5d"] = safe_pct_change(market_df["SPY"], periods=5)
    market_df["QQQ_Return_5d"] = safe_pct_change(market_df["QQQ"], periods=5)

    market_features = ["Date", "SPY_Return_1d", "QQQ_Return_1d", "SPY_Return_5d", "QQQ_Return_5d"]
    xgb_df = xgb_df.merge(market_df[market_features], on="Date", how="left")

    xgb_df["Target_Close"] = xgb_df["Close"].shift(-1)
    xgb_df["Target_Return_1d"] = (xgb_df["Target_Close"] / xgb_df["Close"]) - 1
    xgb_df["Target_Date"] = xgb_df["Date"].shift(-1)
    xgb_df["Target_Index"] = xgb_df.index + 1

    feature_cols = [
        "Close",
        "Open_Close_Return",
        "High_Low_Range",
        "Volume_Change",
        "RSI_14",
        "MACD",
        "MACD_Signal",
        "MACD_Hist",
        "DayOfWeek",
        "Month",
        "SPY_Return_1d",
        "QQQ_Return_1d",
        "SPY_Return_5d",
        "QQQ_Return_5d",
    ]
    feature_cols += [f"Return_lag_{lag}" for lag in range(0, LOOKBACK)]

    for window in rolling_windows:
        feature_cols += [f"Close_to_SMA_{window}", f"Volatility_{window}", f"Momentum_{window}"]
    feature_cols += ["Volume_to_SMA_5", "Volume_to_SMA_20"]

    xgb_df = xgb_df.replace([np.inf, -np.inf], np.nan).dropna().copy()
    train_df = xgb_df[xgb_df["Target_Index"] < training_set].copy()
    test_df = xgb_df[xgb_df["Target_Index"] >= training_set].copy()
    return train_df, test_df, feature_cols

def run_xgboost(stock_df: pd.DataFrame, full_data: pd.DataFrame, training_set: int) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    """
    Run and evaluate the XGBoost model for the user's selected ticker.
    """
    if not XGB_AVAILABLE: # Ensure XGBoost is installed
        raise ImportError("XGBoost is not installed. Install xgboost to run this model.")

    train_df, test_df, feature_cols = prepare_xgboost_data(stock_df, full_data, training_set)
    if train_df.empty or test_df.empty:
        raise ValueError("Not enough valid rows were available to train and test XGBoost.")

    xgb_model = xgb.XGBRegressor(
        n_estimators=800,
        max_depth=3,
        learning_rate=0.02,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=2.0,
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=1,
        tree_method="hist",
        verbosity=0,
    )

    xgb_model.fit(train_df[feature_cols].values, train_df["Target_Return_1d"].values, verbose=False)
    pred_returns = xgb_model.predict(test_df[feature_cols].values)

    current_close = test_df["Close"].values
    predicted = current_close * (1 + pred_returns)
    actual = test_df["Target_Close"].values
    dates = test_df["Target_Date"].values

    pred_df = make_prediction_frame("XGBoost", dates, actual, predicted, current_close)
    metrics = compute_metrics(actual, predicted, current_close)

    importance_df = pd.DataFrame(
        {"Feature": feature_cols, "Importance": xgb_model.feature_importances_}
    ).sort_values("Importance", ascending=False)

    return pred_df, metrics, importance_df


# -----------------------------
# N-BEATS MODEL
# -----------------------------
def prepare_nbeats_data(series: np.ndarray, first_target_idx: int, last_target_exclusive: int, lookback: int = LOOKBACK):
    """Build sequence windows where each target is the next day after the window.

    Each X window contains the prior lookback values.
    Each y target is the next value immediately after that window.
    """
    x_values, y_values = [], []
    for i in range(first_target_idx, last_target_exclusive):
        x_values.append(series[i - lookback : i])
        y_values.append(series[i])
    return np.array(x_values, dtype=np.float32), np.array(y_values, dtype=np.float32).reshape(-1, 1)

def run_nbeats(stock_df: pd.DataFrame, training_set: int, epochs: int) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Run and evaluate N-BEATS model for the user's selected ticker.
    """
    if not NBEATS_AVAILABLE: # Ensure nbeats-pytorch is installed
        raise ImportError("nbeats-pytorch is not installed. Install nbeats-pytorch and torch to run N-BEATS.")

    torch.manual_seed(RANDOM_SEED)
    torch.set_num_threads(1)

    close_vals = stock_df["Close"].values.astype(float)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(close_vals[:training_set].reshape(-1, 1))
    scaled = scaler.transform(close_vals.reshape(-1, 1)).flatten()

    x_train, y_train = prepare_nbeats_data(scaled, LOOKBACK, training_set)
    x_test, y_test = prepare_nbeats_data(scaled, training_set, len(scaled))

    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError("Not enough valid rows were available to train and test N-BEATS.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    model = NBeatsNet(
        device=device,
        stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK),
        nb_blocks_per_stack=2,
        forecast_length=1,
        backcast_length=LOOKBACK,
        thetas_dim=(8, 8),
        share_weights_in_stack=False,
        hidden_layer_units=64,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            backcast, forecast = model(batch_x)
            loss = loss_fn(forecast, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
        backcast, preds_scaled_tensor = model(test_tensor)

    predicted = scaler.inverse_transform(preds_scaled_tensor.cpu().numpy()).flatten()
    actual = close_vals[training_set:]
    current_close = close_vals[training_set - 1 : len(close_vals) - 1]
    dates = stock_df["Date"].values[training_set:]

    pred_df = make_prediction_frame("N-BEATS", dates, actual, predicted, current_close)
    metrics = compute_metrics(actual, predicted, current_close)
    return pred_df, metrics


# -----------------------------
# VISUALIZATION FUNCTIONS
# -----------------------------
def plot_candlestick_with_volume(stock_df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Plot candlestick price chart with trading volume underneath.
    """
    df = stock_df.sort_values("Date").copy()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.72, 0.28],specs=[[{"type": "candlestick"}], [{"type": "bar"}]])

    fig.add_trace(go.Candlestick(x=df["Date"],open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="OHLC", increasing_line_color="#2ca25f", decreasing_line_color="#de2d26"),row=1,col=1)

    volume_colors = np.where(
        df["Close"] >= df["Open"],
        "rgba(44, 162, 95, 0.55)",
        "rgba(222, 45, 38, 0.55)",
    )

    fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="Volume", marker_color=volume_colors, opacity=0.70), row=2, col=1)

    fig.update_layout(title=f"{ticker}: Price and Volume", template="plotly_white", hovermode="x unified", height=750, legend_title="Series", margin=dict(l=40, r=40, t=70, b=40))

    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    fig.update_xaxes(rangeslider_visible=False)

    return fig

def plot_test_period_predictions(predictions: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Plot model predictions against actual closing prices during the test period only.
    """
    fig = go.Figure()

    actual_df = (predictions[["Date", "Actual"]].drop_duplicates().sort_values("Date"))

    fig.add_trace(go.Scatter(x=actual_df["Date"], y=actual_df["Actual"], mode="lines", name="Actual close", line=dict(width=3)))

    for model_name in predictions["Model"].unique():
        model_df = predictions[predictions["Model"] == model_name].sort_values("Date")

        fig.add_trace(go.Scatter(x=model_df["Date"], y=model_df["Predicted"], mode="lines", name=f"{model_name} predicted", line=dict(dash="dash", width=2)))

    fig.update_layout(title=f"{ticker}: Model Predictions During Test Period", xaxis_title="Date", yaxis_title="Close Price (USD)", hovermode="x unified", template="plotly_white", legend_title="Series", height=600, margin=dict(l=40, r=40, t=70, b=40))

    return fig

def plot_metric_bars(metrics_df: pd.DataFrame, metric: str) -> go.Figure:
    """
    Plot bar chart comparing one performance metric across models.
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(x=metrics_df.index, y=metrics_df[metric], text=metrics_df[metric].round(3), textposition="auto"))

    fig.update_layout(title=metric, xaxis_title="Model", yaxis_title=metric, template="plotly_white", height=400, margin=dict(l=40, r=40, t=60, b=40))

    return fig


# -----------------------------
# STREAMLIT USER INTERFACE
# -----------------------------
def render_project_explanation():
    """
    Generate project description when the UI is created.
    """
    st.markdown(
        """
        This app lets users select a stock ticker and date range, then runs all three models using the same
        30-trading-day lookback window and chronological train/test split.

        The goal is to compare how different modeling approaches perform on the same prediction task, not to
        provide investment advice or trading recommendations.

        <u>Model Design:</u>
        - **LSTM:** uses a 30-trading-day sequence with multiple daily features to predict the next closing price.
        - **XGBoost:** uses engineered technical, volume, calendar, and market features to predict next-day return,
          then converts that return into a predicted closing price.
        - **N-BEATS:** uses only the prior 30 closing prices as a univariate time-series model.
        """, 
        unsafe_allow_html=True
    )

def render_company_profile(profile):
    st.subheader(f"{profile['company_name']} ({profile['ticker']})")

    details = []
    if profile["sector"]:
        details.append(f"**Sector:** {profile['sector']}")
    if profile["industry"]:
        details.append(f"**Industry:** {profile['industry']}")
    
    if details:
        st.markdown(" | ".join(details))

    with st.expander(f"{profile['company_name']} Summary"):
        st.write(profile["summary"])

    link_cols = st.columns(3)

    with link_cols[0]:
        st.link_button(f"{profile['company_name']} Quote Page", profile["yahoo_quote_url"])
    
    with link_cols[1]:
        st.link_button(f"{profile['company_name']} Stock News", profile["yahoo_news_url"])
    
    if profile["website"]:
        with link_cols[2]:
            st.link_button("Company Website", profile["website"])

def main():
    """
    Design the UI to support user selection of a ticker and time frame for which the 3 models are run and to evaluate and compare model performances.
    """
    st.set_page_config(page_title="Financial Forecasting Model Comparison", layout="wide")

    # --------------------------------------
    # Home Page Display
    # --------------------------------------
    st.title("Financial Forecasting Model Comparison")
    st.caption("Interactive demo comparing machine learning and deep learning models (LSTM, XGBoost, and N-BEATS) for daily stock prediction.")

    st.markdown("""<style>
        details summary p {font-size: 1.2rem !important;}
        </style>""",
        unsafe_allow_html=True)

    with st.expander("**Project Overview**", expanded=True):
        render_project_explanation()

    current_year = datetime.now().year

    st.sidebar.header("Controls")
    ticker_choice = st.sidebar.selectbox("Common Tickers", POPULAR_TICKERS, index=POPULAR_TICKERS.index("NVDA"))
    custom_ticker = st.sidebar.text_input("Or enter a custom ticker", value="")
    ticker = normalize_ticker(custom_ticker) if custom_ticker.strip() else ticker_choice

    start_year = st.sidebar.number_input("Start year", min_value=current_year - 26, max_value=current_year - 3, value=current_year - 6, step=1, help="Note: Use at least 3 years and no more than 10 years of data after 2000.")
    end_year = st.sidebar.number_input("End year", max_value=current_year, value=current_year-1, step=1, help="Note: End year is treated as inclusive. If it is the current year, data is downloaded through the latest available trading day.")

    st.sidebar.markdown("---")
    st.sidebar.write(f"**Look-back Window:** {LOOKBACK} trading days")
    st.sidebar.write(f"**Train/test Split:** {int(TRAIN_FRACTION * 100)}% / {int((1 - TRAIN_FRACTION) * 100)}%")

    run_button = st.sidebar.button("Run model comparison", type="primary")

    if not run_button:
        st.info("Choose a ticker and date range in the sidebar, then click **Run model comparison**.")
        return

    st.divider()

    # --------------------------------------
    # Data Download & Ticker Information
    # --------------------------------------
    try:
        with st.spinner("Downloading market data..."):
            full_data = download_market_data(ticker, int(start_year), int(end_year))
            stock_df = get_stock_frame(full_data, ticker)
            validate_inputs(stock_df, int(start_year), int(end_year))
            company_profile = get_company_profile(ticker)
            render_company_profile(company_profile)
    except Exception as exc:
        st.error(str(exc))
        return
     
    st.markdown("---")
    training_set = int(np.ceil(len(stock_df) * TRAIN_FRACTION))

    st.header("Training Overview", divider = "gray")  
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ticker", ticker)
    c2.metric("Rows", f"{len(stock_df):,}")
    c3.metric("Training Rows", f"{training_set:,}")
    c4.metric("Test Rows", f"{len(stock_df) - training_set:,}")

    st.plotly_chart(plot_candlestick_with_volume(stock_df, ticker), use_container_width=True)

    with st.expander("Downloaded data preview"):
        st.dataframe(stock_df.tail(10), hide_index = True, use_container_width=True)

    prediction_frames = []
    metrics_records = {}
    xgb_importance_df = pd.DataFrame()
    lstm_feature_cols = []

    st.divider()
    st.header("Model Status")
    for model_name in MODEL_OPTIONS:
        try:
            with st.spinner(f"Training and evaluating {model_name}..."):
                if model_name == "LSTM":
                    pred_df, metrics, lstm_feature_cols = run_lstm(stock_df, full_data, training_set, 30) # epochs set to 30 by default 
                elif model_name == "XGBoost":
                    pred_df, metrics, xgb_importance_df = run_xgboost(stock_df, full_data, training_set)
                elif model_name == "N-BEATS":
                    pred_df, metrics = run_nbeats(stock_df, training_set, 50) # epochs set to 50 by default 
                else:
                    continue
            prediction_frames.append(pred_df)
            metrics_records[model_name] = metrics
            st.success(f"{model_name} complete")
        except Exception as exc:
            st.warning(f"{model_name} could not run: {exc}")

    if not prediction_frames:
        st.error("No model results were produced. Check package installation and input settings.")
        return

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics_df = pd.DataFrame(metrics_records).T

    # --------------------------------------
    # Model Results & Analysis
    # --------------------------------------
    results_tab, details_tab = st.tabs(["Primary Results", "Model Details & Prediction Table"])
    with results_tab:    
        st.header("Model Performance & Metrics")

        st.plotly_chart(plot_test_period_predictions(predictions, ticker), use_container_width=True)
        st.caption("This chart displays model predictions for the test period so these prices can be clearly compared against the actual closing prices.")

        st.write("")
        st.write("**Model Metric Comparison**")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.plotly_chart(plot_metric_bars(metrics_df, "RMSE"), use_container_width=True)
        with m2:
            st.plotly_chart(plot_metric_bars(metrics_df, "MAE"), use_container_width=True)
        with m3:
            st.plotly_chart(plot_metric_bars(metrics_df, "MAPE%"), use_container_width=True)
        with m4:
            st.plotly_chart(plot_metric_bars(metrics_df, "Directional Accuracy%"), use_container_width=True)
            
        st.dataframe(metrics_df.round(4))

        error_metrics = ["RMSE", "MAE", "MAPE%"]

        rank_df = metrics_df[error_metrics].rank(ascending=True)
        rank_df["Average Error Rank"] = rank_df.mean(axis=1)

        best_error_model = rank_df["Average Error Rank"].idxmin()
        st.info(f"""Result summary: Based on the error-based metrics shown above, {best_error_model} had the strongest overall performance for this selected ticker and time period.
        Directional accuracy should be interpreted separately because it measures the accuracy of price direction rather than the size of the prediction error.""")

        with st.expander("**Metrics Explained**"):
            st.markdown(
                """
                <u>RMSE</u>: Measures the average size of prediction errors, with larger errors penalized more heavily. 
                Lower RMSE indicates that the model made fewer large mistakes.

                <u>MAE</u>: Measures the average absolute difference between predicted and actual closing prices. 
                Lower MAE means the model's predictions were closer to the actual prices on average.

                <u>MAPE</u>: Measures prediction error as a percentage of the actual closing price. This makes it easier to compare
                 model performance across stocks with different price levels because the error is expressed relative to the stock price rather than in dollars.

                <u>Directional Accuracy</u>: Measures how often the model correctly predicted whether the next closing price would move up or down. 
                Higher directional accuracy is better, but it should be interpreted alongside RMSE, MAE, and MAPE because a model can predict direction correctly while still being far from the actual price.
                """,
                unsafe_allow_html=True
            )
        
        st.write("")
        st.caption("Educational project only. yfinance data may be delayed, revised, or unavailable for some symbols. Results depend on the selected period and are not financial advice.")

    with details_tab:
        st.header("Model Inputs & Prediction Approach")

        note_cols = st.columns(3)
        with note_cols[0]:
            st.subheader("LSTM")
            st.write(
                """The LSTM model uses a 30-trading-day lookback window to predict the next closing price. 
                For each day in the sequence, the model receives multiple daily features, including the 
                stock’s closing price, daily return, intraday price range, volume change, moving-average 
                relationship, and broader market movement through SPY returns. Because the input is 
                structured as a sequence, the LSTM is designed to learn patterns across time rather than 
                treating each trading day independently. The model outputs a direct prediction of the next 
                closing price."""
            )
            if lstm_feature_cols:
                st.write("**Sequence features:**")
                st.write(", ".join(lstm_feature_cols))

        with note_cols[1]:
            st.subheader("XGBoost")
            st.write(
                """The XGBoost model uses engineered tabular features to predict the stock’s next-day return. 
                Its inputs include lagged returns, moving averages, volatility measures, momentum indicators, 
                RSI, MACD, volume-based features, calendar variables, and market benchmark returns from SPY 
                and QQQ. Unlike the LSTM and N-BEATS models, XGBoost does not process the data as a sequence.
                Instead, each row represents one trading day with a set of explanatory features. After predicting 
                the next-day return, the model converts that return into a predicted closing price."""
            )
            if not xgb_importance_df.empty:
                st.write("**Top feature importances:**")
                st.dataframe(xgb_importance_df.head(10), use_container_width=True, hide_index=True)

        with note_cols[2]:
            st.subheader("N-BEATS")
            st.write(
                """The N-BEATS model is used as a univariate time-series model, meaning it relies only on the stock’s 
                historical closing prices. It takes the previous 30 closing prices as its input window and predicts 
                the next closing price. Unlike the LSTM and XGBoost models, N-BEATS does not use additional features 
                such as volume, technical indicators, calendar variables, or market benchmark returns. This makes it 
                useful as a comparison model for evaluating how much predictive signal can be captured from the 
                closing-price history alone."""
            )

        with st.expander("Prediction Table"):
            display_predictions = predictions.copy()
            display_predictions["Error"] = display_predictions["Predicted"] - display_predictions["Actual"]
            display_predictions["Absolute_Error"] = display_predictions["Error"].abs()
            st.dataframe(display_predictions.sort_values(["Date", "Model"]).round(4), hide_index = True, use_container_width=True)

        st.caption("Educational project only. yfinance data may be delayed, revised, or unavailable for some symbols. Results depend on the selected period and are not financial advice.")

if __name__ == "__main__":
    main()
