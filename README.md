# Financial Forecasting Model Comparison

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://financial-forecasting-model-comparison.streamlit.app/)

An educational project comparing three models for daily stock prediction: **LSTM**, **XGBoost**, and **N-BEATS**.

The project includes a Jupyter notebook for the full modeling workflow and a Streamlit app for an interactive demo.

> **Note:** Streamlit may put the app to sleep after a period of inactivity. If prompted, click **Yes, get this app back up!** and allow a moment for the app to restart.

## App Preview

![Live App Overview](app-overview.png)

![Model Performance Metrics](sample_model_results.png)

![Model Comparison](sample_model_comparison.png)

## Project Overview

This project compares different machine learning and deep learning approaches for financial forecasting. The goal is not to create a trading tool, but to evaluate how different model types perform on the same stock prediction task.

## Learning Objectives

This project was designed as an educational experience to explore the end-to-end workflow of a machine learning project in finance. It provided practice with financial data collection, exploratory data analysis, feature engineering, model training, performance evaluation, and interactive app deployment.

The project also served as an opportunity to learn about machine learning and deep learning approaches such as LSTM networks, gradient-boosted trees, and N-BEATS, as well as the Python libraries commonly used to support these workflows.

## Data Collection

Financial data is collected using `yfinance`. The project retrieves historical daily stock data based on the selected ticker and date range, including price and volume information. Benchmark market data, such as SPY and QQQ returns, is also used as part of the feature set for selected models.

The Streamlit app also uses `yfinance` to display basic company profile information for the selected ticker.

## Notebook Structure

The Jupyter notebook walks through the full project workflow, including:

- Importing required libraries
- Loading financial data
- Performing exploratory data analysis across selected market sectors
- Setting up the train/test split
- Building and evaluating the LSTM, XGBoost, and N-BEATS
- Comparing model performance using shared metrics and visualizations

## Models

- **LSTM**: Multivariate sequence model using a 30-trading-day lookback window.
- **XGBoost**: Gradient-boosted tree model using engineered tabular features such as lagged returns, moving averages, volatility, RSI, MACD, and benchmark returns.
- **N-BEATS**: Univariate time-series model using the previous 30 closing prices.

## Metrics

Models are evaluated using:

- RMSE
- MAE
- MAPE
- Directional Accuracy

## Streamlit App

The Streamlit app allows users to select a ticker and date range, view company information, explore candlestick and volume charts, and compare model predictions and performance metrics.

Run locally with:

```bash
streamlit run streamlit_stock_prediction.py
```

## Repository Structure

```text
financial-forecasting-model-comparison/
├── README.md
├── requirements.txt
├── stock_prediction.ipynb
├── streamlit_stock_prediction.py
└── .gitignore
```

## Installation

This project was developed and tested using Python 3.13. The package versions
listed in `requirements.txt` are the versions used to support the deployed
Streamlit demo and are recommended for consistent code execution.

Clone the repository and install the dependencies:

```bash
git clone https://github.com/MichaelKolby1/financial-forecasting-model-comparison.git
cd financial-forecasting-model-comparison
python -m pip install -r requirements.txt
```

## Limitations

This project is for educational and portfolio purposes only. Stock prediction is highly uncertain, and results depend on the selected ticker, date range, model settings, train/test split, and market conditions. Data from `yfinance` may be delayed, revised, incomplete, or unavailable for some symbols. This project is not financial advice or a trading recommendation tool.
