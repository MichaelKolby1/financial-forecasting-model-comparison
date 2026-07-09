# Financial Forecasting Model Comparison

An educational project comparing three models for daily stock prediction: **LSTM**, **XGBoost**, and **N-BEATS**.

The project includes a Jupyter notebook for the full modeling workflow and a Streamlit app for an interactive demo.

## Project Overview

This project compares different machine learning and deep learning approaches for financial forecasting. The goal is not to create a trading tool, but to evaluate how different model types perform on the same stock prediction task.

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
python -m streamlit run streamlit_stock_prediction.py
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

Clone the repository and install the required packages:

```bash
git clone https://github.com/MichaelKolby1/financial-forecasting-model-comparison.git
cd financial-forecasting-model-comparison
pip install -r requirements.txt
```

## Limitations

This project is for educational and portfolio purposes only. Stock prediction is highly uncertain, and results depend on the selected ticker, date range, model settings, train/test split, and market conditions. This project is not financial advice or a trading recommendation tool.
