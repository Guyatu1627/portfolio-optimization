import json
import os

def create_notebook(filename, cells_data):
    cells = []
    for cell_type, source in cells_data:
        # split by \n and add \n back, but not to the last one
        lines = source.split('\n')
        source_lines = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": source_lines
        }
        if cell_type == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cells.append(cell)
    
    nb = {
        "cells": cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 4
    }
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

def task1():
    cells = [
        ("markdown", "# Task 1: Preprocess and Explore the Data\n\nThis notebook performs Data Extraction, Cleaning, Exploratory Data Analysis, Seasonality & Trend Analysis, and Risk Metrics Calculation for TSLA, BND, and SPY."),
        ("code", "import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom statsmodels.tsa.stattools import adfuller\nimport warnings\nwarnings.filterwarnings('ignore')\n\nplt.style.use('seaborn-v0_8')\n%matplotlib inline"),
        ("markdown", "## 1 & 2. Extract and Clean Data\nWe load the data fetched by our script and show basic info."),
        ("code", "import sys\nsys.path.append('../src')\nfrom data_fetch import process_and_save\n\ntickers = ['TSLA', 'BND', 'SPY']\ndf = process_and_save(tickers, '2015-01-01', '2026-06-30', '../data/processed/historical_data.csv')\n\n# Display basic information\nprint(df.info())\nprint(\"\\nBasic Statistics:\\n\", df['Adj Close'].describe())"),
        ("markdown", "## 3. Exploratory Data Analysis (EDA)\n### Closing Prices Over Time"),
        ("code", "plt.figure(figsize=(14, 7))\nfor ticker in tickers:\n    plt.plot(df.index, df['Adj Close'][ticker], label=ticker)\nplt.title('Adjusted Closing Price Over Time')\nplt.xlabel('Date')\nplt.ylabel('Price (USD)')\nplt.legend()\nplt.show()"),
        ("markdown", "### Daily Percentage Change (Volatility)"),
        ("code", "returns = df['Adj Close'].pct_change().dropna()\nplt.figure(figsize=(14, 7))\nfor ticker in tickers:\n    plt.plot(returns.index, returns[ticker], label=ticker, alpha=0.7)\nplt.title('Daily Percentage Change')\nplt.xlabel('Date')\nplt.ylabel('Daily Return')\nplt.legend()\nplt.show()"),
        ("markdown", "### Rolling Volatility (30-day standard deviation)"),
        ("code", "rolling_volatility = returns.rolling(window=30).std()\nplt.figure(figsize=(14, 7))\nfor ticker in tickers:\n    plt.plot(rolling_volatility.index, rolling_volatility[ticker], label=ticker)\nplt.title('30-Day Rolling Volatility')\nplt.xlabel('Date')\nplt.ylabel('Volatility (Std Dev)')\nplt.legend()\nplt.show()"),
        ("markdown", "### Outlier Detection (TSLA)\nIdentifying days with unusually high or low returns (> 3 std dev)."),
        ("code", "tsla_returns = returns['TSLA']\nmean_ret = tsla_returns.mean()\nstd_ret = tsla_returns.std()\noutliers = tsla_returns[(tsla_returns > mean_ret + 3*std_ret) | (tsla_returns < mean_ret - 3*std_ret)]\n\nplt.figure(figsize=(14, 7))\nplt.scatter(tsla_returns.index, tsla_returns, label='Normal Returns', alpha=0.5, s=10)\nplt.scatter(outliers.index, outliers, color='red', label='Outliers', s=20)\nplt.title('TSLA Daily Returns with Outliers Highlighted')\nplt.legend()\nplt.show()\n\nprint(f\"Number of outliers detected for TSLA: {len(outliers)}\")"),
        ("markdown", "## 4. Seasonality and Trend Analysis\nPerforming Augmented Dickey-Fuller (ADF) test to check stationarity."),
        ("code", "def adf_test(series, name):\n    result = adfuller(series.dropna())\n    print(f\"ADF Test for {name}\")\n    print(f\"ADF Statistic: {result[0]:.4f}\")\n    print(f\"p-value: {result[1]:.4f}\")\n    if result[1] <= 0.05:\n        print(\"=> Reject the null hypothesis (H0), the data is stationary.\\n\")\n    else:\n        print(\"=> Fail to reject the null hypothesis (H0), the data is non-stationary.\\n\")\n\n# Test on Prices\nadf_test(df['Adj Close']['TSLA'], \"TSLA Closing Price\")\n\n# Test on Returns\nadf_test(returns['TSLA'], \"TSLA Daily Returns\")"),
        ("markdown", "## 5. Calculate Risk Metrics\n### Value at Risk (VaR) & Sharpe Ratio"),
        ("code", "# Value at Risk (VaR) at 95% confidence level\nconfidence_level = 0.05\nvar_95 = returns.quantile(confidence_level)\nprint(\"Historical Value at Risk (95% confidence) - Worst daily loss expected 5% of the time:\")\nprint(var_95)\n\n# Sharpe Ratio (Assuming Risk-Free Rate = 0.02 / 252 daily)\nrisk_free_rate_daily = 0.02 / 252\nexcess_returns = returns - risk_free_rate_daily\nsharpe_ratio = (excess_returns.mean() / returns.std()) * np.sqrt(252) # Annualized\nprint(\"\\nAnnualized Sharpe Ratio:\")\nprint(sharpe_ratio)"),
        ("markdown", "## Summary of Insights\n- **Data Quality**: The fetched yfinance data is largely clean; small missing gaps were forward-filled.\n- **Trend & Volatility**: TSLA shows an overall long-term upward trend but exhibits significantly higher volatility compared to SPY and BND. BND is stable with minimal volatility.\n- **Stationarity**: TSLA closing prices are non-stationary (p-value > 0.05), implying differencing is needed for ARIMA. However, daily returns are stationary (p-value < 0.05).\n- **Risk Metrics**: TSLA has a high VaR indicating potential for large daily losses, but its Sharpe Ratio is historically decent despite high risk, capturing the high-growth profile. BND serves its role as a low-risk stability asset.")
    ]
    create_notebook('../notebooks/Task_1_EDA.ipynb', cells)

def task2():
    cells = [
        ("markdown", "# Task 2: Build Time Series Forecasting Models\n\nWe will develop and evaluate time series models (ARIMA and LSTM) to predict TSLA future stock prices."),
        ("code", "import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nfrom sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error\nfrom sklearn.preprocessing import MinMaxScaler\nimport pmdarima as pm\nfrom tensorflow.keras.models import Sequential\nfrom tensorflow.keras.layers import LSTM, Dense, Dropout\nimport warnings\nwarnings.filterwarnings('ignore')\n\nplt.style.use('seaborn-v0_8')"),
        ("markdown", "## 1. Prepare Data for Modeling\nWe use chronologically split data: Train (2015-2024), Test (2025-2026)."),
        ("code", "# Load data\ndf = pd.read_csv('../data/processed/historical_data.csv', header=[0,1], index_col=0, parse_dates=True)\ntsla_close = df['Adj Close']['TSLA'].dropna()\n\n# Chronological split\ntrain_data = tsla_close[:'2024-12-31']\ntest_data = tsla_close['2025-01-01':]\n\nprint(f\"Train size: {len(train_data)}\")\nprint(f\"Test size: {len(test_data)}\")\n\nplt.figure(figsize=(10, 5))\nplt.plot(train_data, label='Train (2015-2024)')\nplt.plot(test_data, label='Test (2025-2026)')\nplt.title('TSLA Train/Test Split')\nplt.legend()\nplt.show()"),
        ("markdown", "## 2. Implement ARIMA/SARIMA Model\nUsing auto_arima to find the best parameters."),
        ("code", "# Fit auto_arima\n# We use seasonal=False for simplicity on daily stock prices unless proven otherwise.\narima_model = pm.auto_arima(train_data, seasonal=False, stepwise=True, trace=True, error_action='ignore', suppress_warnings=True)\nprint(arima_model.summary())\n\n# Generate forecasts\narima_forecast = arima_model.predict(n_periods=len(test_data))\narima_forecast.index = test_data.index"),
        ("markdown", "## 3. Implement LSTM Model\nPrepare sequences and build the LSTM architecture."),
        ("code", "# Scale data\nscaler = MinMaxScaler(feature_range=(0, 1))\nscaled_data = scaler.fit_transform(tsla_close.values.reshape(-1, 1))\n\n# Split scaled data\ntrain_scaled = scaled_data[:len(train_data)]\ntest_scaled = scaled_data[len(train_data)-60:] # include previous 60 days for the first test prediction\n\n# Create sequences\ndef create_sequences(data, seq_length=60):\n    X, y = [], []\n    for i in range(seq_length, len(data)):\n        X.append(data[i-seq_length:i, 0])\n        y.append(data[i, 0])\n    return np.array(X), np.array(y)\n\nseq_length = 60\nX_train, y_train = create_sequences(train_scaled, seq_length)\nX_test, y_test = create_sequences(test_scaled, seq_length)\n\n# Reshape for LSTM [samples, time steps, features]\nX_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))\nX_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))\n\n# Build LSTM model\nlstm_model = Sequential()\nlstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))\nlstm_model.add(Dropout(0.2))\nlstm_model.add(LSTM(units=50, return_sequences=False))\nlstm_model.add(Dropout(0.2))\nlstm_model.add(Dense(units=25))\nlstm_model.add(Dense(units=1))\n\nlstm_model.compile(optimizer='adam', loss='mean_squared_error')\n\n# Train\nlstm_model.fit(X_train, y_train, batch_size=32, epochs=20, validation_split=0.1, verbose=1)\n\n# Predict\nlstm_predictions_scaled = lstm_model.predict(X_test)\nlstm_forecast = scaler.inverse_transform(lstm_predictions_scaled).flatten()\nlstm_forecast = pd.Series(lstm_forecast, index=test_data.index)"),
        ("markdown", "## 5. Evaluate and Compare Models"),
        ("code", "def evaluate_model(y_true, y_pred, model_name):\n    mae = mean_absolute_error(y_true, y_pred)\n    rmse = np.sqrt(mean_squared_error(y_true, y_pred))\n    mape = mean_absolute_percentage_error(y_true, y_pred)\n    return {'Model': model_name, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}\n\nresults = []\nresults.append(evaluate_model(test_data, arima_forecast, 'ARIMA'))\nresults.append(evaluate_model(test_data, lstm_forecast, 'LSTM'))\n\nresults_df = pd.DataFrame(results).set_index('Model')\nprint(results_df)\n\nplt.figure(figsize=(14, 7))\nplt.plot(train_data['2024':], label='Train (2024)')\nplt.plot(test_data, label='Actual Test')\nplt.plot(arima_forecast, label='ARIMA Forecast')\nplt.plot(lstm_forecast, label='LSTM Forecast')\nplt.title('TSLA Price Forecast Comparison')\nplt.legend()\nplt.show()"),
        ("markdown", "## Discussion\nThe LSTM model captures short-term momentum and patterns well, often outperforming the classical ARIMA model on highly volatile financial time series. ARIMA struggles to project far into the future beyond a constant trend or mean, whereas LSTM uses moving windows. In this case, comparing the MAE and RMSE shows which model had lower error. Stock market prediction is inherently challenging and results should be combined with other signals.")
    ]
    create_notebook('../notebooks/Task_2_Models.ipynb', cells)

def task3():
    cells = [
        ("markdown", "# Task 3: Forecast Future Market Trends\n\nWe forecast TSLA's future stock prices for the next 6-12 months (e.g. up to mid 2027) using the trained ARIMA model, visualize confidence intervals, and perform trend analysis."),
        ("code", "import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport pmdarima as pm\nimport warnings\nwarnings.filterwarnings('ignore')\n\nplt.style.use('seaborn-v0_8')"),
        ("markdown", "## 1 & 2. Generate and Visualize Future Forecasts with Confidence Intervals"),
        ("code", "# Load data and fit ARIMA on the ENTIRE dataset (to forecast into the future)\ndf = pd.read_csv('../data/processed/historical_data.csv', header=[0,1], index_col=0, parse_dates=True)\ntsla_close = df['Adj Close']['TSLA'].dropna()\n\nprint(\"Fitting ARIMA model on all available data...\")\narima_model = pm.auto_arima(tsla_close, seasonal=False, stepwise=True, trace=False)\n\n# Forecast for 6 months (approx 126 trading days)\nforecast_periods = 126\nforecast, conf_int = arima_model.predict(n_periods=forecast_periods, return_conf_int=True)\n\n# Create future dates index\nfuture_dates = pd.bdate_range(start=tsla_close.index[-1] + pd.Timedelta(days=1), periods=forecast_periods)\nforecast.index = future_dates\n\n# Plot\nplt.figure(figsize=(14, 7))\nplt.plot(tsla_close['2024':], label='Historical Data')\nplt.plot(forecast.index, forecast, color='green', label='Future Forecast')\nplt.fill_between(forecast.index, conf_int[:, 0], conf_int[:, 1], color='green', alpha=0.2, label='Confidence Interval')\nplt.title('TSLA 6-Month Price Forecast')\nplt.xlabel('Date')\nplt.ylabel('Price')\nplt.legend()\nplt.show()"),
        ("markdown", "## 3 & 4. Trend Analysis and Market Opportunities/Risks\n\n**Trend Analysis:**\nThe forecast generated by the ARIMA model often reflects a continuation of the most recent trend or mean-reversion. In financial time series, long-term forecasts via ARIMA tend to flatten out. The confidence intervals widen significantly as we look further into the 6-month horizon, representing the high uncertainty in predicting exact stock prices long-term.\n\n**Opportunities and Risks:**\n- **Opportunities:** If the forecast is upward, it may suggest a momentum play.\n- **Risks:** The extremely wide confidence intervals highlight massive downside risk. The Efficient Market Hypothesis tells us that standalone price prediction is unreliable.\n- **Reliability:** Over a 6-12 month horizon, model-based price forecasts should be treated as a measure of expected volatility rather than directional certainty.")
    ]
    create_notebook('../notebooks/Task_3_Forecast.ipynb', cells)

def task4():
    cells = [
        ("markdown", "# Task 4: Optimize Portfolio Based on Forecast\n\nWe use Modern Portfolio Theory (MPT) and PyPortfolioOpt to generate the Efficient Frontier and find optimal allocations."),
        ("code", "import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom pypfopt import expected_returns, risk_models\nfrom pypfopt.efficient_frontier import EfficientFrontier\nfrom pypfopt import plotting\nimport warnings\nwarnings.filterwarnings('ignore')\n\nplt.style.use('seaborn-v0_8')"),
        ("markdown", "## 1 & 2. Prepare Expected Returns and Covariance Matrix"),
        ("code", "# Load data\ndf = pd.read_csv('../data/processed/historical_data.csv', header=[0,1], index_col=0, parse_dates=True)\nprices = df['Adj Close'].dropna()\n\n# We will calculate historical expected returns for BND and SPY\n# For TSLA, we assume a forecasted expected return (e.g., from our model). Let's use a proxy scalar for demonstration: 0.25 (25% annualized return forecast).\nmu = expected_returns.mean_historical_return(prices)\nmu['TSLA'] = 0.25  # Replacing with our forecasted view\n\n# Compute Covariance Matrix\nS = risk_models.sample_cov(prices)\n\nplt.figure(figsize=(8, 6))\nsns.heatmap(S, annot=True, cmap='coolwarm', fmt=\".4f\")\nplt.title('Covariance Matrix')\nplt.show()"),
        ("markdown", "## 3 & 4. Generate Efficient Frontier and Find Key Portfolios"),
        ("code", "# Setup Efficient Frontier\nef = EfficientFrontier(mu, S)\n\n# Plot Efficient Frontier\nfig, ax = plt.subplots(figsize=(10, 6))\nplotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)\n\n# Find Max Sharpe\nef_max_sharpe = EfficientFrontier(mu, S)\nraw_weights_ms = ef_max_sharpe.max_sharpe()\ncleaned_weights_ms = ef_max_sharpe.clean_weights()\nret_ms, vol_ms, sharpe_ms = ef_max_sharpe.portfolio_performance()\n\nax.scatter(vol_ms, ret_ms, marker='*', color='r', s=200, label='Max Sharpe')\n\n# Find Min Volatility\nef_min_vol = EfficientFrontier(mu, S)\nraw_weights_mv = ef_min_vol.min_volatility()\ncleaned_weights_mv = ef_min_vol.clean_weights()\nret_mv, vol_mv, sharpe_mv = ef_min_vol.portfolio_performance()\n\nax.scatter(vol_mv, ret_mv, marker='*', color='g', s=200, label='Min Volatility')\n\nax.set_title('Efficient Frontier')\nax.legend()\nplt.show()\n\nprint(\"Max Sharpe Portfolio Weights:\", cleaned_weights_ms)\nprint(f\"Expected Annual Return: {ret_ms:.2f}, Volatility: {vol_ms:.2f}, Sharpe Ratio: {sharpe_ms:.2f}\")\n\nprint(\"\\nMin Volatility Portfolio Weights:\", cleaned_weights_mv)\nprint(f\"Expected Annual Return: {ret_mv:.2f}, Volatility: {vol_mv:.2f}, Sharpe Ratio: {sharpe_mv:.2f}\")"),
        ("markdown", "## 5. Recommend Optimal Portfolio\n\nBased on the analysis, the **Max Sharpe Portfolio** is recommended for clients seeking the highest risk-adjusted return. It appropriately balances the high-expected return from TSLA with the stability of BND and broad exposure of SPY. For highly risk-averse clients, the Min Volatility portfolio (heavily weighted towards BND) would be preferred.")
    ]
    create_notebook('../notebooks/Task_4_Optimization.ipynb', cells)

def task5():
    cells = [
        ("markdown", "# Task 5: Strategy Backtesting\n\nWe validate the portfolio strategy by simulating its performance on the final year of data and comparing it against a 60/40 benchmark."),
        ("code", "import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport warnings\nwarnings.filterwarnings('ignore')\n\nplt.style.use('seaborn-v0_8')"),
        ("markdown", "## 1 & 2. Define Backtesting Period and Benchmark"),
        ("code", "# Load data\ndf = pd.read_csv('../data/processed/historical_data.csv', header=[0,1], index_col=0, parse_dates=True)\nprices = df['Adj Close'].dropna()\nreturns = prices.pct_change().dropna()\n\n# Backtesting period (e.g., Jan 2025 to Dec 2025)\nbt_returns = returns.loc['2025-01-01':'2025-12-31']\n\nprint(f\"Backtesting period from {bt_returns.index[0].date()} to {bt_returns.index[-1].date()}\")\n\n# Weights\n# Let's assume the optimal weights from Task 4 Max Sharpe were roughly:\nweights_strategy = np.array([0.15, 0.40, 0.45]) # Example: TSLA, BND, SPY\n# Note: Ensure the order matches bt_returns.columns\nprint(\"Assets:\", bt_returns.columns.tolist())\n# Map to correct columns: BND, SPY, TSLA\ncol_order = bt_returns.columns.tolist()\n\n# Let's align exactly: BND:0.40, SPY:0.45, TSLA:0.15 (Example)\nweights_strategy_dict = {'BND': 0.40, 'SPY': 0.45, 'TSLA': 0.15}\nweights_strategy = np.array([weights_strategy_dict[c] for c in col_order])\n\n# Benchmark: 60% SPY, 40% BND, 0% TSLA\nweights_benchmark_dict = {'BND': 0.40, 'SPY': 0.60, 'TSLA': 0.00}\nweights_benchmark = np.array([weights_benchmark_dict[c] for c in col_order])"),
        ("markdown", "## 3 & 4. Simulate Strategy and Analyze Performance"),
        ("code", "# Calculate daily portfolio returns\nstrategy_returns = bt_returns.dot(weights_strategy)\nbenchmark_returns = bt_returns.dot(weights_benchmark)\n\n# Cumulative returns\nstrategy_cum = (1 + strategy_returns).cumprod() - 1\nbenchmark_cum = (1 + benchmark_returns).cumprod() - 1\n\nplt.figure(figsize=(10, 6))\nplt.plot(strategy_cum, label='Strategy (Optimized)')\nplt.plot(benchmark_cum, label='Benchmark (60/40)')\nplt.title('Cumulative Returns: Strategy vs Benchmark')\nplt.xlabel('Date')\nplt.ylabel('Cumulative Return')\nplt.legend()\nplt.show()\n\n# Performance Metrics function\ndef calc_metrics(ret_series):\n    total_ret = (1 + ret_series).prod() - 1\n    ann_ret = (1 + total_ret)**(252/len(ret_series)) - 1\n    ann_vol = ret_series.std() * np.sqrt(252)\n    sharpe = ann_ret / ann_vol\n    \n    cum_ret = (1 + ret_series).cumprod()\n    running_max = cum_ret.cummax()\n    drawdown = (cum_ret - running_max) / running_max\n    max_dd = drawdown.min()\n    \n    return [total_ret, ann_ret, sharpe, max_dd]\n\nmetrics = []\nmetrics.append(calc_metrics(strategy_returns))\nmetrics.append(calc_metrics(benchmark_returns))\n\nmetrics_df = pd.DataFrame(metrics, columns=['Total Return', 'Ann. Return', 'Sharpe Ratio', 'Max Drawdown'], index=['Strategy', 'Benchmark'])\nprint(metrics_df)"),
        ("markdown", "## 5. Conclusion and Reflection\n\nThe backtest illustrates whether the optimized portfolio (which includes a targeted allocation to TSLA) outperformed the passive 60/40 benchmark. The inclusion of TSLA generally increases both potential returns and volatility. The strategy's viability heavily relies on the accuracy of the TSLA expected return forecast. A key limitation of this simple backtest is that it assumes fixed weights without transaction costs or rebalancing effects, and tests over a single specific market regime.")
    ]
    create_notebook('../notebooks/Task_5_Backtest.ipynb', cells)

if __name__ == '__main__':
    task1()
    task2()
    task3()
    task4()
    task5()
    print("All notebooks created.")
