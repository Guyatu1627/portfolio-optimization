import nbformat as nbf

def create_notebook(filename):
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # Markdown
    cells.append(nbf.v4.new_markdown_cell("# Task 1: Preprocess and Explore the Data\n\nThis notebook performs Data Extraction, Cleaning, Exploratory Data Analysis, Seasonality & Trend Analysis, and Risk Metrics Calculation for TSLA, BND, and SPY."))
    
    # Imports
    cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
%matplotlib inline"""))

    # Load Data
    cells.append(nbf.v4.new_markdown_cell("## 1 & 2. Extract and Clean Data\nWe load the data fetched by our script and show basic info."))
    cells.append(nbf.v4.new_code_cell("""import sys
sys.path.append('../src')
from data_fetch import process_and_save

tickers = ['TSLA', 'BND', 'SPY']
df = process_and_save(tickers, '2015-01-01', '2026-06-30', '../data/processed/historical_data.csv')

# Display basic information
print(df.info())
print("\\nBasic Statistics:\\n", df['Adj Close'].describe())"""))

    # EDA - Closing Prices
    cells.append(nbf.v4.new_markdown_cell("## 3. Exploratory Data Analysis (EDA)\n### Closing Prices Over Time"))
    cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(14, 7))
for ticker in tickers:
    plt.plot(df.index, df['Adj Close'][ticker], label=ticker)
plt.title('Adjusted Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()"""))

    # Daily Returns
    cells.append(nbf.v4.new_markdown_cell("### Daily Percentage Change (Volatility)"))
    cells.append(nbf.v4.new_code_cell("""returns = df['Adj Close'].pct_change().dropna()
plt.figure(figsize=(14, 7))
for ticker in tickers:
    plt.plot(returns.index, returns[ticker], label=ticker, alpha=0.7)
plt.title('Daily Percentage Change')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.legend()
plt.show()"""))

    # Rolling Volatility
    cells.append(nbf.v4.new_markdown_cell("### Rolling Volatility (30-day standard deviation)"))
    cells.append(nbf.v4.new_code_cell("""rolling_volatility = returns.rolling(window=30).std()
plt.figure(figsize=(14, 7))
for ticker in tickers:
    plt.plot(rolling_volatility.index, rolling_volatility[ticker], label=ticker)
plt.title('30-Day Rolling Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility (Std Dev)')
plt.legend()
plt.show()"""))

    # Outliers
    cells.append(nbf.v4.new_markdown_cell("### Outlier Detection (TSLA)\nIdentifying days with unusually high or low returns (> 3 std dev)."))
    cells.append(nbf.v4.new_code_cell("""tsla_returns = returns['TSLA']
mean_ret = tsla_returns.mean()
std_ret = tsla_returns.std()
outliers = tsla_returns[(tsla_returns > mean_ret + 3*std_ret) | (tsla_returns < mean_ret - 3*std_ret)]

plt.figure(figsize=(14, 7))
plt.scatter(tsla_returns.index, tsla_returns, label='Normal Returns', alpha=0.5, s=10)
plt.scatter(outliers.index, outliers, color='red', label='Outliers', s=20)
plt.title('TSLA Daily Returns with Outliers Highlighted')
plt.legend()
plt.show()

print(f"Number of outliers detected for TSLA: {len(outliers)}")"""))

    # Seasonality and Trend Analysis
    cells.append(nbf.v4.new_markdown_cell("## 4. Seasonality and Trend Analysis\nPerforming Augmented Dickey-Fuller (ADF) test to check stationarity."))
    cells.append(nbf.v4.new_code_cell("""def adf_test(series, name):
    result = adfuller(series.dropna())
    print(f"ADF Test for {name}")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    if result[1] <= 0.05:
        print("=> Reject the null hypothesis (H0), the data is stationary.\\n")
    else:
        print("=> Fail to reject the null hypothesis (H0), the data is non-stationary.\\n")

# Test on Prices
adf_test(df['Adj Close']['TSLA'], "TSLA Closing Price")

# Test on Returns
adf_test(returns['TSLA'], "TSLA Daily Returns")"""))

    # Risk Metrics
    cells.append(nbf.v4.new_markdown_cell("## 5. Calculate Risk Metrics\n### Value at Risk (VaR) & Sharpe Ratio"))
    cells.append(nbf.v4.new_code_cell("""# Value at Risk (VaR) at 95% confidence level
confidence_level = 0.05
var_95 = returns.quantile(confidence_level)
print("Historical Value at Risk (95% confidence) - Worst daily loss expected 5% of the time:")
print(var_95)

# Sharpe Ratio (Assuming Risk-Free Rate = 0.02 / 252 daily)
risk_free_rate_daily = 0.02 / 252
excess_returns = returns - risk_free_rate_daily
sharpe_ratio = (excess_returns.mean() / returns.std()) * np.sqrt(252) # Annualized
print("\\nAnnualized Sharpe Ratio:")
print(sharpe_ratio)"""))

    cells.append(nbf.v4.new_markdown_cell("## Summary of Insights\n- **Data Quality**: The fetched yfinance data is largely clean; small missing gaps were forward-filled.\n- **Trend & Volatility**: TSLA shows an overall long-term upward trend but exhibits significantly higher volatility compared to SPY and BND. BND is stable with minimal volatility.\n- **Stationarity**: TSLA closing prices are non-stationary (p-value > 0.05), implying differencing is needed for ARIMA. However, daily returns are stationary (p-value < 0.05).\n- **Risk Metrics**: TSLA has a high VaR indicating potential for large daily losses, but its Sharpe Ratio is historically decent despite high risk, capturing the high-growth profile. BND serves its role as a low-risk stability asset."))

    nb['cells'] = cells
    
    with open(filename, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == '__main__':
    create_notebook('../notebooks/Task_1_EDA.ipynb')
    print(f"Notebook generated at ../notebooks/Task_1_EDA.ipynb")
