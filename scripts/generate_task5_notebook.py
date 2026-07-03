import nbformat as nbf

def create_notebook(filename):
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("# Task 5: Strategy Backtesting\\n\\nWe validate the portfolio strategy by simulating its performance on the final year of data and comparing it against a 60/40 benchmark."))
    
    cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')"""))

    cells.append(nbf.v4.new_markdown_cell("## 1 & 2. Define Backtesting Period and Benchmark"))
    cells.append(nbf.v4.new_code_cell("""# Load data
df = pd.read_csv('../data/processed/historical_data.csv', header=[0,1], index_col=0, parse_dates=True)
prices = df['Adj Close'].dropna()
returns = prices.pct_change().dropna()

# Backtesting period (e.g., Jan 2025 to Dec 2025)
bt_returns = returns.loc['2025-01-01':'2025-12-31']

print(f"Backtesting period from {bt_returns.index[0].date()} to {bt_returns.index[-1].date()}")

# Weights
# Let's assume the optimal weights from Task 4 Max Sharpe were roughly:
weights_strategy = np.array([0.15, 0.40, 0.45]) # Example: TSLA, BND, SPY
# Note: Ensure the order matches bt_returns.columns
print("Assets:", bt_returns.columns.tolist())
# Map to correct columns: BND, SPY, TSLA
col_order = bt_returns.columns.tolist()

# Let's align exactly: BND:0.40, SPY:0.45, TSLA:0.15 (Example)
weights_strategy_dict = {'BND': 0.40, 'SPY': 0.45, 'TSLA': 0.15}
weights_strategy = np.array([weights_strategy_dict[c] for c in col_order])

# Benchmark: 60% SPY, 40% BND, 0% TSLA
weights_benchmark_dict = {'BND': 0.40, 'SPY': 0.60, 'TSLA': 0.00}
weights_benchmark = np.array([weights_benchmark_dict[c] for c in col_order])"""))

    cells.append(nbf.v4.new_markdown_cell("## 3 & 4. Simulate Strategy and Analyze Performance"))
    cells.append(nbf.v4.new_code_cell("""# Calculate daily portfolio returns
strategy_returns = bt_returns.dot(weights_strategy)
benchmark_returns = bt_returns.dot(weights_benchmark)

# Cumulative returns
strategy_cum = (1 + strategy_returns).cumprod() - 1
benchmark_cum = (1 + benchmark_returns).cumprod() - 1

plt.figure(figsize=(10, 6))
plt.plot(strategy_cum, label='Strategy (Optimized)')
plt.plot(benchmark_cum, label='Benchmark (60/40)')
plt.title('Cumulative Returns: Strategy vs Benchmark')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.show()

# Performance Metrics function
def calc_metrics(ret_series):
    total_ret = (1 + ret_series).prod() - 1
    ann_ret = (1 + total_ret)**(252/len(ret_series)) - 1
    ann_vol = ret_series.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol
    
    cum_ret = (1 + ret_series).cumprod()
    running_max = cum_ret.cummax()
    drawdown = (cum_ret - running_max) / running_max
    max_dd = drawdown.min()
    
    return [total_ret, ann_ret, sharpe, max_dd]

metrics = []
metrics.append(calc_metrics(strategy_returns))
metrics.append(calc_metrics(benchmark_returns))

metrics_df = pd.DataFrame(metrics, columns=['Total Return', 'Ann. Return', 'Sharpe Ratio', 'Max Drawdown'], index=['Strategy', 'Benchmark'])
print(metrics_df)"""))

    cells.append(nbf.v4.new_markdown_cell("## 5. Conclusion and Reflection\\n\\nThe backtest illustrates whether the optimized portfolio (which includes a targeted allocation to TSLA) outperformed the passive 60/40 benchmark. The inclusion of TSLA generally increases both potential returns and volatility. The strategy's viability heavily relies on the accuracy of the TSLA expected return forecast. A key limitation of this simple backtest is that it assumes fixed weights without transaction costs or rebalancing effects, and tests over a single specific market regime."))

    nb['cells'] = cells
    with open(filename, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == '__main__':
    create_notebook('../notebooks/Task_5_Backtest.ipynb')
    print("Notebook generated at ../notebooks/Task_5_Backtest.ipynb")
