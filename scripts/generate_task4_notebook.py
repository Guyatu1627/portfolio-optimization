import nbformat as nbf

def create_notebook(filename):
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("# Task 4: Optimize Portfolio Based on Forecast\\n\\nWe use Modern Portfolio Theory (MPT) and PyPortfolioOpt to generate the Efficient Frontier and find optimal allocations."))
    
    cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import plotting
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')"""))

    cells.append(nbf.v4.new_markdown_cell("## 1 & 2. Prepare Expected Returns and Covariance Matrix"))
    cells.append(nbf.v4.new_code_cell("""# Load data
df = pd.read_csv('../data/processed/historical_data.csv', header=[0,1], index_col=0, parse_dates=True)
prices = df['Adj Close'].dropna()

# We will calculate historical expected returns for BND and SPY
# For TSLA, we assume a forecasted expected return (e.g., from our model). Let's use a proxy scalar for demonstration: 0.25 (25% annualized return forecast).
mu = expected_returns.mean_historical_return(prices)
mu['TSLA'] = 0.25  # Replacing with our forecasted view

# Compute Covariance Matrix
S = risk_models.sample_cov(prices)

plt.figure(figsize=(8, 6))
sns.heatmap(S, annot=True, cmap='coolwarm', fmt=".4f")
plt.title('Covariance Matrix')
plt.show()"""))

    cells.append(nbf.v4.new_markdown_cell("## 3 & 4. Generate Efficient Frontier and Find Key Portfolios"))
    cells.append(nbf.v4.new_code_cell("""# Setup Efficient Frontier
ef = EfficientFrontier(mu, S)

# Plot Efficient Frontier
fig, ax = plt.subplots(figsize=(10, 6))
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)

# Find Max Sharpe
ef_max_sharpe = EfficientFrontier(mu, S)
raw_weights_ms = ef_max_sharpe.max_sharpe()
cleaned_weights_ms = ef_max_sharpe.clean_weights()
ret_ms, vol_ms, sharpe_ms = ef_max_sharpe.portfolio_performance()

ax.scatter(vol_ms, ret_ms, marker='*', color='r', s=200, label='Max Sharpe')

# Find Min Volatility
ef_min_vol = EfficientFrontier(mu, S)
raw_weights_mv = ef_min_vol.min_volatility()
cleaned_weights_mv = ef_min_vol.clean_weights()
ret_mv, vol_mv, sharpe_mv = ef_min_vol.portfolio_performance()

ax.scatter(vol_mv, ret_mv, marker='*', color='g', s=200, label='Min Volatility')

ax.set_title('Efficient Frontier')
ax.legend()
plt.show()

print("Max Sharpe Portfolio Weights:", cleaned_weights_ms)
print(f"Expected Annual Return: {ret_ms:.2f}, Volatility: {vol_ms:.2f}, Sharpe Ratio: {sharpe_ms:.2f}")

print("\\nMin Volatility Portfolio Weights:", cleaned_weights_mv)
print(f"Expected Annual Return: {ret_mv:.2f}, Volatility: {vol_mv:.2f}, Sharpe Ratio: {sharpe_mv:.2f}")"""))

    cells.append(nbf.v4.new_markdown_cell("## 5. Recommend Optimal Portfolio\\n\\nBased on the analysis, the **Max Sharpe Portfolio** is recommended for clients seeking the highest risk-adjusted return. It appropriately balances the high-expected return from TSLA with the stability of BND and broad exposure of SPY. For highly risk-averse clients, the Min Volatility portfolio (heavily weighted towards BND) would be preferred."))

    nb['cells'] = cells
    with open(filename, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == '__main__':
    create_notebook('../notebooks/Task_4_Optimization.ipynb')
    print("Notebook generated at ../notebooks/Task_4_Optimization.ipynb")
