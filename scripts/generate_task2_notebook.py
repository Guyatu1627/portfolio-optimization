import nbformat as nbf

def create_notebook(filename):
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("# Task 2: Build Time Series Forecasting Models\\n\\nWe will develop and evaluate time series models (ARIMA and LSTM) to predict TSLA future stock prices."))
    
    cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import pmdarima as pm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')"""))

    cells.append(nbf.v4.new_markdown_cell("## 1. Prepare Data for Modeling\\nWe use chronologically split data: Train (2015-2024), Test (2025-2026)."))
    cells.append(nbf.v4.new_code_cell("""# Load data
df = pd.read_csv('../data/processed/historical_data.csv', header=[0,1], index_col=0, parse_dates=True)
tsla_close = df['Adj Close']['TSLA'].dropna()

# Chronological split
train_data = tsla_close[:'2024-12-31']
test_data = tsla_close['2025-01-01':]

print(f"Train size: {len(train_data)}")
print(f"Test size: {len(test_data)}")

plt.figure(figsize=(10, 5))
plt.plot(train_data, label='Train (2015-2024)')
plt.plot(test_data, label='Test (2025-2026)')
plt.title('TSLA Train/Test Split')
plt.legend()
plt.show()"""))

    cells.append(nbf.v4.new_markdown_cell("## 2. Implement ARIMA/SARIMA Model\\nUsing auto_arima to find the best parameters."))
    cells.append(nbf.v4.new_code_cell("""# Fit auto_arima
# We use seasonal=False for simplicity on daily stock prices unless proven otherwise.
arima_model = pm.auto_arima(train_data, seasonal=False, stepwise=True, trace=True, error_action='ignore', suppress_warnings=True)
print(arima_model.summary())

# Generate forecasts
arima_forecast = arima_model.predict(n_periods=len(test_data))
arima_forecast.index = test_data.index"""))

    cells.append(nbf.v4.new_markdown_cell("## 3. Implement LSTM Model\\nPrepare sequences and build the LSTM architecture."))
    cells.append(nbf.v4.new_code_cell("""# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(tsla_close.values.reshape(-1, 1))

# Split scaled data
train_scaled = scaled_data[:len(train_data)]
test_scaled = scaled_data[len(train_data)-60:] # include previous 60 days for the first test prediction

# Create sequences
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

seq_length = 60
X_train, y_train = create_sequences(train_scaled, seq_length)
X_test, y_test = create_sequences(test_scaled, seq_length)

# Reshape for LSTM [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=50, return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units=25))
lstm_model.add(Dense(units=1))

lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train
lstm_model.fit(X_train, y_train, batch_size=32, epochs=20, validation_split=0.1, verbose=1)

# Predict
lstm_predictions_scaled = lstm_model.predict(X_test)
lstm_forecast = scaler.inverse_transform(lstm_predictions_scaled).flatten()
lstm_forecast = pd.Series(lstm_forecast, index=test_data.index)"""))

    cells.append(nbf.v4.new_markdown_cell("## 5. Evaluate and Compare Models"))
    cells.append(nbf.v4.new_code_cell("""def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {'Model': model_name, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

results = []
results.append(evaluate_model(test_data, arima_forecast, 'ARIMA'))
results.append(evaluate_model(test_data, lstm_forecast, 'LSTM'))

results_df = pd.DataFrame(results).set_index('Model')
print(results_df)

plt.figure(figsize=(14, 7))
plt.plot(train_data['2024':], label='Train (2024)')
plt.plot(test_data, label='Actual Test')
plt.plot(arima_forecast, label='ARIMA Forecast')
plt.plot(lstm_forecast, label='LSTM Forecast')
plt.title('TSLA Price Forecast Comparison')
plt.legend()
plt.show()"""))
    
    cells.append(nbf.v4.new_markdown_cell("## Discussion\nThe LSTM model captures short-term momentum and patterns well, often outperforming the classical ARIMA model on highly volatile financial time series. ARIMA struggles to project far into the future beyond a constant trend or mean, whereas LSTM uses moving windows. In this case, comparing the MAE and RMSE shows which model had lower error. Stock market prediction is inherently challenging and results should be combined with other signals."))

    nb['cells'] = cells
    with open(filename, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == '__main__':
    create_notebook('../notebooks/Task_2_Models.ipynb')
    print("Notebook generated at ../notebooks/Task_2_Models.ipynb")
