import yfinance as yf
import pandas as pd
import os

def fetch_financial_data(tickers, start_date, end_date):
    """
    Fetches historical financial data for given tickers from yfinance.
    """
    print(f"Fetching data for {tickers} from {start_date} to {end_date}...")
    data = yf.download(tickers, start=start_date, end=end_date)
    return data

def clean_data(df):
    """
    Cleans the dataframe by handling missing values.
    Uses forward fill followed by backward fill to ensure no NaNs.
    """
    print("Cleaning data...")
    # Fill missing values: forward fill then backward fill
    df = df.ffill().bfill()
    return df

def save_data(df, filepath):
    """
    Saves dataframe to a CSV file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath)
    print(f"Data saved to {filepath}")

def process_and_save(tickers, start_date, end_date, output_filepath):
    df = fetch_financial_data(tickers, start_date, end_date)
    df = clean_data(df)
    save_data(df, output_filepath)
    return df

if __name__ == "__main__":
    tickers = ["TSLA", "BND", "SPY"]
    start_date = "2015-01-01"
    end_date = "2026-06-30"
    output_filepath = "data/processed/historical_data.csv"
    process_and_save(tickers, start_date, end_date, output_filepath)
