import yfinance as yf
import pandas as pd
import datetime
import os

def fetch_market_data(ticker: str, interval: str = '1m', period: str = '1d'):
    print(f"Fetching market data for {ticker}...")
    try:
        stock_data = yf.download(ticker, interval=interval, period=period)
        stock_data.reset_index(inplace=True)
        return stock_data
    except Exception as e:
        print(f"Error fetching market data: {e}")
        return None

def save_data(df: pd.DataFrame, filename: str):
    """
    Saves DataFrame to CSV file.
    """
    os.makedirs("../data/market_data", exist_ok=True)
    filepath = os.path.join("../data/market_data", filename)
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")

if __name__ == "__main__":
    ticker = "AAPL"  # Example: Apple stock
    data = fetch_market_data(ticker)
    if data is not None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_data(data, f"{ticker}_market_data_{timestamp}.csv")
