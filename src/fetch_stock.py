import yfinance as yf
import pandas as pd
import time
import os
from datetime import datetime

# Constants
DATA_PATH = "../data/"
STOCK_SYMBOL = "AAPL"
INTERVAL = "5m"
DAYS = 7
SLEEP_DURATION = 300  # 5 minutes in seconds

def fetch_stock_data(stock_symbol=STOCK_SYMBOL, interval=INTERVAL, days=DAYS):
    """Fetch stock data every 5 minutes and save it."""
    os.makedirs(DATA_PATH, exist_ok=True)  # Ensure data directory exists

    while True:
        try:
            # Fetch stock data
            stock_data = yf.download(stock_symbol, period=f"{days}d", interval=interval)

            # Add a timestamp for when the data was fetched
            stock_data["fetch_time"] = datetime.now()

            # Save data to a Parquet file
            file_path = os.path.join(DATA_PATH, f"{stock_symbol}_stock_data.parquet")
            stock_data.to_parquet(file_path, compression="snappy")

            print(f"Fetched latest stock data for {stock_symbol} at {datetime.now()}")

        except Exception as e:
            print(f"Error fetching data: {e}")

        # Wait for 5 minutes before the next update
        time.sleep(SLEEP_DURATION)

if __name__ == "__main__":
    fetch_stock_data()