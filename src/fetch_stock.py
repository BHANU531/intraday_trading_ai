import yfinance as yf
import pandas as pd
import time
import os
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/stock_fetcher.log'),
        logging.StreamHandler()
    ]
)

# Constants
DATA_PATH = "data/"
STOCK_SYMBOL = "AAPL"
INTERVAL = "5m"
DAYS = 7

def setup_directories():
    """Ensure all required directories exist."""
    os.makedirs(DATA_PATH, exist_ok=True)
    logging.info(f"Ensuring data directory exists at {DATA_PATH}")

def fetch_stock_data(stock_symbol=STOCK_SYMBOL, interval=INTERVAL, days=DAYS):
    """Fetch stock data and save it."""
    setup_directories()
    
    try:
        # Calculate the start date to ensure we get data during market hours
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logging.info(f"Fetching stock data for {stock_symbol} from {start_date} to {end_date}")
        
        # Fetch stock data
        stock_data = yf.download(
            stock_symbol,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False
        )
        
        if stock_data.shape[0] == 0:
            logging.warning(f"No data received for {stock_symbol}")
            return None
        
        # Add a timestamp for when the data was fetched
        stock_data["fetch_time"] = datetime.now()
        
        # Save data to a Parquet file
        file_path = os.path.join(DATA_PATH, f"{stock_symbol}_stock_data.parquet")
        stock_data.to_parquet(file_path, compression="snappy")
        
        logging.info(f"Successfully saved {stock_data.shape[0]} records to {file_path}")
        
        # Log the latest close price if available
        if stock_data.shape[0] > 0:
            latest_close = stock_data['Close'].iloc[-1]
            if pd.notna(latest_close):
                logging.info(f"Latest close price: ${latest_close:.2f}")
        
        return stock_data
        
    except Exception as e:
        logging.error(f"Error fetching data: {str(e)}")
        return None

if __name__ == "__main__":
    try:
        logging.info("Starting stock data fetcher...")
        data = fetch_stock_data()
        if data is not None and data.shape[0] > 0:
            logging.info("Stock data fetching completed successfully")
        else:
            logging.error("Failed to fetch stock data")
    except KeyboardInterrupt:
        logging.info("Stock data fetcher stopped by user")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")