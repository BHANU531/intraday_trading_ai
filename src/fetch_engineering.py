import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/engineering.log'),
        logging.StreamHandler()
    ]
)

# Constants
DATA_PATH = "data/"
STOCK_SYMBOL = "AAPL"

def setup_directories():
    """Ensure all required directories exist."""
    os.makedirs(DATA_PATH, exist_ok=True)
    logging.info(f"Ensuring data directory exists at {DATA_PATH}")

def load_stock_data():
    """Load the most recent stock data."""
    file_path = os.path.join(DATA_PATH, f"{STOCK_SYMBOL}_stock_data.parquet")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Stock data not found at {file_path}")
    
    df = pd.read_parquet(file_path)
    logging.info(f"Loaded {len(df)} records of stock data")
    return df

def calculate_sma(data, window):
    """Calculate Simple Moving Average."""
    return data.rolling(window=window).mean()

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    """Calculate MACD (Moving Average Convergence Divergence)."""
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(data, window=20):
    """Calculate Bollinger Bands."""
    sma = calculate_sma(data, window)
    std = data.rolling(window=window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, lower_band

def calculate_indicators():
    """Calculate all technical indicators."""
    try:
        # Load stock data
        df = load_stock_data()
        
        # Calculate basic indicators
        logging.info("Calculating technical indicators...")
        
        # SMA for different periods
        df['SMA_10'] = calculate_sma(df['Close'], 10)
        df['SMA_20'] = calculate_sma(df['Close'], 20)
        df['SMA_50'] = calculate_sma(df['Close'], 50)
        
        # RSI
        df['RSI'] = calculate_rsi(df['Close'])
        
        # MACD
        df['MACD'], df['MACD_Signal'] = calculate_macd(df['Close'])
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Upper'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
        df['BB_Middle'] = calculate_sma(df['Close'], 20)
        
        # Volume indicators
        df['Volume_SMA'] = calculate_sma(df['Volume'], 20)
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Price momentum
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_Pct'] = df['Price_Change'] * 100
        
        # Volatility
        df['Volatility'] = df['Price_Change'].rolling(window=20).std()
        
        # Save the updated DataFrame with indicators
        output_file = os.path.join(DATA_PATH, "stock_data_features.parquet")
        df.to_parquet(output_file, compression="snappy")
        logging.info(f"Technical indicators saved to {output_file}")
        
        # Print some statistics
        latest = df.iloc[-1]
        logging.info("\nLatest Technical Indicators:")
        logging.info(f"RSI: {latest['RSI'].iloc[0]:.2f}")
        logging.info(f"MACD: {latest['MACD'].iloc[0]:.2f}")
        logging.info(f"Volume Change: {latest['Volume_Change'].iloc[0]*100:.2f}%")
        logging.info(f"Volatility: {latest['Volatility'].iloc[0]*100:.2f}%")
        
    except Exception as e:
        logging.error(f"Error calculating indicators: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        setup_directories()
        calculate_indicators()
    except KeyboardInterrupt:
        logging.info("Technical indicator calculation stopped by user")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")