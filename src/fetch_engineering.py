import pandas as pd
import os

# Constants
DATA_PATH = "../data/"
INPUT_FILE = f"{DATA_PATH}stock_data.parquet"
OUTPUT_FILE = f"{DATA_PATH}stock_data_features.parquet"

def calculate_indicators():
    """Calculate SMA, RSI, MACD, and Volume Change for the stock data."""
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Stock data not found at {INPUT_FILE}. Please run fetch_stock.py first.")
        return

    # Load stock data
    df = pd.read_parquet(INPUT_FILE)

    # Compute technical indicators
    df["SMA_10"] = df["Close"].rolling(window=10, min_periods=1).mean()
    df["RSI"] = calculate_rsi(df["Close"], window=14)
    df["MACD"] = calculate_macd(df["Close"])
    df["Volume_Change"] = df["Volume"].pct_change()

    # Drop rows with missing values (due to rolling calculations)
    df.dropna(inplace=True)

    # Save the updated DataFrame with indicators
    df.to_parquet(OUTPUT_FILE, compression="snappy")
    print(f"Technical indicators computed and saved to {OUTPUT_FILE}.")

def calculate_rsi(series, window=14):
    """Calculate Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, short_window=12, long_window=26):
    """Calculate Moving Average Convergence Divergence (MACD)."""
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    return short_ema - long_ema

if __name__ == "__main__":
    calculate_indicators()