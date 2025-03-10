import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Download AAPL stock data
end_date = datetime.now()
start_date = end_date - timedelta(days=30)  # Get last 30 days of data
stock_df = yf.download('AAPL', start=start_date, end=end_date, interval='1h')

# Flatten column names
stock_df.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in stock_df.columns]

# Add technical indicators
def calculate_technical_indicators(df):
    # Calculate RSI
    delta = df['Close_AAPL'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    exp1 = df['Close_AAPL'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close_AAPL'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Calculate SMAs
    df['SMA_20'] = df['Close_AAPL'].rolling(window=20).mean()
    df['SMA_50'] = df['Close_AAPL'].rolling(window=50).mean()
    
    return df

# Process stock data
stock_df = calculate_technical_indicators(stock_df)

# Create sample sentiment data
dates = pd.date_range(start=start_date, end=end_date, freq='h')
sentiment_data = pd.DataFrame({
    'timestamp': dates,
    'sentiment_score': np.random.normal(0.2, 0.5, len(dates)),
    'sentiment_magnitude': np.random.uniform(0, 1, len(dates))
})

# Create sample financial data
balance_sheet = pd.DataFrame({
    'date': [end_date],
    'Total Assets': [3000000000000],
    'Total Liabilities': [2000000000000],
    'Stockholder Equity': [1000000000000],
    'Current Assets': [500000000000],
    'Current Liabilities': [300000000000],
    'Cash': [200000000000],
    'Short Term Investments': [100000000000],
    'Receivables': [50000000000],
    'Inventory': [30000000000],
    'Long Term Debt': [800000000000]
})

ratios = pd.DataFrame({
    'date': [end_date],
    'PE Ratio': [25.5],
    'Price to Book': [8.2],
    'Debt to Equity': [1.5],
    'Current Ratio': [1.8],
    'Quick Ratio': [1.2],
    'Return on Equity': [0.35],
    'Return on Assets': [0.15]
})

earnings = pd.DataFrame({
    'date': [end_date],
    'Revenue': [400000000000],
    'Net Income': [100000000000],
    'EPS': [6.5],
    'Revenue Growth': [0.15],
    'Net Income Growth': [0.12]
})

# Save all data files
stock_df.to_parquet('data/AAPL_stock_data.parquet')
sentiment_data.to_parquet('data/news_sentiment.parquet')
balance_sheet.to_parquet('data/balance_sheets.parquet')
ratios.to_parquet('data/financial_ratios.parquet')
earnings.to_parquet('data/earnings.parquet')

# Create empty predictions file
predictions_df = pd.DataFrame(columns=['timestamp', 'prediction', 'confidence', 'actual_movement'])
predictions_df.to_csv('data/predictions.log', index=False)

print("Data files generated successfully!") 