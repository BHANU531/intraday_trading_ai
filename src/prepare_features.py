import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime, timedelta

# Constants
DATA_PATH = "../data/"
LOOKBACK_WINDOW = 12  # Number of 5-min intervals to look back
PRICE_CHANGE_THRESHOLD = 0.001  # 0.1% price change threshold for buy/sell signals

def load_and_prepare_stock_data():
    """Load and prepare the main stock price data."""
    stock_file = os.path.join(DATA_PATH, "AAPL_stock_data.parquet")
    if not os.path.exists(stock_file):
        raise FileNotFoundError(f"Stock data not found at {stock_file}")
    
    df = pd.read_parquet(stock_file)
    
    # Calculate returns
    df['returns'] = df['Close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=LOOKBACK_WINDOW).std()
    
    # Calculate price momentum
    df['price_momentum'] = df['Close'].pct_change(periods=LOOKBACK_WINDOW)
    
    # Calculate high-low range
    df['high_low_range'] = (df['High'] - df['Low']) / df['Close']
    
    return df

def load_technical_indicators():
    """Load technical indicators."""
    indicators_file = os.path.join(DATA_PATH, "stock_data_features.parquet")
    if not os.path.exists(indicators_file):
        return None
    
    return pd.read_parquet(indicators_file)

def load_news_sentiment():
    """Load and aggregate news sentiment data."""
    news_file = os.path.join(DATA_PATH, "news_sentiment.parquet")
    if not os.path.exists(news_file):
        return None
    
    news_df = pd.read_parquet(news_file)
    
    # Aggregate sentiment by timestamp
    news_df['timestamp'] = pd.to_datetime(news_df['published'])
    news_df.set_index('timestamp', inplace=True)
    
    # Resample to 5-minute intervals
    agg_sentiment = news_df.resample('5T').agg({
        'weighted_sentiment': 'mean',
        'confidence': 'mean'
    }).fillna(method='ffill')
    
    return agg_sentiment

def load_twitter_sentiment():
    """Load and aggregate Twitter sentiment data."""
    twitter_file = os.path.join(DATA_PATH, "trading_tweets_sentiment.parquet")
    if not os.path.exists(twitter_file):
        return None
    
    twitter_df = pd.read_parquet(twitter_file)
    
    # Convert sentiment to numeric
    sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    twitter_df['sentiment_value'] = twitter_df['sentiment'].map(sentiment_map)
    
    # Calculate engagement-weighted sentiment
    twitter_df['engagement'] = twitter_df['likes'] + twitter_df['retweets'] * 2 + twitter_df['replies'] * 3
    twitter_df['weighted_sentiment'] = twitter_df['sentiment_value'] * twitter_df['sentiment_score'] * \
                                     np.log1p(twitter_df['engagement'])
    
    # Aggregate by timestamp
    twitter_df['timestamp'] = pd.to_datetime(twitter_df['date'])
    twitter_df.set_index('timestamp', inplace=True)
    
    # Resample to 5-minute intervals
    agg_twitter = twitter_df.resample('5T').agg({
        'weighted_sentiment': 'mean',
        'engagement': 'sum'
    }).fillna(method='ffill')
    
    return agg_twitter

def create_target_variable(df):
    """Create buy/sell target variable based on future returns."""
    # Calculate future returns (next 5-minute return)
    df['future_returns'] = df['Close'].pct_change().shift(-1)
    
    # Create target variable
    df['target'] = 0  # Hold
    df.loc[df['future_returns'] > PRICE_CHANGE_THRESHOLD, 'target'] = 1  # Buy
    df.loc[df['future_returns'] < -PRICE_CHANGE_THRESHOLD, 'target'] = -1  # Sell
    
    return df

def prepare_features():
    """Prepare and combine all features for XGBoost."""
    # Load main stock data
    print("Loading stock data...")
    stock_df = load_and_prepare_stock_data()
    
    # Load technical indicators
    print("Loading technical indicators...")
    tech_df = load_technical_indicators()
    if tech_df is not None:
        stock_df = stock_df.join(tech_df)
    
    # Load news sentiment
    print("Loading news sentiment...")
    news_df = load_news_sentiment()
    if news_df is not None:
        stock_df = stock_df.join(news_df, how='left')
    
    # Load Twitter sentiment
    print("Loading Twitter sentiment...")
    twitter_df = load_twitter_sentiment()
    if twitter_df is not None:
        stock_df = stock_df.join(twitter_df, how='left')
    
    # Create target variable
    stock_df = create_target_variable(stock_df)
    
    # Handle missing values
    stock_df = stock_df.fillna(method='ffill').fillna(method='bfill')
    
    # Scale features
    scaler = StandardScaler()
    feature_columns = [
        'returns', 'volatility', 'price_momentum', 'high_low_range',
        'Volume_Change', 'SMA_10', 'RSI', 'MACD'
    ]
    
    if 'weighted_sentiment_x' in stock_df.columns:  # News sentiment
        feature_columns.extend(['weighted_sentiment_x', 'confidence'])
    
    if 'weighted_sentiment_y' in stock_df.columns:  # Twitter sentiment
        feature_columns.extend(['weighted_sentiment_y', 'engagement'])
    
    stock_df[feature_columns] = scaler.fit_transform(stock_df[feature_columns])
    
    # Save processed features
    output_file = os.path.join(DATA_PATH, "ml_features.parquet")
    stock_df.to_parquet(output_file, compression="snappy")
    print(f"Features saved to {output_file}")
    
    return stock_df

if __name__ == "__main__":
    prepare_features() 