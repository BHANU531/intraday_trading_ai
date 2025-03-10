import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime, timedelta
import yfinance as yf

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

def calculate_market_regime(data, window=20):
    """Calculate market regime indicators."""
    # Volatility regime
    returns = data['returns']
    rolling_vol = returns.rolling(window=window).std()
    vol_percentile = rolling_vol.rolling(window=window*5).rank(pct=True)
    
    # Trend regime
    sma_short = data['Close'].rolling(window=window//2).mean()
    sma_long = data['Close'].rolling(window=window).mean()
    trend_strength = (sma_short - sma_long) / sma_long
    
    # Volume regime
    volume_ma = data['Volume'].rolling(window=window).mean()
    volume_percentile = data['Volume'].rolling(window=window*5).rank(pct=True)
    
    return pd.DataFrame({
        'volatility_regime': vol_percentile,
        'trend_regime': trend_strength,
        'volume_regime': volume_percentile
    }, index=data.index)

def calculate_cross_asset_correlations(stock_df):
    """Calculate correlations with other market indicators."""
    try:
        # Download SPY data for market correlation
        spy_data = yf.download('SPY', 
                             start=(stock_df.index[0] - timedelta(days=1)),
                             end=stock_df.index[-1],
                             interval='5m')
        
        # Download QQQ data for tech sector correlation
        qqq_data = yf.download('QQQ',
                             start=(stock_df.index[0] - timedelta(days=1)),
                             end=stock_df.index[-1],
                             interval='5m')
        
        # Calculate correlations
        correlations = pd.DataFrame(index=stock_df.index)
        
        # Rolling correlation with market (SPY)
        correlations['market_correlation'] = stock_df['returns'].rolling(window=12).corr(
            spy_data['Close'].pct_change()
        )
        
        # Rolling correlation with tech sector (QQQ)
        correlations['sector_correlation'] = stock_df['returns'].rolling(window=12).corr(
            qqq_data['Close'].pct_change()
        )
        
        # Calculate relative strength
        correlations['market_rs'] = (
            stock_df['Close'] / stock_df['Close'].shift(12)
        ) / (
            spy_data['Close'] / spy_data['Close'].shift(12)
        )
        
        correlations['sector_rs'] = (
            stock_df['Close'] / stock_df['Close'].shift(12)
        ) / (
            qqq_data['Close'] / qqq_data['Close'].shift(12)
        )
        
        return correlations.fillna(method='ffill')
    
    except Exception as e:
        print(f"Error calculating cross-asset correlations: {e}")
        return pd.DataFrame()

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
    
    # Calculate market regime features
    print("Calculating market regime features...")
    regime_features = calculate_market_regime(stock_df)
    stock_df = stock_df.join(regime_features)
    
    # Calculate cross-asset correlations
    print("Calculating cross-asset correlations...")
    correlation_features = calculate_cross_asset_correlations(stock_df)
    if not correlation_features.empty:
        stock_df = stock_df.join(correlation_features)
    
    # Load news sentiment
    print("Loading news sentiment...")
    news_df = load_news_sentiment()
    if news_df is not None:
        stock_df = stock_df.join(news_df, how='left')
    
    # Load Twitter sentiment with enhanced features
    print("Loading Twitter sentiment...")
    twitter_df = load_twitter_sentiment()
    if twitter_df is not None:
        stock_df = stock_df.join(twitter_df, how='left')
    
    # Create target variable
    stock_df = create_target_variable(stock_df)
    
    # Handle missing values with more sophisticated approach
    stock_df = handle_missing_values(stock_df)
    
    # Scale features
    scaler = StandardScaler()
    feature_columns = [
        # Price and volume features
        'returns', 'volatility', 'price_momentum', 'high_low_range',
        'Volume_Change', 'Volume_SMA', 'OBV',
        
        # Technical indicators
        'SMA_10', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'ATR', 'MFI', 'ROC',
        
        # Market regime features
        'volatility_regime', 'trend_regime', 'volume_regime',
        
        # Cross-asset correlations
        'market_correlation', 'sector_correlation', 'market_rs', 'sector_rs'
    ]
    
    # Add sentiment features if available
    if 'weighted_sentiment' in stock_df.columns:
        feature_columns.extend(['weighted_sentiment', 'relevance_score', 'engagement_score'])
    
    if 'final_sentiment' in stock_df.columns:
        feature_columns.append('final_sentiment')
    
    # Scale features and handle any missing columns
    available_features = [col for col in feature_columns if col in stock_df.columns]
    stock_df[available_features] = scaler.fit_transform(stock_df[available_features])
    
    # Save processed features
    output_file = os.path.join(DATA_PATH, "ml_features.parquet")
    stock_df.to_parquet(output_file, compression="snappy")
    print(f"Features saved to {output_file}")
    
    # Save feature scaler for prediction
    scaler_file = os.path.join(DATA_PATH, "feature_scaler.pkl")
    pd.to_pickle(scaler, scaler_file)
    print(f"Feature scaler saved to {scaler_file}")
    
    return stock_df

def handle_missing_values(df):
    """Handle missing values with sophisticated approach."""
    # Forward fill for most recent values
    df = df.fillna(method='ffill')
    
    # For any remaining NAs, use rolling median
    for column in df.columns:
        if df[column].isnull().any():
            df[column] = df[column].fillna(
                df[column].rolling(window=12, min_periods=1, center=True).median()
            )
    
    return df

if __name__ == "__main__":
    prepare_features() 