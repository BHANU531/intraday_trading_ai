import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import os
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/model_training.log'),
        logging.StreamHandler()
    ]
)

# Constants
DATA_PATH = "data/"
MODEL_PATH = "models/"
os.makedirs(MODEL_PATH, exist_ok=True)

def load_stock_features():
    """Load stock data with technical indicators."""
    try:
        df = pd.read_parquet(os.path.join(DATA_PATH, "stock_data_features.parquet"))
        logging.info(f"Loaded {len(df)} records of stock data with technical indicators")
        return df
    except Exception as e:
        logging.error(f"Error loading stock features: {e}")
        return None

def load_sentiment_data():
    """Load and aggregate sentiment data from news and Twitter."""
    try:
        news_hourly = None
        twitter_hourly = None
        
        # Load news sentiment
        try:
            news_df = pd.read_parquet(os.path.join(DATA_PATH, "news_sentiment.parquet"))
            if not news_df.empty:
                news_df['timestamp'] = pd.to_datetime(news_df['date'], errors='coerce')
                news_df = news_df.dropna(subset=['timestamp'])
                
                # Aggregate sentiments by hour
                news_hourly = news_df.resample('1h', on='timestamp').agg({
                    'sentiment': lambda x: x.value_counts().index[0] if len(x) > 0 else 'neutral',
                    'sentiment_score': 'mean'
                }).reset_index()
                
                # Convert sentiment to numeric
                sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
                news_hourly['sentiment_numeric'] = news_hourly['sentiment'].map(sentiment_map)
        except Exception as e:
            logging.warning(f"Error loading news sentiment data: {e}")
        
        # Load Twitter sentiment
        try:
            twitter_df = pd.read_parquet(os.path.join(DATA_PATH, "trading_tweets_sentiment.parquet"))
            if not twitter_df.empty:
                twitter_df['timestamp'] = pd.to_datetime(twitter_df['date'], errors='coerce')
                twitter_df = twitter_df.dropna(subset=['timestamp'])
                
                # Aggregate sentiments by hour
                twitter_hourly = twitter_df.resample('1h', on='timestamp').agg({
                    'sentiment': lambda x: x.value_counts().index[0] if len(x) > 0 else 'neutral',
                    'sentiment_score': 'mean'
                }).reset_index()
                
                # Convert sentiment to numeric
                twitter_hourly['sentiment_numeric'] = twitter_hourly['sentiment'].map(sentiment_map)
        except Exception as e:
            logging.warning(f"Error loading Twitter sentiment data: {e}")
        
        if news_hourly is not None or twitter_hourly is not None:
            logging.info(f"Processed {len(news_df) if news_df is not None else 0} news articles and {len(twitter_df) if twitter_df is not None else 0} tweets")
        else:
            logging.warning("No sentiment data loaded")
        
        return news_hourly, twitter_hourly
    
    except Exception as e:
        logging.error(f"Error loading sentiment data: {e}")
        return None, None

def load_financial_data():
    """Load financial data from balance sheets and ratios."""
    try:
        # Load balance sheet data
        balance_df = pd.read_parquet(os.path.join(DATA_PATH, "balance_sheets.parquet"))
        
        # Load financial ratios
        ratios_df = pd.read_parquet(os.path.join(DATA_PATH, "financial_ratios.parquet"))
        
        # Load earnings data
        earnings_df = pd.read_parquet(os.path.join(DATA_PATH, "earnings.parquet"))
        
        logging.info("Loaded financial data successfully")
        return balance_df, ratios_df, earnings_df
    
    except Exception as e:
        logging.error(f"Error loading financial data: {e}")
        return None, None, None

def prepare_features(stock_df, news_df, twitter_df, balance_df, ratios_df, earnings_df):
    """Prepare features by combining all data sources."""
    try:
        # Start with stock data and technical indicators
        features_df = stock_df.copy()
        
        # Convert index to datetime and extract time features
        features_df.index = pd.to_datetime(features_df.index)
        features_df['hour'] = features_df.index.hour
        features_df['minute'] = features_df.index.minute
        features_df['day_of_week'] = features_df.index.dayofweek
        
        # Flatten multi-index columns
        if isinstance(features_df.columns, pd.MultiIndex):
            features_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in features_df.columns]
        
        # Initialize sentiment columns with zeros
        features_df['news_sentiment'] = 0.0
        features_df['news_sentiment_score'] = 0.0
        features_df['twitter_sentiment'] = 0.0
        features_df['twitter_sentiment_score'] = 0.0
        
        # Add sentiment features if available
        if news_df is not None and not news_df.empty:
            news_df.set_index('timestamp', inplace=True)
            # Resample news sentiment to match stock data frequency
            resampled_news = news_df.resample('5min').agg({
                'sentiment_numeric': 'mean',
                'sentiment_score': 'mean'
            }).fillna(method='ffill')
            features_df['news_sentiment'] = resampled_news['sentiment_numeric']
            features_df['news_sentiment_score'] = resampled_news['sentiment_score']
        
        if twitter_df is not None and not twitter_df.empty:
            twitter_df.set_index('timestamp', inplace=True)
            # Resample twitter sentiment to match stock data frequency
            resampled_twitter = twitter_df.resample('5min').agg({
                'sentiment_numeric': 'mean',
                'sentiment_score': 'mean'
            }).fillna(method='ffill')
            features_df['twitter_sentiment'] = resampled_twitter['sentiment_numeric']
            features_df['twitter_sentiment_score'] = resampled_twitter['sentiment_score']
        
        # Add financial features (latest available)
        if balance_df is not None and not balance_df.empty:
            latest_balance = balance_df.iloc[-1]
            for col in balance_df.columns:
                features_df[f'balance_{col}'] = latest_balance[col]
        
        if ratios_df is not None and not ratios_df.empty:
            for col in ratios_df.columns:
                if col != 'Stock':  # Skip the stock symbol column
                    features_df[f'ratio_{col}'] = ratios_df[col].iloc[0]
        
        if earnings_df is not None and not earnings_df.empty:
            latest_earnings = earnings_df.iloc[-1]
            features_df['latest_revenue'] = latest_earnings['Revenue']
            features_df['latest_earnings'] = latest_earnings['Earnings']
            
            # Calculate growth rates if possible
            if len(earnings_df) > 1:
                prev_earnings = earnings_df.iloc[-2]
                features_df['revenue_growth'] = (latest_earnings['Revenue'] - prev_earnings['Revenue']) / prev_earnings['Revenue']
                features_df['earnings_growth'] = (latest_earnings['Earnings'] - prev_earnings['Earnings']) / prev_earnings['Earnings']
        
        # Calculate target variable (future returns)
        features_df['future_return'] = features_df['Close_AAPL'].shift(-1) / features_df['Close_AAPL'] - 1
        features_df['target'] = (features_df['future_return'] > 0).astype(int)
        
        # Drop unnecessary columns and NaN values
        features_df = features_df.drop(['future_return', 'fetch_time'], axis=1, errors='ignore')
        features_df = features_df.fillna(0)  # Fill remaining NaN values with 0
        
        # Convert index to feature columns
        features_df = features_df.reset_index()
        features_df = features_df.rename(columns={'index': 'timestamp'})
        
        logging.info(f"Prepared {len(features_df)} samples with {len(features_df.columns)} features")
        
        # Display feature names for debugging
        logging.info("\nFeature columns:")
        for col in features_df.columns:
            logging.info(f"- {col}: {features_df[col].dtype}")
        
        return features_df
    
    except Exception as e:
        logging.error(f"Error preparing features: {e}")
        return None

def train_model(features_df):
    """Train XGBoost model with the prepared features."""
    try:
        # Drop the timestamp column and target from features
        feature_cols = [col for col in features_df.columns if col not in ['timestamp', 'Datetime', 'target']]
        X = features_df[feature_cols]
        y = features_df['target']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train XGBoost model
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            eval_metric='auc',
            use_label_encoder=False,
            early_stopping_rounds=10,
            random_state=42
        )
        
        # Create evaluation set
        eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]
        
        # Train the model
        model.fit(
            X_train_scaled,
            y_train,
            eval_set=eval_set,
            verbose=True
        )
        
        # Save the model and scaler
        model.save_model(os.path.join(MODEL_PATH, "trading_model.json"))
        pd.to_pickle(scaler, os.path.join(MODEL_PATH, "feature_scaler.pkl"))
        pd.to_pickle(feature_cols, os.path.join(MODEL_PATH, "feature_columns.pkl"))
        
        # Calculate and log metrics
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        logging.info("\nModel Performance:")
        logging.info(f"Train accuracy: {train_score:.4f}")
        logging.info(f"Test accuracy: {test_score:.4f}")
        
        # Log feature importance
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logging.info("\nTop 10 Most Important Features:")
        for _, row in importance_df.head(10).iterrows():
            logging.info(f"{row['feature']}: {row['importance']:.4f}")
        
        return model, scaler
    
    except Exception as e:
        logging.error(f"Error training model: {e}")
        return None, None

def main():
    """Main function to load data, prepare features, and train the model."""
    try:
        # Load all data sources
        stock_df = load_stock_features()
        news_df, twitter_df = load_sentiment_data()
        balance_df, ratios_df, earnings_df = load_financial_data()
        
        if stock_df is None:
            raise ValueError("Could not load stock data")
        
        # Prepare features
        features_df = prepare_features(stock_df, news_df, twitter_df, balance_df, ratios_df, earnings_df)
        
        if features_df is None:
            raise ValueError("Could not prepare features")
        
        # Train model
        model, scaler = train_model(features_df)
        
        if model is not None and scaler is not None:
            logging.info("\n✅ Model training completed successfully!")
            logging.info(f"Model saved to {os.path.join(MODEL_PATH, 'trading_model.json')}")
            logging.info(f"Scaler saved to {os.path.join(MODEL_PATH, 'feature_scaler.pkl')}")
        else:
            logging.error("Model training failed")
    
    except Exception as e:
        logging.error(f"Error in main: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("\nModel training interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error: {e}") 