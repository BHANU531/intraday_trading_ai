import pandas as pd
import numpy as np
import xgboost as xgb
import logging
import os
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/predictions.log'),
        logging.StreamHandler()
    ]
)

# Constants
DATA_PATH = "data/"
MODEL_PATH = "models/"
STOCK_SYMBOL = "AAPL"

def load_model_and_scaler():
    """Load the trained model and feature scaler."""
    try:
        # Load model
        model = xgb.XGBClassifier()
        model.load_model(os.path.join(MODEL_PATH, "trading_model.json"))
        
        # Load scaler and feature columns
        scaler = pd.read_pickle(os.path.join(MODEL_PATH, "feature_scaler.pkl"))
        feature_cols = pd.read_pickle(os.path.join(MODEL_PATH, "feature_columns.pkl"))
        
        logging.info("Model and scaler loaded successfully")
        return model, scaler, feature_cols
    
    except Exception as e:
        logging.error(f"Error loading model and scaler: {e}")
        return None, None, None

def load_latest_data():
    """Load and prepare the latest data for prediction."""
    try:
        # Load latest stock data with features
        stock_df = pd.read_parquet(os.path.join(DATA_PATH, "stock_data_features.parquet"))
        stock_df.index = pd.to_datetime(stock_df.index)
        
        # Load latest sentiment data
        try:
            news_df = pd.read_parquet(os.path.join(DATA_PATH, "news_sentiment.parquet"))
            news_df['timestamp'] = pd.to_datetime(news_df['date'])
            latest_news = news_df.iloc[-1] if not news_df.empty else None
        except Exception as e:
            logging.warning(f"Could not load news sentiment: {e}")
            latest_news = None
        
        try:
            twitter_df = pd.read_parquet(os.path.join(DATA_PATH, "trading_tweets_sentiment.parquet"))
            twitter_df['timestamp'] = pd.to_datetime(twitter_df['date'])
            latest_twitter = twitter_df.iloc[-1] if not twitter_df.empty else None
        except Exception as e:
            logging.warning(f"Could not load Twitter sentiment: {e}")
            latest_twitter = None
        
        # Load latest financial data
        try:
            balance_df = pd.read_parquet(os.path.join(DATA_PATH, "balance_sheets.parquet"))
            ratios_df = pd.read_parquet(os.path.join(DATA_PATH, "financial_ratios.parquet"))
            earnings_df = pd.read_parquet(os.path.join(DATA_PATH, "earnings.parquet"))
            
            latest_balance = balance_df.iloc[-1] if not balance_df.empty else None
            latest_ratios = ratios_df.iloc[0] if not ratios_df.empty else None
            latest_earnings = earnings_df.iloc[-1] if not earnings_df.empty else None
        except Exception as e:
            logging.warning(f"Could not load financial data: {e}")
            latest_balance = None
            latest_ratios = None
            latest_earnings = None
        
        return stock_df, latest_news, latest_twitter, latest_balance, latest_ratios, latest_earnings
    
    except Exception as e:
        logging.error(f"Error loading latest data: {e}")
        return None, None, None, None, None, None

def prepare_prediction_features(stock_df, latest_news, latest_twitter, latest_balance, latest_ratios, latest_earnings):
    """Prepare features for the latest data point."""
    try:
        # Get the latest stock data point
        features = stock_df.iloc[-1].copy()
        
        # Convert to DataFrame and reset index
        features = pd.DataFrame(features).T
        features.index = pd.to_datetime(features.index)
        
        # Flatten multi-index columns if needed
        if isinstance(features.columns, pd.MultiIndex):
            features.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in features.columns]
        
        # Add time features
        features['hour'] = features.index.hour
        features['minute'] = features.index.minute
        features['day_of_week'] = features.index.dayofweek
        
        # Add sentiment features
        sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
        
        # Add news sentiment
        if latest_news is not None:
            sentiment = latest_news['sentiment'].lower() if 'sentiment' in latest_news else 'neutral'
            features['news_sentiment'] = sentiment_map.get(sentiment, 0)
            features['news_sentiment_score'] = latest_news.get('sentiment_score', 0)
        else:
            features['news_sentiment'] = 0
            features['news_sentiment_score'] = 0
        
        # Add Twitter sentiment
        if latest_twitter is not None:
            sentiment = latest_twitter['sentiment'].lower() if 'sentiment' in latest_twitter else 'neutral'
            features['twitter_sentiment'] = sentiment_map.get(sentiment, 0)
            features['twitter_sentiment_score'] = latest_twitter.get('sentiment_score', 0)
        else:
            features['twitter_sentiment'] = 0
            features['twitter_sentiment_score'] = 0
        
        # Add financial features
        if latest_balance is not None:
            for col in latest_balance.index:
                features[f'balance_{col}'] = latest_balance[col]
        
        if latest_ratios is not None:
            for col in latest_ratios.index:
                if col != 'Stock':
                    features[f'ratio_{col}'] = latest_ratios[col]
        
        if latest_earnings is not None:
            features['latest_revenue'] = latest_earnings['Revenue']
            features['latest_earnings'] = latest_earnings['Earnings']
            
            # Calculate growth rates if possible
            features['revenue_growth'] = 0
            features['earnings_growth'] = 0
        
        # Fill missing values
        features = features.fillna(0)
        
        # Log feature names for debugging
        logging.info("\nFeature columns:")
        for col in features.columns:
            logging.info(f"- {col}: {features[col].iloc[0]}")
        
        return features
    
    except Exception as e:
        logging.error(f"Error preparing prediction features: {e}")
        return None

def make_prediction(model, scaler, features, feature_cols):
    """Make prediction using the trained model."""
    try:
        # Ensure all required features are present
        missing_cols = set(feature_cols) - set(features.columns)
        if missing_cols:
            logging.warning(f"Missing features: {missing_cols}")
            for col in missing_cols:
                features[col] = 0
        
        # Select and order features according to training
        X = features[feature_cols]
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        pred_proba = model.predict_proba(X_scaled)[0]
        prediction = model.predict(X_scaled)[0]
        
        return prediction, pred_proba
    
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        return None, None

def format_prediction(prediction, pred_proba, features):
    """Format the prediction results."""
    try:
        action = "BUY" if prediction == 1 else "SELL"
        confidence = pred_proba[1] if prediction == 1 else pred_proba[0]
        
        result = {
            'timestamp': features.index[0],
            'action': action,
            'confidence': confidence,
            'price': float(features['Close_AAPL'].iloc[0]),
            'rsi': float(features['RSI'].iloc[0]),
            'macd': float(features['MACD'].iloc[0]),
            'news_sentiment': float(features['news_sentiment'].iloc[0]),
            'twitter_sentiment': float(features['twitter_sentiment'].iloc[0])
        }
        
        return result
    
    except Exception as e:
        logging.error(f"Error formatting prediction: {e}")
        return None

def save_prediction(prediction_result):
    """Save the prediction result to a CSV file."""
    try:
        file_path = os.path.join(DATA_PATH, "predictions.csv")
        df = pd.DataFrame([prediction_result])
        
        # Append to existing file or create new one
        if os.path.exists(file_path):
            df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            df.to_csv(file_path, index=False)
        
        logging.info(f"Prediction saved to {file_path}")
    
    except Exception as e:
        logging.error(f"Error saving prediction: {e}")

def main():
    """Main function to make predictions."""
    try:
        # Load model and scaler
        model, scaler, feature_cols = load_model_and_scaler()
        if model is None or scaler is None or feature_cols is None:
            raise ValueError("Could not load model, scaler, or feature columns")
        
        # Load latest data
        stock_df, latest_news, latest_twitter, latest_balance, latest_ratios, latest_earnings = load_latest_data()
        if stock_df is None:
            raise ValueError("Could not load latest data")
        
        # Prepare features
        features = prepare_prediction_features(
            stock_df, latest_news, latest_twitter,
            latest_balance, latest_ratios, latest_earnings
        )
        if features is None:
            raise ValueError("Could not prepare features")
        
        # Make prediction
        prediction, pred_proba = make_prediction(model, scaler, features, feature_cols)
        if prediction is None or pred_proba is None:
            raise ValueError("Could not make prediction")
        
        # Format results
        result = format_prediction(prediction, pred_proba, features)
        if result is None:
            raise ValueError("Could not format prediction results")
        
        # Log prediction
        logging.info("\n🔮 Prediction Results:")
        logging.info(f"Timestamp: {result['timestamp']}")
        logging.info(f"Action: {result['action']}")
        logging.info(f"Confidence: {result['confidence']:.2%}")
        logging.info(f"Current Price: ${result['price']:.2f}")
        logging.info(f"RSI: {result['rsi']:.2f}")
        logging.info(f"MACD: {result['macd']:.2f}")
        logging.info(f"News Sentiment: {result['news_sentiment']:.2f}")
        logging.info(f"Twitter Sentiment: {result['twitter_sentiment']:.2f}")
        
        # Save prediction
        save_prediction(result)
        
        return result
    
    except Exception as e:
        logging.error(f"Error in main: {e}")
        return None

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("\nPrediction interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error: {e}") 