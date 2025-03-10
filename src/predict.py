import pandas as pd
import numpy as np
import joblib
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
        # Load model, scaler and feature columns using joblib
        model = joblib.load(os.path.join(MODEL_PATH, "trading_model.joblib"))
        scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.joblib"))
        feature_cols = joblib.load(os.path.join(MODEL_PATH, "feature_columns.joblib"))
        
        logging.info("Model and scaler loaded successfully")
        return model, scaler, feature_cols
    
    except Exception as e:
        logging.error(f"Error loading model and scaler: {e}")
        return None, None, None

def load_latest_data():
    """Load and prepare the latest data for prediction."""
    try:
        # Load latest stock data with features
        stock_df = pd.read_parquet(os.path.join(DATA_PATH, "AAPL_stock_data.parquet"))
        stock_df.index = pd.to_datetime(stock_df.index)
        
        # Load latest sentiment data
        try:
            news_df = pd.read_parquet(os.path.join(DATA_PATH, "news_sentiment.parquet"))
            news_df['timestamp'] = pd.to_datetime(news_df['timestamp'])
            latest_news = news_df.iloc[-1] if not news_df.empty else None
        except Exception as e:
            logging.warning(f"Could not load news sentiment: {e}")
            latest_news = None
        
        try:
            twitter_df = pd.read_parquet(os.path.join(DATA_PATH, "news_sentiment.parquet"))  # Using same sentiment data for now
            twitter_df['timestamp'] = pd.to_datetime(twitter_df['timestamp'])
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
        
        # Add sentiment features if available
        if latest_news is not None:
            features['news_sentiment'] = latest_news.get('sentiment_score', 0)
        else:
            features['news_sentiment'] = 0
            
        if latest_twitter is not None:
            features['twitter_sentiment'] = latest_twitter.get('sentiment_score', 0)
        else:
            features['twitter_sentiment'] = 0
        
        # Fill missing values and ensure float type
        for col in features.columns:
            features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0.0)
        
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
        # Select and order features according to training
        X = features[feature_cols].copy()
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        pred_proba = model.predict_proba(X_scaled)[0]
        prediction = model.predict(X_scaled)[0]
        
        return prediction, pred_proba
    
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        return None, None

def save_prediction(prediction_result):
    """Save the prediction result to a CSV file."""
    try:
        file_path = os.path.join(DATA_PATH, "predictions.log")
        
        # Ensure the data directory exists
        os.makedirs(DATA_PATH, exist_ok=True)
        
        # Define columns
        columns = [
            'timestamp', 'action', 'confidence', 'price',
            'rsi', 'macd', 'news_sentiment', 'twitter_sentiment'
        ]
        
        # Create a new DataFrame with only the required columns
        data = {
            'timestamp': [prediction_result['timestamp'].strftime('%Y-%m-%d %H:%M:%S') 
                         if isinstance(prediction_result['timestamp'], (pd.Timestamp, datetime)) 
                         else prediction_result['timestamp']],
            'action': [str(prediction_result['action'])],
            'confidence': [float(prediction_result['confidence'])],
            'price': [float(prediction_result['price'])],
            'rsi': [float(prediction_result['rsi'])],
            'macd': [float(prediction_result['macd'])],
            'news_sentiment': [float(prediction_result['news_sentiment'])],
            'twitter_sentiment': [float(prediction_result['twitter_sentiment'])]
        }
        
        df = pd.DataFrame(data, columns=columns)  # Ensure column order
        
        # Write to file
        write_header = not (os.path.exists(file_path) and os.path.getsize(file_path) > 0)
        df.to_csv(file_path, mode='a' if not write_header else 'w', header=write_header, index=False)
        
        logging.info(f"Prediction saved to {file_path}")
    
    except Exception as e:
        logging.error(f"Error saving prediction: {e}")
        logging.error(f"Prediction data: {prediction_result}")
        raise

def format_prediction(prediction, pred_proba, features):
    """Format the prediction results."""
    try:
        action = "BUY" if prediction == 1 else "SELL"
        confidence = float(pred_proba[1] if prediction == 1 else pred_proba[0])
        
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
        
        # Save prediction to CSV
        save_prediction(result)
        
        return result
    
    except Exception as e:
        logging.error(f"Error formatting prediction: {e}")
        return None

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