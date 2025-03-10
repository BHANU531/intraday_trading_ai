import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load the stock data
stock_df = pd.read_parquet('data/AAPL_stock_data.parquet')

# Print column names for debugging
print("Available columns:", stock_df.columns.tolist())

# Create target variable (1 if price goes up, 0 if down)
stock_df['target'] = (stock_df['Close_AAPL'].shift(-1) > stock_df['Close_AAPL']).astype(int)

# Select features
feature_columns = [
    'Open_AAPL', 'High_AAPL', 'Low_AAPL', 'Close_AAPL', 'Volume_AAPL',
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'SMA_20', 'SMA_50'
]

# Prepare features and target
X = stock_df[feature_columns].dropna()
y = stock_df['target'].loc[X.index]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train a simple model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Save the model and scaler
joblib.dump(model, 'models/trading_model.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(feature_columns, 'models/feature_columns.joblib')

print("Model trained and saved successfully!") 