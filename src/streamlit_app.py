import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
from predict import load_model_and_scaler, load_latest_data, prepare_prediction_features, make_prediction, format_prediction
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
DATA_PATH = "data/"
MODEL_PATH = "models/"
STOCK_SYMBOL = "AAPL"

def create_candlestick_chart(stock_df):
    """Create a candlestick chart with technical indicators."""
    try:
        # Debug: Print DataFrame info
        logging.info("DataFrame columns: %s", stock_df.columns.tolist())
        logging.info("DataFrame head:\n%s", stock_df.head())
        
        # Check if required columns exist
        required_columns = ['Open_AAPL', 'High_AAPL', 'Low_AAPL', 'Close_AAPL']
        for col in required_columns:
            if col not in stock_df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame. Available columns: {stock_df.columns.tolist()}")
        
        fig = go.Figure()
        
        # Add candlestick
        fig.add_trace(go.Candlestick(
            x=stock_df.index,
            open=stock_df['Open_AAPL'],
            high=stock_df['High_AAPL'],
            low=stock_df['Low_AAPL'],
            close=stock_df['Close_AAPL'],
            name='OHLC'
        ))
        
        # Add Moving Averages if they exist
        if 'SMA_20' in stock_df.columns:
            fig.add_trace(go.Scatter(
                x=stock_df.index,
                y=stock_df['SMA_20'],
                line=dict(color='orange', width=1),
                name='20 SMA'
            ))
        
        if 'SMA_50' in stock_df.columns:
            fig.add_trace(go.Scatter(
                x=stock_df.index,
                y=stock_df['SMA_50'],
                line=dict(color='blue', width=1),
                name='50 SMA'
            ))
        
        # Update layout
        fig.update_layout(
            title=f'{STOCK_SYMBOL} Price Chart',
            yaxis_title='Price',
            xaxis_title='Date',
            template='plotly_dark',
            height=600
        )
        
        return fig
    except Exception as e:
        logging.error("Error in create_candlestick_chart: %s", str(e))
        raise

def create_technical_indicators_chart(stock_df):
    """Create a chart with technical indicators (RSI, MACD)."""
    fig = go.Figure()
    
    # Add RSI
    fig.add_trace(go.Scatter(
        x=stock_df.index,
        y=stock_df['RSI'],
        name='RSI',
        line=dict(color='purple', width=1)
    ))
    
    # Add MACD
    fig.add_trace(go.Scatter(
        x=stock_df.index,
        y=stock_df['MACD'],
        name='MACD',
        line=dict(color='blue', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=stock_df.index,
        y=stock_df['MACD_Signal'],
        name='Signal Line',
        line=dict(color='orange', width=1)
    ))
    
    # Add MACD histogram
    fig.add_trace(go.Bar(
        x=stock_df.index,
        y=stock_df['MACD_Histogram'],
        name='MACD Histogram',
        marker_color='gray'
    ))
    
    # Update layout with separate y-axes
    fig.update_layout(
        title='Technical Indicators',
        yaxis=dict(
            title='RSI',
            range=[0, 100],
            side='left'
        ),
        yaxis2=dict(
            title='MACD',
            overlaying='y',
            side='right'
        ),
        xaxis_title='Date',
        template='plotly_dark',
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update traces to use appropriate y-axes
    fig.update_traces(yaxis="y", selector=dict(name="RSI"))
    fig.update_traces(yaxis="y2", selector=dict(name=["MACD", "Signal Line", "MACD Histogram"]))
    
    return fig

def load_prediction_history():
    """Load prediction history from CSV file."""
    try:
        predictions_file = os.path.join(DATA_PATH, "predictions.log")
        
        # Define columns
        columns = [
            'timestamp', 'action', 'confidence', 'price',
            'rsi', 'macd', 'news_sentiment', 'twitter_sentiment'
        ]
        
        # Return empty DataFrame if file doesn't exist or is empty
        if not os.path.exists(predictions_file) or os.path.getsize(predictions_file) == 0:
            return pd.DataFrame(columns=columns)
        
        try:
            # First try reading the first line to check if it's a header
            with open(predictions_file, 'r') as f:
                first_line = f.readline().strip()
            
            # Determine if file has header
            has_header = 'timestamp' in first_line.lower()
            
            # Read CSV file
            history_df = pd.read_csv(
                predictions_file,
                header=0 if has_header else None,
                names=columns,
                dtype={
                    'action': str,
                    'confidence': float,
                    'price': float,
                    'rsi': float,
                    'macd': float,
                    'news_sentiment': float,
                    'twitter_sentiment': float
                }
            )
            
            # Convert timestamp to datetime
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            
            return history_df
            
        except Exception as e:
            logging.error(f"Error reading predictions file: {e}")
            # If there's an error, delete the corrupted file
            try:
                os.remove(predictions_file)
                logging.info("Deleted corrupted predictions file")
            except Exception as del_e:
                logging.error(f"Error deleting corrupted file: {del_e}")
            
            # Return empty DataFrame
            return pd.DataFrame(columns=columns)
    
    except Exception as e:
        logging.error(f"Error loading prediction history: {e}")
        return pd.DataFrame(columns=columns)

def main():
    st.set_page_config(
        page_title="Stock Trading AI",
        page_icon="📈",
        layout="wide"
    )
    
    # Title and description
    st.title("🤖 Stock Trading AI Dashboard")
    st.markdown("""
    This dashboard shows real-time trading predictions using machine learning.
    The model analyzes technical indicators, sentiment data, and financial metrics to make predictions.
    """)
    
    # Sidebar
    st.sidebar.title("Controls")
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 30, 300, 60)
    
    # Load model and data
    try:
        model, scaler, feature_cols = load_model_and_scaler()
        if model is None:
            st.error("Could not load the model. Please check if the model files exist.")
            return
        
        stock_df, latest_news, latest_twitter, latest_balance, latest_ratios, latest_earnings = load_latest_data()
        if stock_df is None:
            st.error("Could not load the latest data. Please check if the data files exist.")
            return
            
        # Debug: Print column names
        #st.write("Available columns:", stock_df.columns.tolist())
        
        # Create two columns for the main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Price chart
            st.subheader("Price Chart with Technical Indicators")
            try:
                fig_price = create_candlestick_chart(stock_df)
                st.plotly_chart(fig_price, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating price chart: {e}")
            
            # Technical indicators chart
            st.subheader("Technical Indicators")
            try:
                fig_tech = create_technical_indicators_chart(stock_df)
                st.plotly_chart(fig_tech, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating technical indicators chart: {e}")
        
        with col2:
            # Latest prediction
            st.subheader("Latest Prediction")
            
            # Prepare features and make prediction
            features = prepare_prediction_features(
                stock_df, latest_news, latest_twitter,
                latest_balance, latest_ratios, latest_earnings
            )
            
            if features is not None:
                prediction, pred_proba = make_prediction(model, scaler, features, feature_cols)
                if prediction is not None:
                    result = format_prediction(prediction, pred_proba, features)
                    
                    # Display prediction with styling
                    action_color = "green" if result['action'] == "BUY" else "red"
                    st.markdown(f"""
                    ### Signal: <span style='color:{action_color}'>{result['action']}</span>
                    """, unsafe_allow_html=True)
                    
                    st.metric("Confidence", f"{result['confidence']:.1%}")
                    st.metric("Current Price", f"${result['price']:.2f}")
                    
                    # Technical indicators
                    col_tech1, col_tech2 = st.columns(2)
                    with col_tech1:
                        st.metric("RSI", f"{result['rsi']:.1f}")
                    with col_tech2:
                        st.metric("MACD", f"{result['macd']:.3f}")
                    
                    # Sentiment indicators
                    st.subheader("Sentiment Analysis")
                    col_sent1, col_sent2 = st.columns(2)
                    with col_sent1:
                        st.metric("News Sentiment", f"{result['news_sentiment']:.1f}")
                    with col_sent2:
                        st.metric("Twitter Sentiment", f"{result['twitter_sentiment']:.1f}")
            
            # Prediction history
            st.subheader("Prediction History")
            history_df = load_prediction_history()
            if not history_df.empty:
                # Format the history dataframe
                display_df = history_df.copy()
                display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                display_df['confidence'] = display_df['confidence'].map('{:.1%}'.format)
                display_df['price'] = display_df['price'].map('${:.2f}'.format)
                display_df['rsi'] = display_df['rsi'].map('{:.1f}'.format)
                display_df['macd'] = display_df['macd'].map('{:.3f}'.format)
                st.dataframe(
                    display_df.tail(),
                    column_config={
                        "timestamp": "Time",
                        "action": "Signal",
                        "confidence": "Confidence",
                        "price": "Price",
                        "rsi": "RSI",
                        "macd": "MACD",
                        "news_sentiment": "News",
                        "twitter_sentiment": "Twitter"
                    },
                    use_container_width=True
                )
        
        # Auto-refresh using a placeholder
        if auto_refresh:
            placeholder = st.empty()
            with placeholder.container():
                st.markdown(f"Next refresh in {refresh_interval} seconds...")
                time.sleep(refresh_interval)
            st.rerun()
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.write("Error details:", str(e))
        # Log the full error traceback
        import traceback
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main() 