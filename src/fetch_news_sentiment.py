import feedparser
import pandas as pd
import os
from transformers import pipeline

# Define data path
DATA_PATH = "../data/"
os.makedirs(DATA_PATH, exist_ok=True)


def fetch_news(stock="AAPL", num_articles=10):
    """
    Fetch latest news related to the stock from Google News RSS.

    Parameters:
    - stock (str): Stock ticker (e.g., "AAPL")
    - num_articles (int): Number of latest articles to fetch

    Returns:
    - DataFrame containing news headlines and published dates.
    """
    feed_url = f"https://news.google.com/rss/search?q={stock}+stock"
    feed = feedparser.parse(feed_url)

    headlines = [{"headline": entry.title, "published": entry.published} for entry in feed.entries[:num_articles]]

    df = pd.DataFrame(headlines)

    if "headline" not in df.columns:
        print("❌ Error: No 'headline' column found in fetched news!")
        return None

    return df


def analyze_news_sentiment(df):
    """
    Perform sentiment analysis on news headlines using FinBERT.

    Parameters:
    - df (DataFrame): DataFrame containing news headlines.

    Returns:
    - DataFrame with sentiment scores.
    """
    if df is None or df.empty:
        print("❌ No news data available for sentiment analysis.")
        return None

    # Load FinBERT model
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

    # Analyze sentiment
    sentiments = sentiment_pipeline(df["headline"].tolist())

    # Convert results to DataFrame
    df["sentiment"] = [s["label"] for s in sentiments]
    df["confidence"] = [s["score"] for s in sentiments]

    # Assign numeric sentiment score
    df["sentiment_score"] = df["sentiment"].map({"positive": 1, "neutral": 0, "negative": -1})
    df["weighted_sentiment"] = df["sentiment_score"] * df["confidence"]

    return df


def save_to_parquet(df, filename="news_sentiment.parquet"):
    """
    Save the processed DataFrame to a Parquet file.

    Parameters:
    - df (DataFrame): DataFrame to save.
    - filename (str): Name of the Parquet file.
    """
    if df is None or df.empty:
        print("❌ No data to save.")
        return

    file_path = os.path.join(DATA_PATH, filename)
    df.to_parquet(file_path, compression="snappy")
    print(f"✅ News sentiment analysis saved to {file_path}")


def main():
    """Main function to fetch news, analyze sentiment, and save results."""
    stock_symbol = "AAPL"  # Change to the stock ticker you want

    print(f"🔍 Fetching news for {stock_symbol}...")
    news_df = fetch_news(stock_symbol)

    print(f"🤖 Performing sentiment analysis...")
    sentiment_df = analyze_news_sentiment(news_df)

    print(f"💾 Saving results to Parquet file...")
    save_to_parquet(sentiment_df)

    print(f"✅ Completed processing news sentiment for {stock_symbol}!")


if __name__ == "__main__":
    main()