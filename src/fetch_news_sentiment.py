import feedparser
import pandas as pd
import os
from transformers import pipeline
import logging
from datetime import datetime
import urllib.parse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/news_sentiment.log'),
        logging.StreamHandler()
    ]
)

# Define data path
DATA_PATH = "data/"
os.makedirs(DATA_PATH, exist_ok=True)

def fetch_news(search_term, num_articles=50):
    """
    Fetch latest news related to the search term from Google News RSS.

    Parameters:
    - search_term (str): Search term (e.g., "AAPL")
    - num_articles (int): Number of latest articles to fetch

    Returns:
    - List of dictionaries containing news data.
    """
    logging.info(f"🔍 Fetching news for {search_term}...")
    
    # URL encode the search term
    encoded_term = urllib.parse.quote(f"{search_term} stock market")
    feed_url = f"https://news.google.com/rss/search?q={encoded_term}"
    
    try:
        feed = feedparser.parse(feed_url)
        
        if not feed.entries:
            logging.warning(f"No news found for {search_term}")
            return []

        news_items = []
        for entry in feed.entries[:num_articles]:
            try:
                news_items.append({
                    "title": entry.title,
                    "text": entry.description if hasattr(entry, 'description') else entry.title,
                    "date": entry.published,
                    "link": entry.link,
                    "source": entry.source.title if hasattr(entry, 'source') else "Unknown",
                    "search_term": search_term
                })
            except Exception as e:
                logging.warning(f"Error parsing news entry: {e}")
                continue

        logging.info(f"✅ Fetched {len(news_items)} news articles for {search_term}")
        return news_items
        
    except Exception as e:
        logging.error(f"Error fetching news for {search_term}: {e}")
        return []

def perform_sentiment_analysis(news_items):
    """
    Perform sentiment analysis on news using FinBERT.

    Parameters:
    - news_items (list): List of dictionaries containing news data.

    Returns:
    - DataFrame with sentiment scores.
    """
    if not news_items:
        logging.warning("No news data available for sentiment analysis.")
        return pd.DataFrame()

    try:
        # Load FinBERT model
        logging.info("Loading FinBERT model...")
        sentiment_pipeline = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

        # Analyze sentiment
        logging.info("Performing sentiment analysis...")
        sentiments = []
        for item in news_items:
            try:
                # Analyze both title and text
                title_result = sentiment_pipeline(item["title"])[0]
                text_result = sentiment_pipeline(item["text"])[0]
                
                # Use the more confident sentiment between title and text
                if title_result["score"] > text_result["score"]:
                    sentiment = title_result
                else:
                    sentiment = text_result

                sentiments.append({
                    "title": item["title"],
                    "text": item["text"],
                    "date": item["date"],
                    "source": item["source"],
                    "link": item["link"],
                    "search_term": item["search_term"],
                    "sentiment": sentiment["label"],
                    "sentiment_score": sentiment["score"]
                })
            except Exception as e:
                logging.warning(f"Error analyzing sentiment for news item: {e}")
                continue

        # Convert to DataFrame
        sentiment_df = pd.DataFrame(sentiments)
        return sentiment_df

    except Exception as e:
        logging.error(f"Error in sentiment analysis: {e}")
        return pd.DataFrame()

def main():
    """Main function to fetch news, analyze sentiment, and save results."""
    # List of search terms (stocks and market topics)
    search_terms = [
        "AAPL",          # Apple
        "TSLA",          # Tesla
        "BTC",           # Bitcoin
        "NVDA",          # NVIDIA
        "AI-stocks",     # AI-related stocks
        "Fed-rates",     # Federal Reserve rates
        "market-trends", # Market trends
        "trading-news"   # General trading news
    ]
    
    all_news = []
    global_seen_titles = set()  # Track unique news by title
    
    # Fetch news for each search term
    for term in search_terms:
        logging.info(f"\nFetching news for: {term}")
        news_items = fetch_news(term, num_articles=50)
        
        # Filter out global duplicates
        unique_news = []
        for item in news_items:
            if item["title"] not in global_seen_titles:
                global_seen_titles.add(item["title"])
                unique_news.append(item)
        
        if unique_news:
            all_news.extend(unique_news)
            logging.info(f"Added {len(unique_news)} new unique articles for '{term}'")
    
    logging.info(f"\nTotal unique news articles collected: {len(all_news)}")
    
    if all_news:
        # Perform sentiment analysis
        sentiment_df = perform_sentiment_analysis(all_news)
        
        if not sentiment_df.empty:
            # Save to Parquet
            output_file = os.path.join(DATA_PATH, "news_sentiment.parquet")
            sentiment_df.to_parquet(output_file, compression="snappy")
            logging.info(f"\n💾 Data saved to {output_file}")
            
            # Display statistics
            logging.info("\n📊 News Sentiment Statistics:")
            logging.info(f"Total articles analyzed: {len(sentiment_df)}")
            sentiment_counts = sentiment_df["sentiment"].value_counts()
            for sentiment, count in sentiment_counts.items():
                logging.info(f"{sentiment}: {count} articles ({count/len(sentiment_df)*100:.1f}%)")
            
            # Display sample
            logging.info("\n🔍 Sample news articles and their sentiment:")
            sample = sentiment_df.sample(min(5, len(sentiment_df)))
            for _, row in sample.iterrows():
                try:
                    logging.info(f"\n  Source: {row['source']}")
                    logging.info(f"  Title: {row['title']}")
                    logging.info(f"  Date: {row['date']}")
                    logging.info(f"  Sentiment: {row['sentiment']} (Score: {row['sentiment_score']:.2f})")
                    logging.info(f"  Link: {row['link']}")
                except Exception as e:
                    logging.error(f"Error displaying news item: {e}")
                    continue
        else:
            logging.warning("No sentiment analysis results generated.")
    else:
        logging.warning("No news articles collected.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("\nScript interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")