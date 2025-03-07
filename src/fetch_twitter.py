import os
import time
from datetime import datetime, timedelta
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from transformers import pipeline
from webdriver_manager.chrome import ChromeDriverManager
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/twitter_scraper.log'),
        logging.StreamHandler()
    ]
)

# Constants
DATA_PATH = "data/"
os.makedirs(DATA_PATH, exist_ok=True)

def setup_driver():
    """Setup and return a configured Chrome WebDriver with automatic installation."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--remote-debugging-port=9222")
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    try:
        # Automatically download and setup ChromeDriver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Add undetectable properties
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {
            "userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
        })
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        return driver
    except Exception as e:
        logging.error(f"Error setting up Chrome driver: {e}")
        return None

def scrape_twitter_search(search_term, max_tweets=100, scroll_pause=1.5):
    """
    Scrape tweets from Twitter search results using Selenium.
    """
    logging.info(f"🔍 Scraping tweets for '{search_term}'...")

    driver = setup_driver()
    if not driver:
        return []

    tweets = []
    seen_tweets = set()  # Track unique tweets
    try:
        # Use Nitter as an alternative to Twitter
        driver.get(f"https://nitter.net/search?f=tweets&q={search_term}&since=0")
        time.sleep(5)  # Wait for page to load

        last_height = driver.execute_script("return document.body.scrollHeight")
        scroll_attempts = 0
        max_scroll_attempts = 10

        while len(tweets) < max_tweets and scroll_attempts < max_scroll_attempts:
            # Find all tweets on the current page
            tweet_elements = driver.find_elements(By.CLASS_NAME, "timeline-item")

            for tweet_elem in tweet_elements:
                try:
                    # Extract tweet text
                    tweet_text = tweet_elem.find_element(By.CLASS_NAME, "tweet-content").text

                    # Skip if we've seen this tweet before
                    if tweet_text in seen_tweets:
                        continue
                    seen_tweets.add(tweet_text)

                    # Extract username
                    username = tweet_elem.find_element(By.CLASS_NAME, "username").text

                    # Extract date
                    date_elem = tweet_elem.find_element(By.CLASS_NAME, "tweet-date")
                    date = date_elem.get_attribute("title")

                    # Extract stats
                    stats = {}
                    stat_elements = tweet_elem.find_elements(By.CLASS_NAME, "tweet-stat")
                    for stat_elem in stat_elements:
                        stat_text = stat_elem.text.strip()
                        if "reply" in stat_text.lower():
                            stats["replies"] = int(stat_text.split()[0]) if stat_text.split()[0].isdigit() else 0
                        elif "retweet" in stat_text.lower():
                            stats["retweets"] = int(stat_text.split()[0]) if stat_text.split()[0].isdigit() else 0
                        elif "like" in stat_text.lower():
                            stats["likes"] = int(stat_text.split()[0]) if stat_text.split()[0].isdigit() else 0

                    tweets.append({
                        "username": username,
                        "text": tweet_text,
                        "date": date,
                        "replies": stats.get("replies", 0),
                        "retweets": stats.get("retweets", 0),
                        "likes": stats.get("likes", 0),
                        "search_term": search_term  # Add search term for reference
                    })

                    if len(tweets) >= max_tweets:
                        break

                except Exception as e:
                    logging.warning(f"Error parsing tweet: {e}")
                    continue

            if len(tweets) >= max_tweets:
                break

            # Scroll down
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(scroll_pause)

            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                scroll_attempts += 1
            else:
                scroll_attempts = 0
            last_height = new_height

    except Exception as e:
        logging.error(f"Error during scraping: {e}")
    finally:
        driver.quit()

    logging.info(f"✅ Scraped {len(tweets)} unique tweets for '{search_term}'")
    return tweets

def perform_sentiment_analysis(tweets_data):
    """
    Perform sentiment analysis using FinBERT model.
    """
    if not tweets_data:
        logging.warning("No tweets to analyze.")
        return pd.DataFrame()

    try:
        # Load FinBERT model
        logging.info("Loading FinBERT model...")
        sentiment_pipeline = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

        # Perform sentiment analysis on each tweet
        logging.info("Performing sentiment analysis...")
        sentiments = []
        for tweet in tweets_data:
            try:
                result = sentiment_pipeline(tweet["text"])[0]
                sentiments.append({
                    "text": tweet["text"],
                    "username": tweet["username"],
                    "date": tweet["date"],
                    "sentiment": result["label"],
                    "sentiment_score": result["score"],
                    "replies": tweet["replies"],
                    "retweets": tweet["retweets"],
                    "likes": tweet["likes"]
                })
            except Exception as e:
                logging.warning(f"Error analyzing sentiment for tweet: {e}")
                continue

        # Convert to DataFrame
        sentiment_df = pd.DataFrame(sentiments)
        return sentiment_df

    except Exception as e:
        logging.error(f"Error in sentiment analysis: {e}")
        return pd.DataFrame()

def main():
    """Main function to run the Twitter scraper and sentiment analysis."""
    # List of trading-related search terms
    search_terms = ["$AAPL", "Elon Musk", "POTUS", "Doge", "Bitcoin", "Stock Market"]
    
    all_tweets = []
    global_seen_tweets = set()  # Track unique tweets across all search terms
    
    # Scrape tweets for each search term
    for term in search_terms:
        logging.info(f"\nScraping tweets for: {term}")
        tweets_data = scrape_twitter_search(term, max_tweets=100)
        
        # Filter out global duplicates
        unique_tweets = []
        for tweet in tweets_data:
            if tweet["text"] not in global_seen_tweets:
                global_seen_tweets.add(tweet["text"])
                unique_tweets.append(tweet)
        
        if unique_tweets:
            all_tweets.extend(unique_tweets)
            logging.info(f"Added {len(unique_tweets)} new unique tweets for '{term}'")
    
    logging.info(f"\nTotal unique tweets collected: {len(all_tweets)}")
    
    if all_tweets:
        # Perform sentiment analysis using FinBERT
        sentiment_df = perform_sentiment_analysis(all_tweets)
        
        if not sentiment_df.empty:
            # Save to Parquet
            output_file = os.path.join(DATA_PATH, "trading_tweets_sentiment.parquet")
            sentiment_df.to_parquet(output_file, compression="snappy")
            logging.info(f"\n💾 Data saved to {output_file}")
            
            # Display sample and statistics
            logging.info("\n📊 Tweet Statistics:")
            logging.info(f"Total tweets analyzed: {len(sentiment_df)}")
            sentiment_counts = sentiment_df["sentiment"].value_counts()
            for sentiment, count in sentiment_counts.items():
                logging.info(f"{sentiment}: {count} tweets ({count/len(sentiment_df)*100:.1f}%)")
            
            # Display sample
            logging.info("\n🔍 Sample unique tweets and their sentiment:")
            sample = sentiment_df.sample(min(5, len(sentiment_df)))
            for _, row in sample.iterrows():
                try:
                    logging.info(f"\n  User: @{row['username']}")
                    logging.info(f"  Tweet: {row['text'][:100]}..." if len(row['text']) > 100 else f"  Tweet: {row['text']}")
                    logging.info(f"  Sentiment: {row['sentiment']} (Score: {row['sentiment_score']:.2f})")
                    logging.info(f"  Engagement: {row['likes']} likes, {row['retweets']} retweets, {row['replies']} replies")
                except Exception as e:
                    logging.error(f"Error displaying tweet: {e}")
                    continue
        else:
            logging.warning("No sentiment analysis results generated.")
    else:
        logging.warning("No tweets scraped.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("\nScript interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")