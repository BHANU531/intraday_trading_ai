import os
import time
from datetime import datetime, timedelta

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from transformers import pipeline

# Constants
DATA_PATH = "../data/"
os.makedirs(DATA_PATH, exist_ok=True)

def setup_driver():
    """Setup and return a configured Chrome WebDriver."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument(
        "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    )

    try:
        driver = webdriver.Chrome(options=chrome_options)
        return driver
    except Exception as e:
        print(f"Error setting up Chrome driver: {e}")
        return None

def scrape_twitter_search(search_term, max_tweets=100, scroll_pause=1.5):
    """
    Scrape tweets from Twitter search results using Selenium.

    Parameters:
    search_term (str): The term to search for (e.g., "$TSLA" or "Tesla stock").
    max_tweets (int): Maximum number of tweets to collect.
    scroll_pause (float): Time to pause between scrolls.

    Returns:
    list: List of dictionaries containing tweet data.
    """
    print(f"🔍 Scraping tweets for '{search_term}'...")

    driver = setup_driver()
    if not driver:
        return []

    try:
        # Navigate to Twitter search
        driver.get(f"https://twitter.com/search?q={search_term}&src=typed_query")
        time.sleep(5)  # Wait for page to load

        tweets = []
        last_height = driver.execute_script("return document.body.scrollHeight")

        while len(tweets) < max_tweets:
            # Find all tweets on the current page
            tweet_elements = driver.find_elements(By.CSS_SELECTOR, 'article[data-testid="tweet"]')

            for tweet_elem in tweet_elements:
                try:
                    # Extract tweet text
                    tweet_text = tweet_elem.find_element(By.CSS_SELECTOR, 'div[data-testid="tweetText"]').text

                    # Extract username
                    username = tweet_elem.find_element(By.CSS_SELECTOR, 'div[data-testid="User-Name"]').text.split("\n")[0]

                    # Extract date
                    date = tweet_elem.find_element(By.TAG_NAME, "time").get_attribute("datetime")

                    # Extract engagement stats
                    stats = {}
                    stat_elements = tweet_elem.find_elements(By.CSS_SELECTOR, 'div[role="group"] > div')
                    for stat_elem in stat_elements:
                        stat_text = stat_elem.text
                        if "Reply" in stat_elem.get_attribute("aria-label"):
                            stats["replies"] = int(stat_text) if stat_text else 0
                        elif "Retweet" in stat_elem.get_attribute("aria-label"):
                            stats["retweets"] = int(stat_text) if stat_text else 0
                        elif "Like" in stat_elem.get_attribute("aria-label"):
                            stats["likes"] = int(stat_text) if stat_text else 0

                    # Append tweet data
                    tweets.append({
                        "username": username,
                        "text": tweet_text,
                        "date": date,
                        "replies": stats.get("replies", 0),
                        "retweets": stats.get("retweets", 0),
                        "likes": stats.get("likes", 0),
                    })

                    if len(tweets) >= max_tweets:
                        break

                except Exception as e:
                    print(f"Error parsing tweet: {e}")
                    continue

            # Scroll down to load more tweets
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(scroll_pause)

            # Check if we've reached the end of the page
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    except Exception as e:
        print(f"Error during scraping: {e}")
    finally:
        driver.quit()

    print(f"✅ Scraped {len(tweets)} tweets for '{search_term}'")
    return tweets

def perform_sentiment_analysis(tweets_data):
    """
    Perform sentiment analysis using Hugging Face's FinBERT model.

    Parameters:
    tweets_data (list): List of dictionaries containing tweet data.

    Returns:
    pandas.DataFrame: DataFrame with sentiment analysis results.
    """
    if not tweets_data:
        print("No tweets to analyze.")
        return pd.DataFrame()

    # Load FinBERT model
    print("Loading FinBERT model...")
    sentiment_pipeline = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

    # Perform sentiment analysis on each tweet
    print("Performing sentiment analysis...")
    sentiments = []
    for tweet in tweets_data:
        try:
            result = sentiment_pipeline(tweet["text"])[0]
            sentiments.append({
                "text": tweet["text"],
                "sentiment": result["label"],
                "sentiment_score": result["score"],
            })
        except Exception as e:
            print(f"Error analyzing sentiment for tweet: {e}")
            sentiments.append({
                "text": tweet["text"],
                "sentiment": "neutral",
                "sentiment_score": 0.5,
            })

    # Convert to DataFrame
    sentiment_df = pd.DataFrame(sentiments)
    return sentiment_df

def main():
    """Main function to run the Twitter scraper and sentiment analysis."""
    # List of trading-related search terms
    search_terms = ["$AAPL", "Elon Musk", "POTUS", "Doge", "Bitcoin", "Stock Market"]

    all_tweets = []

    # Scrape tweets for each search term
    for term in search_terms:
        print(f"\nScraping tweets for: {term}")
        tweets_data = scrape_twitter_search(term, max_tweets=100)
        if tweets_data:
            all_tweets.extend(tweets_data)

    if all_tweets:
        # Perform sentiment analysis using FinBERT
        sentiment_df = perform_sentiment_analysis(all_tweets)

        # Combine tweets and sentiment data
        combined_df = pd.DataFrame(all_tweets).merge(sentiment_df, on="text", how="left")

        # Save to Parquet
        output_file = os.path.join(DATA_PATH, "trading_tweets_sentiment.parquet")
        combined_df.to_parquet(output_file, compression="snappy")
        print(f"\n💾 Data saved to {output_file}")

        # Display sample
        print("\n🔍 Sample tweets and their sentiment:")
        sample = combined_df.sample(min(5, len(combined_df)))
        for _, row in sample.iterrows():
            print(f"\n  User: @{row['username']}")
            print(f"  Tweet: {row['text'][:100]}..." if len(row['text']) > 100 else f"  Tweet: {row['text']}")
            print(f"  Sentiment: {row['sentiment']} (Score: {row['sentiment_score']:.2f})")
    else:
        print("\nNo tweets scraped. Consider generating synthetic stock tweet data for testing...")

if __name__ == "__main__":
    main()