import snscrape.modules.twitter as sntwitter
import pandas as pd
import os
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Define influential accounts & keywords
TRACKED_ACCOUNTS = ["elonmusk", "POTUS",]
TRACKED_KEYWORDS = ["$TSLA", "$AAPL", "crypto", "stock market"]

# Initialize sentiment analysis models
finbert_sentiment = pipeline("text-classification", model="yiyanghkust/finbert-tone")
vader_analyzer = SentimentIntensityAnalyzer()


def scrape_tweets(query, max_tweets=20):
    """Scrapes recent tweets based on a search query."""
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i >= max_tweets:
            break
        finbert_score = finbert_sentiment(tweet.content)[0]
        vader_score = vader_analyzer.polarity_scores(tweet.content)["compound"]

        tweets.append({
            "date": tweet.date,
            "username": tweet.user.username,
            "tweet": tweet.content,
            "finbert_sentiment": finbert_score["label"],
            "finbert_confidence": finbert_score["score"],
            "vader_sentiment": vader_score
        })

    return tweets


def collect_tweets():
    """Scrape tweets from influencers & financial keywords."""
    all_tweets = []

    # Scrape tweets from specific users
    for account in TRACKED_ACCOUNTS:
        query = f"(from:{account}) since:2024-03-01"
        all_tweets.extend(scrape_tweets(query, max_tweets=10))

    # Scrape tweets based on keywords
    for keyword in TRACKED_KEYWORDS:
        query = f"({keyword}) since:2024-03-01"
        all_tweets.extend(scrape_tweets(query, max_tweets=10))

    # Save tweets
    df = pd.DataFrame(all_tweets)
    os.makedirs("../data/twitter_sentiment/", exist_ok=True)
    df.to_csv("../data/twitter_sentiment/recent_tweets.csv", index=False)
    print("✅ Tweets collected & sentiment analyzed.")


if __name__ == "__main__":
    collect_tweets()
