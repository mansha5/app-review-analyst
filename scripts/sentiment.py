import pandas as pd
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

PROCESSED_DIR = "data/processed"
analyzer = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return compound, label

def score_app(app_name):
    path = f"{PROCESSED_DIR}/{app_name}.parquet"
    df = pd.read_parquet(path)

    results = df["review_text"].apply(get_vader_sentiment)
    df["vader_score"]     = results.apply(lambda x: x[0])
    df["vader_sentiment"] = results.apply(lambda x: x[1])

    df["sentiment_agreement"] = df["rating_bucket"] == df["vader_sentiment"]

    df.to_parquet(path, index=False)

    agree_pct = df["sentiment_agreement"].mean() * 100
    print(f"{app_name:<15} {len(df):>4} reviews | agreement: {agree_pct:.1f}%")

if __name__ == "__main__":
    apps = [
        f.replace(".parquet", "")
        for f in os.listdir(PROCESSED_DIR)
        if f.endswith(".parquet") and f != "all_apps.parquet"
    ]

    for app in sorted(apps):
        score_app(app)

    all_dfs = [
        pd.read_parquet(f"{PROCESSED_DIR}/{a}.parquet")
        for a in sorted(apps)
    ]
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_parquet(f"{PROCESSED_DIR}/all_apps.parquet", index=False)
    print(f"\nCombined updated: {len(combined)} reviews with sentiment scores")
    