import pandas as pd
import numpy as np
import json
import os
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

PROCESSED_DIR = "data/processed"
VECTOR_DIR = "vectors"
OUT_DIR = "analysis"
os.makedirs(OUT_DIR, exist_ok=True)

HINDI_STOPS = [
    "hai", "nhi", "aap", "ki", "se", "ho", "kar",
    "ek", "bahut", "acha", "nahin", "app", "good",
    "nice", "best", "great", "very", "really", "just",
    "use", "using", "used", "also", "one", "get"
]

all_stops = list(ENGLISH_STOP_WORDS) + HINDI_STOPS

# stops bertopic from picking up common useless words
vectorizer = CountVectorizer(
    stop_words=all_stops,
    min_df=2,
    ngram_range=(1, 2)  # allows two-word topics like "battery drain"
)

def model_app(app_name):
    print(f"\nModeling topics for {app_name}...")

    df = pd.read_parquet(f"{PROCESSED_DIR}/{app_name}.parquet")
    embeddings = np.load(f"{VECTOR_DIR}/{app_name}_embeddings.npy")
    texts = df["review_text"].tolist()

    topic_model = BERTopic(
        vectorizer_model=vectorizer,
        nr_topics="auto",       # let bertopic decide
        min_topic_size=10,      # minimum 10 reviews per topic
        verbose=False
    )

    topics, probs = topic_model.fit_transform(texts, embeddings)

    df["topic_id"] = topics

    # get topic info
    topic_info = topic_model.get_topic_info()

    # build output: topic summary with sample reviews
    results = []
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            continue  # -1 is BERTopic's "outlier" bucket, skip

        topic_reviews = df[df["topic_id"] == tid]
        top_words = [w for w, _ in topic_model.get_topic(tid)[:6]]

        results.append({
            "topic_id":      tid,
            "keywords":      top_words,
            "review_count":  int(row["Count"]),
            "avg_rating":    round(topic_reviews["rating"].mean(), 2),
            "avg_sentiment": round(topic_reviews["vader_score"].mean(), 3),
            "sample_reviews": topic_reviews["review_text"].head(3).tolist()
        })

    # sort by review count
    results.sort(key=lambda x: x["review_count"], reverse=True)

    out_path = f"{OUT_DIR}/{app_name}_topics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # save df with topic assignments back
    df.to_parquet(f"{PROCESSED_DIR}/{app_name}.parquet", index=False)

    print(f"  {len(results)} topics found → {out_path}")
    for r in results[:5]:
        print(f"    topic {r['topic_id']:>2} | "
              f"{r['review_count']:>3} reviews | "
              f"rating {r['avg_rating']} | "
              f"keywords: {', '.join(r['keywords'][:4])}")

if __name__ == "__main__":
    apps = [
        f.replace(".parquet", "")
        for f in os.listdir(PROCESSED_DIR)
        if f.endswith(".parquet") and f != "all_apps.parquet"
    ]

    for app in sorted(apps):
        model_app(app)

    print("\nAll topic modeling done.")