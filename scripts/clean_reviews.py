import pandas as pd
import os
import re

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

KEEP = {
    "reviewId":             "review_id",
    "userName":             "user_name",
    "content":              "review_text",
    "score":                "rating",
    "thumbsUpCount":        "thumbs_up",
    "reviewCreatedVersion": "app_version",
    "at":                   "review_date",
    "app_name":             "app_name",
}

def clean_text(text):
    if not isinstance(text, str):
        return None
    text = text.strip()
    # remove emojis and non-ascii noise but keep punctuation
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    text = re.sub(r"\s+", " ", text)
    return text if len(text) > 10 else None

def clean_app(filepath):
    app = os.path.basename(filepath).replace(".csv", "")
    df = pd.read_csv(filepath)

    # keep and rename columns
    df = df[list(KEEP.keys())].rename(columns=KEEP)

    # parse date
    df["review_date"] = pd.to_datetime(df["review_date"], utc=True).dt.tz_localize(None)

    # extract year-month for trend analysis
    df["year_month"] = df["review_date"].dt.to_period("M").astype(str)

    # clean text
    df["review_text"] = df["review_text"].apply(clean_text)

    # drop nulls and duplicates
    df = df.dropna(subset=["review_text"])
    df = df.drop_duplicates(subset=["review_id"])

    # add sentiment bucket from rating
    df["rating_bucket"] = df["rating"].apply(
        lambda r: "negative" if r <= 2 else ("neutral" if r == 3 else "positive")
    )

    # save as parquet
    out_path = f"{OUT_DIR}/{app}.parquet"
    df.to_parquet(out_path, index=False)

    print(f"{app:<15} {len(df):>5} clean reviews → {out_path}")
    return df

if __name__ == "__main__":
    all_dfs = []
    for file in sorted(os.listdir(RAW_DIR)):
        if file.endswith(".csv"):
            df = clean_app(f"{RAW_DIR}/{file}")
            all_dfs.append(df)

    # save combined parquet for cross-app analysis
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_parquet(f"{OUT_DIR}/all_apps.parquet", index=False)
    print(f"\nCombined: {len(combined)} total reviews saved to data/processed/all_apps.parquet")