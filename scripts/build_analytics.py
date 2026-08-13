import pandas as pd
import json
import os

PROCESSED_DIR = "data/processed"
ANALYSIS_DIR  = "analysis"
OUT_DIR       = "analysis"

def load_insights(app_name):
    path = f"{ANALYSIS_DIR}/{app_name}_insights.json"
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)

def build_app_analytics(app_name):
    df = pd.read_parquet(f"{PROCESSED_DIR}/{app_name}.parquet")
    insights = load_insights(app_name)

    # --- 1. overall summary ---
    summary = {
        "app_name":        app_name,
        "total_reviews":   len(df),
        "avg_rating":      round(df["rating"].mean(), 2),
        "avg_sentiment":   round(df["vader_score"].mean(), 3),
        "pct_positive":    round((df["rating_bucket"] == "positive").mean() * 100, 1),
        "pct_negative":    round((df["rating_bucket"] == "negative").mean() * 100, 1),
        "pct_neutral":     round((df["rating_bucket"] == "neutral").mean() * 100, 1),
    }

    # --- 2. category breakdown from insights ---
    category_counts = {}
    high_priority    = []

    for t in insights:
        cat = t.get("category", "other")
        category_counts[cat] = category_counts.get(cat, 0) + t.get("review_count", 0)

        if t.get("priority") == "high" and "label" in t:
            high_priority.append({
                "label":        t["label"],
                "category":     cat,
                "review_count": t.get("review_count", 0),
                "avg_rating":   t.get("avg_rating", 0),
                "root_cause":   t.get("root_cause", ""),
                "pm_action":    t.get("pm_action", ""),
            })

    summary["category_breakdown"] = category_counts
    summary["high_priority_issues"] = sorted(
        high_priority, key=lambda x: x["review_count"], reverse=True
    )

    # --- 3. monthly trend ---
    monthly = (
        df.groupby("year_month")
        .agg(
            review_count=("rating", "count"),
            avg_rating=("rating", "mean"),
            avg_sentiment=("vader_score", "mean"),
        )
        .round(3)
        .reset_index()
        .sort_values("year_month")
    )
    summary["monthly_trend"] = monthly.to_dict(orient="records")

    # --- 4. rating distribution ---
    rating_dist = df["rating"].value_counts().sort_index()
    summary["rating_distribution"] = rating_dist.to_dict()

    return summary

if __name__ == "__main__":
    apps = [
        f.replace(".parquet", "")
        for f in os.listdir(PROCESSED_DIR)
        if f.endswith(".parquet") and f != "all_apps.parquet"
    ]

    all_analytics = {}
    for app in sorted(apps):
        analytics = build_app_analytics(app)
        all_analytics[app] = analytics
        print(f"{app:<15} "
              f"rating: {analytics['avg_rating']} | "
              f"negative: {analytics['pct_negative']}% | "
              f"high priority issues: {len(analytics['high_priority_issues'])}")

    out_path = f"{OUT_DIR}/all_analytics.json"
    with open(out_path, "w") as f:
        json.dump(all_analytics, f, indent=2)

    print(f"\nSaved → {out_path}")