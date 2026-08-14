import time
import pandas as pd
from google_play_scraper import reviews, Sort

APPS = {
    "spotify":   "com.spotify.music",
    "duolingo":  "com.duolingo",
    "zomato":    "com.application.zomato",
    "paytm":     "net.one97.paytm",
    "airbnb":    "com.airbnb.android",
    "instagram": "com.instagram.android",
    "uber":      "com.ubercab",
    "whatsapp":  "com.whatsapp",
    "netflix":   "com.netflix.mediaclient",
    "cred":      "com.dreamplug.androidapp",
    "swiggy":    "in.swiggy.android",
    "phonepe":   "com.phonepe.app",
    "youtube":   "com.google.android.youtube",
}

# apps that need more reviews
BOOST = {
    "paytm", "phonepe", "zomato",
    "swiggy", "whatsapp", "instagram"
}

def scrape_app(name, package_id, count=500):
    print(f"Scraping {name}...")
    result, _ = reviews(
        package_id,
        lang="en",
        country="in",
        sort=Sort.NEWEST,
        count=count,
    )
    new_df = pd.DataFrame(result)
    new_df["app_name"] = name

    existing_path = f"data/raw/{name}.csv"
    if os.path.exists(existing_path):
        existing_df = pd.read_csv(existing_path)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["reviewId"])
        # keep only most recent 2000 reviews
        combined = combined.sort_values("at", ascending=False).head(2000)
        combined.to_csv(existing_path, index=False)
        print(f"  {name}: {len(new_df)} new scraped → {len(combined)} total kept")
    else:
        new_df.to_csv(existing_path, index=False)
        print(f"  {name}: {len(new_df)} reviews saved")

    time.sleep(2)

if __name__ == "__main__":
    for name, pkg in APPS.items():
        count = 1000 if name in BOOST else 500
        scrape_app(name, pkg, count)
    print("\nAll done.")