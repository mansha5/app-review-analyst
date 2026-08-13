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
    print(f"Scraping {name} ({count} reviews)...")
    result, _ = reviews(
        package_id,
        lang="en",
        country="in",
        sort=Sort.NEWEST,
        count=count,
    )
    df = pd.DataFrame(result)
    df["app_name"] = name
    df.to_csv(f"data/raw/{name}.csv", index=False)
    print(f"  saved {len(df)} reviews → data/raw/{name}.csv")
    time.sleep(2)

if __name__ == "__main__":
    for name, pkg in APPS.items():
        count = 1000 if name in BOOST else 500
        scrape_app(name, pkg, count)
    print("\nAll done.")