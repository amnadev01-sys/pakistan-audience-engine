"""Stage 01 (candidates): find Pakistan-focused smartphone videos and score them.

Each candidate video is scored from a sample of its own comments:
  pk_pct     = % of comments with Pakistan or Roman Urdu signals
  demand_pct = % of comments with buying-need signals (price, warranty, ...)
Raw files are never overwritten.
"""
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

sys.stdout.reconfigure(encoding="utf-8")

# ---------- Settings ----------
SEARCH_QUERIES = [
    "Samsung mobile price in Pakistan",
    "Infinix Tecno price in Pakistan",
    "Xiaomi Oppo Vivo Realme price in Pakistan",
    "mobile price in Pakistan PTA approved",
    "best mobile under 50000 Pakistan",
    "mobile review Urdu Pakistan",
]
RESULTS_PER_QUERY = 15
RECENT_DAYS = 365
MIN_COMMENTS = 300
MIN_VIEWS = 20_000
MAX_CANDIDATES = 40
PK_PASS = 20        # minimum pk_pct to pass
DEMAND_PASS = 10    # minimum demand_pct to pass

OUTPUT_FILE = Path("data/raw/candidates.csv")

PK_WORDS = ["pkr", "rs", "pta", "daraz", "karachi", "lahore", "islamabad",
            "pakistan", "pakistani", "bhai", "yaar", "kitne", "kitna", "kya",
            "nahi", "hai", "mein", "wala", "bohat", "boht", "sir", "aur"]
DEMAND_WORDS = ["price", "rate", "kitne", "kitni", "kitna", "warranty",
                "available", "availability", "stock", "delivery", "installment",
                "installments", "discount", "offer", "buy", "order", "kahan",
                "kab", "cost", "pta"]

BRAND_KEYWORDS = {
    "samsung": "Samsung", "infinix": "Infinix", "tecno": "Tecno",
    "xiaomi": "Xiaomi", "redmi": "Xiaomi", "oppo": "Oppo", "vivo": "Vivo",
    "iphone": "Apple", "apple": "Apple", "realme": "Realme",
}
CONTENT_RULES = [
    ("Unboxing", ["unboxing", "unbox"]),
    ("Comparison", [" vs ", "versus", "comparison", "compare"]),
    ("Review", ["review", "hands on", "camera test", "gaming test"]),
    ("Buying guide", ["best", "under", "top 5", "top 10", "which phone"]),
    ("Price update", ["price", "prices", "rate"]),
    ("Launch news", ["launch", "first look", "leaked", "new phones", "coming"]),
]


def detect_brand(title: str) -> str:
    """Return the first brand found in the title, else 'Other'."""
    text = title.lower()
    for keyword, brand in BRAND_KEYWORDS.items():
        if keyword in text:
            return brand
    return "Other"


def detect_content_type(title: str) -> str:
    """Return a simple content type based on title keywords."""
    text = f" {title.lower()} "
    for content_type, keywords in CONTENT_RULES:
        if any(word in text for word in keywords):
            return content_type
    return "Other"


def share_with_words(comments: list, words: list) -> float:
    """Return the % of comments that contain at least one of the words."""
    if not comments:
        return 0.0
    pattern = re.compile(r"\b(" + "|".join(words) + r")\b", re.IGNORECASE)
    hits = sum(1 for text in comments if pattern.search(text))
    return round(hits / len(comments) * 100, 1)


def search_video_ids(youtube) -> list:
    """Run all search queries and return unique video ids."""
    since = datetime.now(timezone.utc) - timedelta(days=RECENT_DAYS)
    ids = []
    for query in SEARCH_QUERIES:
        response = youtube.search().list(
            q=query, part="id", type="video", order="relevance",
            regionCode="PK", maxResults=RESULTS_PER_QUERY,
            publishedAfter=since.strftime("%Y-%m-%dT%H:%M:%SZ"),
        ).execute()
        ids.extend(item["id"]["videoId"] for item in response["items"])
    return list(dict.fromkeys(ids))


def get_video_table(youtube, video_ids: list) -> pd.DataFrame:
    """Get statistics for all videos (50 ids per request)."""
    today = datetime.now(timezone.utc)
    rows = []
    for start in range(0, len(video_ids), 50):
        chunk = video_ids[start:start + 50]
        response = youtube.videos().list(
            part="snippet,statistics", id=",".join(chunk)
        ).execute()
        for item in response["items"]:
            stats = item["statistics"]
            published = datetime.fromisoformat(
                item["snippet"]["publishedAt"].replace("Z", "+00:00"))
            title = item["snippet"]["title"]
            rows.append({
                "video_id": item["id"],
                "channel": item["snippet"]["channelTitle"],
                "title": title,
                "brand": detect_brand(title),
                "content_type": detect_content_type(title),
                "views": int(stats.get("viewCount", 0)),
                "likes": int(stats.get("likeCount", 0)),
                "comments_count": int(stats.get("commentCount", 0)),
                "days_since_post": (today - published).days,
                "source_url": f"https://www.youtube.com/watch?v={item['id']}",
                "collected_on": today.strftime("%Y-%m-%d"),
            })
    return pd.DataFrame(rows)


def sample_comments(youtube, video_id: str) -> list:
    """Fetch one page (100) of top comments to score the video."""
    try:
        response = youtube.commentThreads().list(
            part="snippet", videoId=video_id, maxResults=100,
            order="relevance", textFormat="plainText",
        ).execute()
    except HttpError:
        return []
    return [item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            for item in response["items"]]


def main() -> None:
    if OUTPUT_FILE.exists():
        raise SystemExit(f"{OUTPUT_FILE} already exists. Raw data is never "
                         "overwritten. Rename it manually to re-run.")
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise SystemExit("YOUTUBE_API_KEY was not found in the .env file.")
    youtube = build("youtube", "v3", developerKey=api_key)

    video_ids = search_video_ids(youtube)
    videos = get_video_table(youtube, video_ids)
    eligible = videos[(videos["comments_count"] >= MIN_COMMENTS)
                      & (videos["views"] >= MIN_VIEWS)]
    eligible = eligible.sort_values("views", ascending=False).head(MAX_CANDIDATES)
    print(f"Unique videos found: {len(videos)} | Eligible for scoring: {len(eligible)}")

    pk_scores, demand_scores = [], []
    for video_id in eligible["video_id"]:
        comments = sample_comments(youtube, video_id)
        pk_scores.append(share_with_words(comments, PK_WORDS))
        demand_scores.append(share_with_words(comments, DEMAND_WORDS))
    eligible = eligible.copy()
    eligible["pk_pct"] = pk_scores
    eligible["demand_pct"] = demand_scores
    eligible["quality_pass"] = ((eligible["pk_pct"] >= PK_PASS)
                                & (eligible["demand_pct"] >= DEMAND_PASS))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    eligible.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    show = eligible[["channel", "brand", "content_type", "views",
                     "comments_count", "pk_pct", "demand_pct", "quality_pass"]].copy()
    show["channel"] = show["channel"].str.slice(0, 20)
    print("\nCandidate videos (top 15 by views):")
    print(show.head(15).to_string(index=False))
    print(f"\nPassed quality score: {int(eligible['quality_pass'].sum())} of {len(eligible)}")
    print("\nContent types among all eligible videos:")
    print(eligible["content_type"].value_counts().to_string())
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()