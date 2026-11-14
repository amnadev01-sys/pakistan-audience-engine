"""Stage 01 (channels): discover Pakistani smartphone channels.

Runs many Pakistan-focused searches, groups the videos by channel, and scores
each channel from the comments of its most-commented video.
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
    "mobile price in Pakistan",
    "new mobile phones price in Pakistan",
    "Samsung Galaxy price in Pakistan",
    "Infinix price in Pakistan",
    "Tecno price in Pakistan",
    "Xiaomi Redmi price in Pakistan",
    "Oppo Vivo price in Pakistan",
    "Realme Poco price in Pakistan",
    "iPhone price in Pakistan PTA",
    "Motorola Google Pixel price in Pakistan",
    "best mobile under 30000 Pakistan",
    "best mobile under 50000 Pakistan",
    "best camera phone under 100000 Pakistan",
    "best gaming phone Pakistan",
    "mobile unboxing Pakistan",
    "mobile review Urdu",
    "mobile phone review Roman Urdu Pakistan",
    "upcoming mobile phones Pakistan",
    "PTA tax mobile phones Pakistan",
    "mobile market Pakistan price update",
]
RESULTS_PER_QUERY = 50
RECENT_DAYS = 540
MIN_COMMENTS = 150
TOP_CHANNELS = 25

VIDEOS_FILE = Path("data/raw/discovered_videos.csv")
CHANNELS_FILE = Path("data/raw/channel_candidates.csv")

PHONE_WORDS = [
    "phone", "phones", "mobile", "mobiles", "smartphone", "smartphones",
    "samsung", "galaxy", "infinix", "tecno", "xiaomi", "redmi", "poco",
    "oppo", "vivo", "iphone", "apple", "realme", "motorola", "moto", "pixel",
]
PK_WORDS = ["pkr", "rs", "pta", "daraz", "karachi", "lahore", "islamabad",
            "pakistan", "pakistani", "bhai", "yaar", "kitne", "kitna", "kya",
            "nahi", "hai", "mein", "wala", "bohat", "boht", "sir", "aur"]
DEMAND_WORDS = ["price", "rate", "kitne", "kitni", "kitna", "warranty",
                "available", "availability", "stock", "delivery", "installment",
                "installments", "discount", "offer", "buy", "order", "kahan",
                "kab", "cost", "pta"]


def has_word(text: str, words: list) -> bool:
    """Return True if any word appears as a whole word in the text."""
    return any(re.search(rf"\b{re.escape(word)}\b", text) for word in words)


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
    for number, query in enumerate(SEARCH_QUERIES, start=1):
        try:
            response = youtube.search().list(
                q=query, part="id", type="video", order="relevance",
                regionCode="PK", maxResults=RESULTS_PER_QUERY,
                publishedAfter=since.strftime("%Y-%m-%dT%H:%M:%SZ"),
            ).execute()
        except HttpError as error:
            print(f"Search stopped at query {number}: HTTP {error.resp.status}")
            break
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
            rows.append({
                "video_id": item["id"],
                "channel_id": item["snippet"]["channelId"],
                "channel": item["snippet"]["channelTitle"],
                "title": item["snippet"]["title"],
                "views": int(stats.get("viewCount", 0)),
                "likes": int(stats.get("likeCount", 0)),
                "comments_count": int(stats.get("commentCount", 0)),
                "days_since_post": (today - published).days,
                "source_url": f"https://www.youtube.com/watch?v={item['id']}",
                "collected_on": today.strftime("%Y-%m-%d"),
            })
    return pd.DataFrame(rows)


def sample_comments(youtube, video_id: str) -> list:
    """Fetch one page (100) of top comments to score the channel."""
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
    if VIDEOS_FILE.exists() or CHANNELS_FILE.exists():
        raise SystemExit("Output files already exist. Raw data is never "
                         "overwritten. Rename them manually to re-run.")
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise SystemExit("YOUTUBE_API_KEY was not found in the .env file.")
    youtube = build("youtube", "v3", developerKey=api_key)

    video_ids = search_video_ids(youtube)
    videos = get_video_table(youtube, video_ids)
    is_phone = videos["title"].str.lower().apply(lambda t: has_word(t, PHONE_WORDS))
    phone_videos = videos[is_phone].copy()
    phone_videos["enough_comments"] = phone_videos["comments_count"] >= MIN_COMMENTS
    print(f"Unique videos found: {len(videos)} | Phone-related: {len(phone_videos)}")

    summary = (phone_videos.groupby(["channel_id", "channel"])
               .agg(videos_found=("video_id", "count"),
                    videos_150plus=("enough_comments", "sum"),
                    median_views=("views", "median"))
               .reset_index())
    top = summary.sort_values(["videos_150plus", "videos_found"],
                              ascending=False).head(TOP_CHANNELS).copy()

    best_video = (phone_videos.sort_values("comments_count", ascending=False)
                  .drop_duplicates("channel_id")
                  .set_index("channel_id")["video_id"])
    pk_scores, demand_scores = [], []
    for channel_id in top["channel_id"]:
        comments = sample_comments(youtube, best_video[channel_id])
        pk_scores.append(share_with_words(comments, PK_WORDS))
        demand_scores.append(share_with_words(comments, DEMAND_WORDS))
    top["median_views"] = top["median_views"].round().astype(int)
    top["sample_video_id"] = top["channel_id"].map(best_video)
    top["pk_pct"] = pk_scores
    top["demand_pct"] = demand_scores

    VIDEOS_FILE.parent.mkdir(parents=True, exist_ok=True)
    videos.to_csv(VIDEOS_FILE, index=False, encoding="utf-8-sig")
    top.to_csv(CHANNELS_FILE, index=False, encoding="utf-8-sig")

    show = top[["channel", "videos_found", "videos_150plus",
                "median_views", "pk_pct", "demand_pct"]].copy()
    show["channel"] = show["channel"].str.slice(0, 24)
    print(f"\nTop {len(show)} channels (sorted by videos with {MIN_COMMENTS}+ comments):")
    print(show.to_string(index=False))
    print(f"\nSaved: {VIDEOS_FILE} and {CHANNELS_FILE}")


if __name__ == "__main__":
    main()