"""Stage 01 (verify): check which candidate channels have a Pakistani audience.

For every channel in channel_candidates.csv the script reads the channel country
and subscriber count, then samples comments from its 3 most-commented phone videos
and measures Pakistan-specific, India-specific and buying-demand signals.
Raw files are never overwritten.
"""
import os
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

sys.stdout.reconfigure(encoding="utf-8")

# ---------- Settings ----------
CHANNELS_FILE = Path("data/raw/channel_candidates.csv")
VIDEOS_FILE = Path("data/raw/discovered_videos.csv")
OUTPUT_FILE = Path("data/raw/channel_verification.csv")
VIDEOS_PER_CHANNEL = 3

PHONE_WORDS = [
    "phone", "phones", "mobile", "mobiles", "smartphone", "smartphones",
    "samsung", "galaxy", "infinix", "tecno", "xiaomi", "redmi", "poco",
    "oppo", "vivo", "iphone", "apple", "realme", "motorola", "moto", "pixel",
]
PK_WORDS = ["pkr", "pta", "daraz", "pakistan", "pakistani", "pak", "karachi",
            "lahore", "islamabad", "rawalpindi", "peshawar", "quetta",
            "multan", "faisalabad"]
INDIA_WORDS = ["inr", "india", "indian", "flipkart", "jio", "airtel", "bharat"]
DEMAND_WORDS = ["price", "rate", "kitne", "kitni", "kitna", "warranty",
                "available", "availability", "stock", "delivery", "installment",
                "installments", "discount", "offer", "buy", "order", "kahan",
                "kab", "cost", "pta"]


def has_word(text: str, words: list) -> bool:
    """Return True if any word appears as a whole word in the text."""
    return any(re.search(rf"\b{re.escape(word)}\b", text) for word in words)


def share_with_words(comments: list, words: list, symbols: tuple = ()) -> float:
    """Return the % of comments containing a word (or a symbol)."""
    if not comments:
        return 0.0
    pattern = re.compile(r"\b(" + "|".join(words) + r")\b", re.IGNORECASE)
    hits = sum(1 for text in comments
               if pattern.search(text) or any(sym in text for sym in symbols))
    return round(hits / len(comments) * 100, 1)


def get_channel_info(youtube, channel_ids: list) -> dict:
    """Return country and subscriber count for each channel id."""
    info = {}
    for start in range(0, len(channel_ids), 50):
        chunk = channel_ids[start:start + 50]
        response = youtube.channels().list(
            part="snippet,statistics", id=",".join(chunk)).execute()
        for item in response["items"]:
            stats = item["statistics"]
            hidden = stats.get("hiddenSubscriberCount", False)
            info[item["id"]] = {
                "country": item["snippet"].get("country", "unknown"),
                "subscribers": None if hidden else int(stats.get("subscriberCount", 0)),
            }
    return info


def sample_comments(youtube, video_id: str) -> list:
    """Fetch one page (100) of top comments."""
    try:
        response = youtube.commentThreads().list(
            part="snippet", videoId=video_id, maxResults=100,
            order="relevance", textFormat="plainText").execute()
    except HttpError:
        return []
    return [item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            for item in response["items"]]


def rough_signal(pk_pct: float, india_pct: float) -> str:
    """Return a rough label. This is a hint only; final choice is manual."""
    if india_pct > pk_pct:
        return "India signal"
    if pk_pct >= 5:
        return "PK signal"
    return "unclear"


def main() -> None:
    if OUTPUT_FILE.exists():
        raise SystemExit(f"{OUTPUT_FILE} already exists. Raw data is never "
                         "overwritten. Rename it manually to re-run.")
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise SystemExit("YOUTUBE_API_KEY was not found in the .env file.")
    youtube = build("youtube", "v3", developerKey=api_key)

    channels = pd.read_csv(CHANNELS_FILE)
    videos = pd.read_csv(VIDEOS_FILE)
    videos = videos[videos["title"].str.lower().apply(lambda t: has_word(t, PHONE_WORDS))]
    info = get_channel_info(youtube, channels["channel_id"].tolist())

    rows = []
    for _, channel in channels.iterrows():
        top_videos = (videos[videos["channel_id"] == channel["channel_id"]]
                      .sort_values("comments_count", ascending=False)
                      .head(VIDEOS_PER_CHANNEL))
        comments = []
        for video_id in top_videos["video_id"]:
            comments.extend(sample_comments(youtube, video_id))
        pk_pct = share_with_words(comments, PK_WORDS)
        india_pct = share_with_words(comments, INDIA_WORDS, symbols=("₹",))
        rows.append({
            "channel_id": channel["channel_id"],
            "channel": channel["channel"],
            "country": info.get(channel["channel_id"], {}).get("country", "unknown"),
            "subscribers": info.get(channel["channel_id"], {}).get("subscribers"),
            "videos_found": channel["videos_found"],
            "videos_150plus": channel["videos_150plus"],
            "comments_sampled": len(comments),
            "pk_specific_pct": pk_pct,
            "india_pct": india_pct,
            "demand_pct": share_with_words(comments, DEMAND_WORDS),
            "signal": rough_signal(pk_pct, india_pct),
        })
    result = pd.DataFrame(rows)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    show = result.drop(columns=["channel_id"]).copy()
    show["channel"] = show["channel"].str.slice(0, 20)
    print(show.to_string(index=False))
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()