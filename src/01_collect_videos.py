"""Stage 01 (videos): build Table A for the verified Pakistani smartphone channels.

Reads the channel list from channel_verification.csv, keeps channels whose country
is PK, and collects their recent phone-related videos from the uploads playlist.
Raw files are never overwritten.
"""
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

sys.stdout.reconfigure(encoding="utf-8")

# ---------- Settings ----------
CHANNELS_FILE = Path("data/raw/channel_verification.csv")
OUTPUT_FILE = Path("data/raw/videos_all.csv")
UPLOAD_PAGES = 8              # 8 pages x 50 = latest 400 uploads per channel
MIN_COMMENTS = 150
RECENT_DAYS = 540
MAX_VIDEOS_PER_CHANNEL = 60

PHONE_WORDS = [
    "phone", "phones", "mobile", "mobiles", "smartphone", "smartphones",
    "samsung", "galaxy", "infinix", "tecno", "xiaomi", "redmi", "poco",
    "oppo", "vivo", "iphone", "apple", "realme", "motorola", "moto", "pixel",
]
BRAND_KEYWORDS = {
    "samsung": "Samsung", "galaxy": "Samsung", "infinix": "Infinix",
    "tecno": "Tecno", "xiaomi": "Xiaomi", "redmi": "Xiaomi", "poco": "Poco",
    "oppo": "Oppo", "vivo": "Vivo", "iphone": "Apple", "apple": "Apple",
    "realme": "Realme", "motorola": "Motorola", "moto": "Motorola",
    "pixel": "Google",
}
CONTENT_RULES = [
    ("Unboxing", ["unboxing", "unbox"]),
    ("Comparison", [" vs ", "versus", "comparison", "compare"]),
    ("Review", ["review", "hands on", "hands-on", "camera test"]),
    ("Buying guide", ["buying guide", "best", "under", "top 5", "top 10",
                      "which phone", "kon sa"]),
    ("Launch news", ["launch", "first look", "leaked", "upcoming", "coming",
                     "full details", "new phones"]),
    ("Price update", ["price", "prices", "rate", "mehnga", "mehngay"]),
]


def has_word(text: str, words: list) -> bool:
    """Return True if any word appears as a whole word in the text."""
    return any(re.search(rf"\b{re.escape(word)}\b", text) for word in words)


def detect_brand(title: str) -> str:
    """Return one brand, 'Multiple' if 2+ brands are in the title, else 'Other'."""
    text = title.lower()
    found = []
    for keyword, brand in BRAND_KEYWORDS.items():
        if re.search(rf"\b{keyword}\b", text) and brand not in found:
            found.append(brand)
    if not found:
        return "Other"
    return found[0] if len(found) == 1 else "Multiple"


def detect_content_type(title: str) -> str:
    """Return a simple content type based on title keywords."""
    text = f" {title.lower()} "
    for content_type, keywords in CONTENT_RULES:
        if any(word in text for word in keywords):
            return content_type
    return "Other"


def get_upload_ids(youtube, channel_id: str) -> list:
    """Return the latest video ids from the channel uploads playlist."""
    playlist_id = "UU" + channel_id[2:]
    ids, page_token = [], None
    for _ in range(UPLOAD_PAGES):
        try:
            response = youtube.playlistItems().list(
                part="contentDetails", playlistId=playlist_id,
                maxResults=50, pageToken=page_token).execute()
        except HttpError as error:
            print(f"  Playlist problem for {channel_id}: HTTP {error.resp.status}")
            break
        ids.extend(item["contentDetails"]["videoId"] for item in response["items"])
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return ids


def get_video_rows(youtube, video_ids: list, channel: pd.Series) -> list:
    """Get statistics for the videos (50 ids per request)."""
    today = datetime.now(timezone.utc)
    rows = []
    for start in range(0, len(video_ids), 50):
        chunk = video_ids[start:start + 50]
        response = youtube.videos().list(
            part="snippet,statistics", id=",".join(chunk)).execute()
        for item in response["items"]:
            stats = item["statistics"]
            published = datetime.fromisoformat(
                item["snippet"]["publishedAt"].replace("Z", "+00:00"))
            title = item["snippet"]["title"]
            views = int(stats.get("viewCount", 0))
            likes = int(stats.get("likeCount", 0))
            comments = int(stats.get("commentCount", 0))
            rows.append({
                "video_id": item["id"],
                "channel_id": channel["channel_id"],
                "channel": channel["channel"],
                "channel_country": channel["country"],
                "title": title,
                "brand": detect_brand(title),
                "content_type": detect_content_type(title),
                "views": views,
                "likes": likes,
                "comments_count": comments,
                "days_since_post": (today - published).days,
                "engagement_rate_pct": round((likes + comments) / views * 100, 2) if views else 0,
                "source_url": f"https://www.youtube.com/watch?v={item['id']}",
                "collected_on": today.strftime("%Y-%m-%d"),
            })
    return rows


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
    channels = channels[channels["country"] == "PK"]
    print(f"Verified Pakistani channels: {len(channels)}")

    all_rows = []
    for _, channel in channels.iterrows():
        video_ids = get_upload_ids(youtube, channel["channel_id"])
        rows = get_video_rows(youtube, video_ids, channel)
        videos = pd.DataFrame(rows)
        if videos.empty:
            continue
        is_phone = videos["title"].str.lower().apply(lambda t: has_word(t, PHONE_WORDS))
        keep = videos[is_phone
                      & (videos["comments_count"] >= MIN_COMMENTS)
                      & (videos["days_since_post"] <= RECENT_DAYS)]
        keep = keep.sort_values("days_since_post").head(MAX_VIDEOS_PER_CHANNEL)
        all_rows.append(keep)

    table_a = pd.concat(all_rows, ignore_index=True).drop_duplicates("video_id")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    table_a.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    per_channel = (table_a.groupby("channel")
                   .agg(videos=("video_id", "count"),
                        median_views=("views", "median"),
                        total_comments=("comments_count", "sum"))
                   .sort_values("videos", ascending=False))
    per_channel["median_views"] = per_channel["median_views"].round().astype(int)
    print("\nVideos per channel:")
    print(per_channel.to_string())
    print(f"\nTotal videos (Table A rows): {len(table_a)}")
    print(f"Total comments available on these videos: {table_a['comments_count'].sum()}")
    print("\nContent types:")
    print(table_a["content_type"].value_counts().to_string())
    print("\nBrands:")
    print(table_a["brand"].value_counts().to_string())
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()