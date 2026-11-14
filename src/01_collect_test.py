"""Stage 01 (TEST): collect a small sample of smartphone videos and comments.

This script only checks data quality on 5 videos before the full collection.
Raw files are never overwritten.
"""
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

sys.stdout.reconfigure(encoding="utf-8")

# ---------- Settings ----------
SEARCH_QUERY = "Samsung Galaxy review price in Pakistan"
MAX_SEARCH_RESULTS = 25
MIN_COMMENTS = 300
MIN_VIEWS = 20_000
TEST_VIDEO_LIMIT = 5
PAGES_PER_ORDER = 2          # 2 pages x 100 comments, for each of 2 orders
RECENT_DAYS = 365

RAW_DIR = Path("data/raw")
VIDEOS_FILE = RAW_DIR / "test_videos.csv"
COMMENTS_FILE = RAW_DIR / "test_comments.csv"

BRAND_KEYWORDS = {
    "samsung": "Samsung", "infinix": "Infinix", "tecno": "Tecno",
    "xiaomi": "Xiaomi", "redmi": "Xiaomi", "oppo": "Oppo", "vivo": "Vivo",
    "iphone": "Apple", "apple": "Apple", "realme": "Realme",
}
CONTENT_RULES = [
    ("Unboxing", ["unboxing", "unbox"]),
    ("Comparison", [" vs ", "versus", "comparison"]),
    ("Review", ["review", "hands on", "camera test"]),
    ("Price update", ["price", "prices", "rate"]),
    ("Launch news", ["launch", "first look", "leaked", "new phones"]),
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


def search_video_ids(youtube) -> list:
    """Search for popular, recent videos in Pakistan."""
    since = datetime.now(timezone.utc) - timedelta(days=RECENT_DAYS)
    response = youtube.search().list(
        q=SEARCH_QUERY, part="id", type="video", order="viewCount",
        regionCode="PK", maxResults=MAX_SEARCH_RESULTS,
        publishedAfter=since.strftime("%Y-%m-%dT%H:%M:%SZ"),
    ).execute()
    return [item["id"]["videoId"] for item in response["items"]]


def get_video_table(youtube, video_ids: list) -> pd.DataFrame:
    """Get statistics for each video and build Table A."""
    response = youtube.videos().list(
        part="snippet,statistics", id=",".join(video_ids)
    ).execute()
    today = datetime.now(timezone.utc)
    rows = []
    for item in response["items"]:
        stats = item["statistics"]
        published = datetime.fromisoformat(
            item["snippet"]["publishedAt"].replace("Z", "+00:00")
        )
        views = int(stats.get("viewCount", 0))
        likes = int(stats.get("likeCount", 0))
        comments = int(stats.get("commentCount", 0))
        title = item["snippet"]["title"]
        rows.append({
            "video_id": item["id"],
            "channel": item["snippet"]["channelTitle"],
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
    return pd.DataFrame(rows)


def fetch_comments(youtube, video_id: str) -> list:
    """Fetch comments in two orders (relevance and time) to reduce top-comment bias."""
    rows = []
    for order in ("relevance", "time"):
        page_token = None
        for _ in range(PAGES_PER_ORDER):
            try:
                response = youtube.commentThreads().list(
                    part="snippet", videoId=video_id, maxResults=100,
                    order=order, textFormat="plainText", pageToken=page_token,
                ).execute()
            except HttpError as error:
                print(f"  Skipped {video_id} ({order}): {error.resp.status}")
                break
            for item in response["items"]:
                top = item["snippet"]["topLevelComment"]["snippet"]
                rows.append({
                    "comment_id": item["id"],
                    "video_id": video_id,
                    "comment_text": top["textDisplay"],
                    "like_count": top["likeCount"],
                    "published_at": top["publishedAt"],
                    "reply_count": item["snippet"]["totalReplyCount"],
                    "fetch_order": order,
                })
            page_token = response.get("nextPageToken")
            if not page_token:
                break
    return rows


def main() -> None:
    if VIDEOS_FILE.exists() or COMMENTS_FILE.exists():
        raise SystemExit("Test files already exist. Raw data is never overwritten. "
                         "Delete or rename them manually if you want to re-run.")
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise SystemExit("YOUTUBE_API_KEY was not found in the .env file.")
    youtube = build("youtube", "v3", developerKey=api_key)

    video_ids = search_video_ids(youtube)
    videos = get_video_table(youtube, video_ids)
    qualified = videos[(videos["comments_count"] >= MIN_COMMENTS)
                       & (videos["views"] >= MIN_VIEWS)]
    print(f"Videos found: {len(videos)} | Passed quality filters: {len(qualified)}")
    sample = qualified.head(TEST_VIDEO_LIMIT).copy()

    comment_rows = []
    for video_id in sample["video_id"]:
        comment_rows.extend(fetch_comments(youtube, video_id))
    comments = pd.DataFrame(comment_rows).drop_duplicates(subset="comment_id")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    sample.to_csv(VIDEOS_FILE, index=False, encoding="utf-8-sig")
    comments.to_csv(COMMENTS_FILE, index=False, encoding="utf-8-sig")

    show = sample[["channel", "brand", "content_type", "views", "comments_count"]].copy()
    show["channel"] = show["channel"].str.slice(0, 22)
    print("\nSelected test videos:")
    print(show.to_string(index=False))
    print(f"\nComments collected: {len(comments)}")
    print("\nRandom comment sample (check quality):")
    for text in comments["comment_text"].sample(8, random_state=42):
        print(" -", text.replace("\n", " ")[:110])
    print(f"\nSaved: {VIDEOS_FILE} and {COMMENTS_FILE}")


if __name__ == "__main__":
    main()