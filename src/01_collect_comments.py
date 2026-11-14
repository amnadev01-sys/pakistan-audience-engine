"""Stage 01 (comments): build Table B for the selected smartphone videos.

For each video in videos_all.csv the script fetches up to 50 comments sorted by
relevance and up to 50 sorted by time (to reduce top-comment bias).
Raw files are never overwritten.
"""
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

sys.stdout.reconfigure(encoding="utf-8")

# ---------- Settings ----------
VIDEOS_FILE = Path("data/raw/videos_all.csv")
OUTPUT_FILE = Path("data/raw/comments_raw.csv")
COMMENTS_PER_ORDER = 50       # 50 relevance + 50 time = up to 100 per video


def fetch_comments(youtube, video_id: str) -> list:
    """Fetch top-level comments in two orders for one video."""
    rows = []
    for order in ("relevance", "time"):
        try:
            response = youtube.commentThreads().list(
                part="snippet", videoId=video_id, maxResults=COMMENTS_PER_ORDER,
                order=order, textFormat="plainText").execute()
        except HttpError as error:
            print(f"  Skipped {video_id} ({order}): HTTP {error.resp.status}")
            continue
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

    videos = pd.read_csv(VIDEOS_FILE)
    print(f"Videos to process: {len(videos)}")

    all_rows = []
    for number, video_id in enumerate(videos["video_id"], start=1):
        all_rows.extend(fetch_comments(youtube, video_id))
        if number % 50 == 0:
            print(f"  Processed {number} of {len(videos)} videos...")

    comments = pd.DataFrame(all_rows).drop_duplicates(subset="comment_id")
    comments = comments.merge(videos[["video_id", "channel"]], on="video_id", how="left")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    comments.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    per_video = comments.groupby("video_id").size()
    per_channel = (comments.groupby("channel").size()
                   .sort_values(ascending=False).rename("comments"))
    print(f"\nTotal comments collected: {len(comments)}")
    print(f"Videos with at least one comment: {per_video.shape[0]} of {len(videos)}")
    print(f"Median comments per video: {int(per_video.median())}")
    print("\nComments per channel:")
    print(per_channel.to_string())
    print("\nRandom comment sample (quality check):")
    for text in comments["comment_text"].sample(12, random_state=42):
        print(" -", text.replace("\n", " ")[:110])
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()