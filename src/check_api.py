"""Quick check that the YouTube API key works."""
import os

from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()
api_key = os.getenv("YOUTUBE_API_KEY")
if not api_key:
    raise SystemExit("YOUTUBE_API_KEY was not found in the .env file.")

youtube = build("youtube", "v3", developerKey=api_key)
response = youtube.search().list(
    q="Samsung Galaxy price in Pakistan",
    part="snippet",
    type="video",
    maxResults=3,
).execute()

for item in response["items"]:
    print(item["snippet"]["channelTitle"], "|", item["snippet"]["title"])