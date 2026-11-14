"""Stage 02b: flag general chatter (greetings, praise, fan talk) in clean comments.

Chatter comments are kept in the file but flagged, so the clustering stage can use
only substantive comments. The script also prints an audit of the comments that
were removed as link_or_promo, to check for wrongly removed comments.

Input : data/processed/comments_clean.csv    (read only)
        data/processed/comments_removed.csv  (read only)
Output: data/processed/comments_flagged.csv
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# ---------- Settings ----------
CLEAN_FILE = Path("data/processed/comments_clean.csv")
REMOVED_FILE = Path("data/processed/comments_removed.csv")
OUTPUT_FILE = Path("data/processed/comments_flagged.csv")
MAX_WORDS_FOR_CHATTER = 10
RANDOM_SEED = 42

# A comment with any of these signals is always treated as substantive.
CONTENT_PATTERN = re.compile(
    r"\b(?:price|prices|rate|pkr|rs|rupees|pta|warranty|available|availability|"
    r"stock|camera|battery|display|screen|amoled|ips|processor|performance|"
    r"gaming|pubg|charger|charging|storage|ram|review|compare|comparison|vs|"
    r"launch|kab|kb|kitne|kitna|kitni|lena|lene|buy|bought|order|kahan|kaha|"
    r"under|budget|refurbished|official|installment|installments|discount|"
    r"offer|box|pack|network|signal|heating|hang|lag|update|software|"
    r"samsung|infinix|tecno|xiaomi|redmi|poco|oppo|vivo|iphone|apple|realme|"
    r"motorola|moto|pixel|google|honor|itel|nokia|oneplus|huawei)\b"
    r"|\b\d+(?:\.\d+)?\s?k\b|\b\d{5,}\b",
    re.IGNORECASE,
)

CHATTER_PATTERNS = {
    "greeting": re.compile(
        r"^\s*(?:hi|hey)\b|\b(?:assalam\w*|asslam\w*|aslam\w*|salam\w*|aoa|hello)\b",
        re.IGNORECASE),
    "fan_or_subscriber": re.compile(
        r"\b(?:subscribers?|subscribed|subscribe|fan|fans|notification|"
        r"notifications|bell|since\s+\d{4}|early|first|giveaway|give away)\b",
        re.IGNORECASE),
    "praise_or_thanks": re.compile(
        r"\b(?:nice|great|wow|good|awesome|amazing|love|loved|thanks|thank|"
        r"jazakallah|mashallah|masha\s?allah|zabardast|zabardust|behtareen|"
        r"superb|excellent|best video|well done|keep it up|god bless)\b",
        re.IGNORECASE),
}

# Same promo rule as stage 02, used here only to audit the removed comments.
PROMO_PATTERN = re.compile(
    r"https?://|www\.|\.com\b"
    r"|\b(?:\+?92|0)3\d{9}\b"
    r"|\b(?:whatsapp|whats app|telegram)\s+(?:channel|group)\b"
    r"|\b(?:click here|earn money|dm me|follow me)\b",
    re.IGNORECASE,
)

def classify(text: str, word_count: int) -> tuple:
    """Return (comment_kind, chatter_reason) for one comment."""
    if CONTENT_PATTERN.search(text):
        return "substantive", ""
    if word_count > MAX_WORDS_FOR_CHATTER:
        return "substantive", ""
    for reason, pattern in CHATTER_PATTERNS.items():
        if pattern.search(text):
            return "chatter", reason
    return "substantive", ""


def show_sample(frame: pd.DataFrame, title: str, size: int) -> None:
    """Print a small random sample of comments."""
    print(f"\n{title}")
    sample = frame.sample(min(size, len(frame)), random_state=RANDOM_SEED)
    for _, row in sample.iterrows():
        print(f"  [{row['language_type']:<10}] {row['text_clean'][:90]}")


def main() -> None:
    if OUTPUT_FILE.exists():
        raise SystemExit(f"{OUTPUT_FILE} already exists. Rename it manually "
                         "if you want to re-run.")
    df = pd.read_csv(CLEAN_FILE)
    df["text_clean"] = df["text_clean"].fillna("").astype(str)
    results = [classify(text, words)
               for text, words in zip(df["text_clean"], df["word_count"])]
    df["comment_kind"] = [r[0] for r in results]
    df["chatter_reason"] = [r[1] for r in results]
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    chatter = df[df["comment_kind"] == "chatter"]
    substantive = df[df["comment_kind"] == "substantive"]
    print("Chatter flagging summary")
    print(f"  Clean comments        : {len(df)}")
    print(f"  Substantive           : {len(substantive)} "
          f"({len(substantive) / len(df) * 100:.1f}%)")
    print(f"  Chatter               : {len(chatter)} "
          f"({len(chatter) / len(df) * 100:.1f}%)")

    print("\nChatter by reason:")
    print(chatter["chatter_reason"].value_counts().to_string())

    share = (df.groupby("channel")["comment_kind"]
             .apply(lambda kind: round((kind == "chatter").mean() * 100, 1))
             .sort_values(ascending=False).rename("chatter_pct"))
    print("\nChatter share per channel (%):")
    print(share.to_string())

    show_sample(chatter, "Chatter sample (should be useless comments):", 12)
    longest = chatter.sort_values("word_count", ascending=False)
    show_sample(longest.head(30), "Longest chatter comments (false-positive risk check):", 8)
    show_sample(substantive, "Substantive sample (should be useful comments):", 12)

    removed = pd.read_csv(REMOVED_FILE)
    promo = removed[removed["removal_reason"] == "link_or_promo"]
    print("\nAudit of comments removed as link_or_promo (matched keyword | text):")
    for _, row in promo.sample(min(12, len(promo)), random_state=RANDOM_SEED).iterrows():
        text = str(row["comment_text"]).replace("\n", " ")
        match = PROMO_PATTERN.search(text)
        keyword = match.group(0) if match else "?"
        print(f"  [{keyword[:14]:<14}] {text[:100]}")

    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()