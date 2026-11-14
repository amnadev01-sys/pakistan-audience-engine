"""Stage 02: clean the raw comments and add a rough language type.

Input : data/raw/comments_raw.csv        (never modified)
Output: data/processed/comments_clean.csv
        data/processed/comments_removed.csv (audit file with removal reasons)
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# ---------- Settings ----------
RAW_FILE = Path("data/raw/comments_raw.csv")
CLEAN_FILE = Path("data/processed/comments_clean.csv")
REMOVED_FILE = Path("data/processed/comments_removed.csv")
MIN_ALNUM_CHARS = 3            # fewer letters/digits than this = too short
MIN_WORDS_FOR_DUP_CHECK = 6    # only long comments are checked for duplicates
RANDOM_SEED = 42

PROMO_PATTERN = re.compile(
    r"https?://|www\.|\.com\b"
    r"|\b(?:\+?92|0)3\d{9}\b"
    r"|\b(?:whatsapp|whats app|telegram)\s+(?:channel|group)\b"
    r"|\b(?:click here|earn money|dm me|follow me)\b",
    re.IGNORECASE,
)
URDU_SCRIPT = re.compile(r"[\u0600-\u06FF]")
LATIN_LETTERS = re.compile(r"[A-Za-z]")

# Roman Urdu marker words. Words that are also common English words are left out.
ROMAN_URDU_WORDS = {
    "hai", "hain", "nahi", "nhi", "kya", "kia", "kaise", "kese", "kitna", "kitne",
    "kitni", "kab", "kahan", "kaha", "bhai", "bhi", "ka", "ki", "ke", "ko", "mein",
    "se", "ap", "aap", "apka", "aur", "ye", "yeh", "wo", "woh", "jo", "bohat",
    "boht", "bahut", "bht", "acha", "achi", "achha", "bilkul", "sahi", "lena",
    "lene", "dena", "karo", "kar", "kr", "karna", "kare", "kren", "krna",
    "chahiye", "chahye", "hoga", "hogi", "hoon", "hun", "tha", "thi", "thay",
    "abhi", "kuch", "koi", "sirf", "zyada", "zada", "lekin", "magar", "phir",
    "bta", "bata", "btao", "batao", "raha", "rahi", "rahe", "rha", "rhi", "mera",
    "meri", "mujhe", "mjhe", "wala", "wali", "walay", "kb", "kon", "konsa",
}


def clean_text(text: str) -> str:
    """Remove @mentions and extra whitespace."""
    text = re.sub(r"@\S+", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()


def alnum_length(text: str) -> int:
    """Count letters and digits only (emoji and punctuation are ignored)."""
    return len(re.sub(r"[\W_]+", "", text))


def detect_language(text: str) -> str:
    """Return a rough language type: Urdu, Roman Urdu, English or Other."""
    urdu_chars = len(URDU_SCRIPT.findall(text))
    latin_chars = len(LATIN_LETTERS.findall(text))
    if urdu_chars > latin_chars:
        return "Urdu"
    tokens = re.findall(r"[a-z']+", text.lower())
    if not tokens:
        return "Other"
    markers = sum(1 for token in tokens if token in ROMAN_URDU_WORDS)
    if markers >= 2 or (markers >= 1 and len(tokens) <= 4):
        return "Roman Urdu"
    return "English"


def main() -> None:
    if CLEAN_FILE.exists() or REMOVED_FILE.exists():
        raise SystemExit("Cleaned files already exist. Rename them manually "
                         "if you want to re-run.")
    raw = pd.read_csv(RAW_FILE)
    df = raw.copy()
    df["text_clean"] = df["comment_text"].apply(clean_text)
    df["word_count"] = df["text_clean"].str.split().str.len()
    df["removal_reason"] = ""

    # Rule 1: too short or emoji-only
    short = df["text_clean"].apply(alnum_length) < MIN_ALNUM_CHARS
    df.loc[short, "removal_reason"] = "too_short_or_emoji_only"

    # Rule 2: link, promo, giveaway or phone number
    promo = df["comment_text"].str.contains(PROMO_PATTERN, na=False)
    df.loc[promo & (df["removal_reason"] == ""), "removal_reason"] = "link_or_promo"

    # Rule 3: long copy-paste duplicates (keep the first copy)
    df["norm"] = df["text_clean"].str.lower().str.replace(r"[\W_]+", "", regex=True)
    candidates = df[(df["word_count"] >= MIN_WORDS_FOR_DUP_CHECK)
                    & (df["removal_reason"] == "")]
    duplicate_index = candidates[candidates.duplicated("norm", keep="first")].index
    df.loc[duplicate_index, "removal_reason"] = "duplicate_text"

    removed = df[df["removal_reason"] != ""].drop(columns=["norm"])
    clean = df[df["removal_reason"] == ""].drop(columns=["norm", "removal_reason"]).copy()
    clean["language_type"] = clean["text_clean"].apply(detect_language)

    CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN_FILE, index=False, encoding="utf-8-sig")
    removed.to_csv(REMOVED_FILE, index=False, encoding="utf-8-sig")

    print("Cleaning summary")
    print(f"  Raw comments            : {len(raw)}")
    for reason, count in removed["removal_reason"].value_counts().items():
        print(f"  Removed - {reason:<18}: {count}")
    print(f"  Clean comments kept     : {len(clean)}")

    language = (clean["language_type"].value_counts(normalize=True) * 100).round(1)
    print("\nLanguage type (% of clean comments, rough rule-based tag):")
    print(language.to_string())
    print(f"\nMedian words per clean comment: {int(clean['word_count'].median())}")

    print("\nClean comments per channel:")
    print(clean["channel"].value_counts().to_string())

    print("\nKept sample (check quality):")
    for _, row in clean.sample(10, random_state=RANDOM_SEED).iterrows():
        print(f"  [{row['language_type']:<10}] {row['text_clean'][:90]}")

    print("\nRemoved sample (check nothing good was lost):")
    for _, row in removed.sample(8, random_state=RANDOM_SEED).iterrows():
        print(f"  [{row['removal_reason']:<22}] {str(row['comment_text']).replace(chr(10), ' ')[:80]}")

    print(f"\nSaved: {CLEAN_FILE} and {REMOVED_FILE}")


if __name__ == "__main__":
    main()