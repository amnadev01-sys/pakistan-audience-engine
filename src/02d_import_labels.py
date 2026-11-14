"""Stage 02d: import the hand-made labels and test the rule-based language detector.

Input : data/processed/label_sheet_completed.xlsx  (read only)
        data/processed/comments_flagged.csv        (read only)
Output: data/processed/labels_handmade.csv         (never overwritten)
"""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# ---------- Settings ----------
LABEL_XLSX = Path("data/processed/label_sheet_completed.xlsx")
FLAGGED_FILE = Path("data/processed/comments_flagged.csv")
OUTPUT_FILE = Path("data/processed/labels_handmade.csv")
EXPECTED_ROWS = 100
MIN_WORDS, MAX_WORDS = 3, 60            # same pool rules as the label sheet

TOPICS = [
    "Price", "Availability / Where to buy", "Warranty / PTA / Authenticity",
    "Phone advice / Comparison", "Content request", "Specs / Performance",
    "Launch / Release date", "Order / Seller service",
    "Appreciation / Greeting", "Other / Unclear",
]
LANGUAGES = ["English", "Roman Urdu", "Urdu script", "Other"]
DETECTOR_NAME = {"Urdu script": "Urdu"}  # the detector calls it "Urdu"
CLASSES = ["English", "Roman Urdu", "Urdu", "Other"]


def check_labels(labels: pd.DataFrame, flagged: pd.DataFrame) -> list:
    """Return a list of problems found in the label sheet (empty list = all good)."""
    problems = []
    if len(labels) != EXPECTED_ROWS:
        problems.append(f"Expected {EXPECTED_ROWS} rows, found {len(labels)}.")
    for column in ("topic", "language_true"):
        blanks = labels[column].isna().sum()
        if blanks:
            problems.append(f"{blanks} blank cells in '{column}'.")
    bad_topics = set(labels["topic"].dropna()) - set(TOPICS)
    bad_languages = set(labels["language_true"].dropna()) - set(LANGUAGES)
    if bad_topics:
        problems.append(f"Unknown topic values: {sorted(bad_topics)}")
    if bad_languages:
        problems.append(f"Unknown language values: {sorted(bad_languages)}")
    if labels["comment_id"].duplicated().any():
        problems.append("Duplicate comment_id values.")
    known = flagged.set_index("comment_id")["text_clean"].astype(str)
    missing = ~labels["comment_id"].isin(known.index)
    if missing.any():
        problems.append(f"{int(missing.sum())} comment_id values not found in the flagged file.")
    else:
        original = labels["comment_id"].map(known).str.strip()
        edited = labels["comment_text"].astype(str).str.strip() != original
        if edited.any():
            ids = ", ".join(labels.loc[edited, "label_id"].head(5))
            problems.append(f"{int(edited.sum())} comment texts differ from the data (e.g. {ids}).")
    return problems


def print_topic_summary(labels: pd.DataFrame) -> None:
    """Print the hand-labelled topic and language counts."""
    topics = labels["topic"].value_counts()
    print("\nHand-labelled topics (n = 100)")
    for topic, count in topics.items():
        print(f"  {topic:<32} {count:>3}  ({count / len(labels) * 100:.0f}%)")
    languages = labels["language_true"].value_counts()
    print("\nHand-labelled language (n = 100)")
    for language, count in languages.items():
        print(f"  {language:<32} {count:>3}")


def test_detector(labels: pd.DataFrame, flagged: pd.DataFrame) -> None:
    """Compare the rule-based language tag with the hand labels."""
    merged = labels.merge(flagged[["comment_id", "language_type"]], on="comment_id")
    merged["true"] = merged["language_true"].replace(DETECTOR_NAME)
    merged["detected"] = merged["language_type"]
    accuracy = (merged["true"] == merged["detected"]).mean() * 100
    print(f"\nLanguage detector accuracy on the 100 hand labels: {accuracy:.0f}%")

    print("\nConfusion table (rows = your label, columns = detector tag)")
    table = pd.crosstab(merged["true"], merged["detected"]).reindex(
        index=CLASSES, columns=CLASSES, fill_value=0)
    print(table.to_string())

    print("\nRecall per language (how many of your labels the detector found)")
    for language in CLASSES:
        total = int((merged["true"] == language).sum())
        if total:
            hit = int(((merged["true"] == language) & (merged["detected"] == language)).sum())
            print(f"  {language:<12} {hit:>2} of {total:>2}  ({hit / total * 100:.0f}%)")

    wrong = merged[(merged["true"] == "Roman Urdu") & (merged["detected"] == "English")]
    print(f"\nRoman Urdu comments tagged as English: {len(wrong)}. First 10:")
    for _, row in wrong.head(10).iterrows():
        print(f"  {row['label_id']}: {str(row['comment_text'])[:85]}")

    # The sample was stratified by detector tag, so weight each tag by its pool size.
    pool = flagged[(flagged["comment_kind"] == "substantive")
                   & flagged["word_count"].between(MIN_WORDS, MAX_WORDS)]
    weights = pool["language_type"].value_counts(normalize=True)
    sampled_tags = [tag for tag in weights.index if (merged["detected"] == tag).any()]
    weights = weights[sampled_tags] / weights[sampled_tags].sum()
    print("\nRough estimate of the real language mix (% of substantive comments)")
    print(f"  {'Language':<12} {'Detector says':>13} {'Estimated real':>15}")
    detector_share = pool["language_type"].value_counts(normalize=True) * 100
    for language in CLASSES:
        estimate = 0.0
        for tag in sampled_tags:
            group = merged[merged["detected"] == tag]
            estimate += weights[tag] * (group["true"] == language).mean() * 100
        print(f"  {language:<12} {detector_share.get(language, 0.0):>12.1f}% {estimate:>14.1f}%")
    print("  (Small sample: treat the estimate as rough, not exact.)")


def main() -> None:
    if OUTPUT_FILE.exists():
        raise SystemExit(f"{OUTPUT_FILE} already exists. Hand labels are never "
                         "overwritten. Rename the file manually to re-run.")
    labels = pd.read_excel(LABEL_XLSX, sheet_name="Labels", engine="openpyxl")
    flagged = pd.read_csv(FLAGGED_FILE)

    problems = check_labels(labels, flagged)
    print("Label sheet check")
    if problems:
        for problem in problems:
            print(f"  PROBLEM: {problem}")
        raise SystemExit("Fix the problems above, then run the script again. Nothing was saved.")
    print(f"  Rows: {len(labels)} | blanks: 0 | invalid values: 0 | texts unchanged: yes")

    labels.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print_topic_summary(labels)
    test_detector(labels, flagged)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()