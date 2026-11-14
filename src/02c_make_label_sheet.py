"""Stage 02c: create the hand-labelling sheet (100 comments).

The sheet is an Excel file with drop-down lists, so labels stay consistent.
The detected language and the chatter flag are NOT shown, to avoid bias.

Input : data/processed/comments_flagged.csv  (read only)
Output: data/processed/label_sheet.xlsx      (never overwritten)
"""
import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

sys.stdout.reconfigure(encoding="utf-8")

# ---------- Settings ----------
FLAGGED_FILE = Path("data/processed/comments_flagged.csv")
OUTPUT_FILE = Path("data/processed/label_sheet.xlsx")
RANDOM_SEED = 42
SAMPLE_PLAN = {"English": 50, "Roman Urdu": 45, "Urdu": 5}   # by detected tag
MIN_WORDS = 3
MAX_WORDS = 60

TOPICS = [
    ("Price", "Asks about or complains about price, cost or tax.",
     "Is ki price kya hai? / Very costly"),
    ("Availability / Where to buy", "Asks where to buy it or if it is in stock.",
     "Lena kaha se hai?"),
    ("Warranty / PTA / Authenticity", "Asks about PTA, warranty, official or refurbished.",
     "Is it PTA approved? Official warranty?"),
    ("Phone advice / Comparison", "Asks which phone to buy, or compares phones.",
     "35k budget mein konsa phone lun? / 13 pro vs gt 20 pro"),
    ("Content request", "Asks the creator to make a video or review about something.",
     "Realme 12 plus ka review banao"),
    ("Specs / Performance", "Talks about camera, battery, display, gaming, heating, etc.",
     "Camera is great but the phone heats up"),
    ("Launch / Release date", "Asks when a phone will launch or arrive.",
     "Moto G Stylus 2025 kab aye ga?"),
    ("Order / Seller service", "Talks about an order, delivery or seller support.",
     "I ordered from your store but got no response"),
    ("Appreciation / Greeting", "Thanks, praise, prayers, greetings, fan talk.",
     "Great video bhai, Allah bless you"),
    ("Other / Unclear", "Anything else, one-word or unclear comments.",
     "Grey"),
]
LANGUAGES = [
    ("English", "A full English sentence.", "Which phone is best under 50k?"),
    ("Roman Urdu",
     "Urdu or Hindi written in English letters. Choose this if the sentence "
     "grammar is Urdu, even when it has English words like phone or camera.",
     "Bhai is phone ka camera kesa hai?"),
    ("Urdu script", "Written in Urdu letters.", "یہ فون کتنے کا ہے؟"),
    ("Other", "Only emoji, numbers or model names, or you cannot tell.", "13 pro"),
]
RULES = [
    "Choose ONE topic for each comment: the main thing the person wants.",
    "Label what the comment says, not what the video is about.",
    "Do not edit label_id, comment_id or comment_text. Do not delete rows.",
    "Fill both yellow columns (topic and language_true) for every row.",
]

HEADER_FILL = PatternFill(fill_type="solid", fgColor="1F3864")
INPUT_FILL = PatternFill(fill_type="solid", fgColor="FFF2CC")
HEADER_FONT = Font(bold=True, color="FFFFFF")


def build_sample(df: pd.DataFrame) -> tuple:
    """Return (eligible pool, shuffled sample with label ids)."""
    pool = df[(df["comment_kind"] == "substantive")
              & df["word_count"].between(MIN_WORDS, MAX_WORDS)]
    parts = []
    for language, size in SAMPLE_PLAN.items():
        group = pool[pool["language_type"] == language]
        parts.append(group.sample(min(size, len(group)), random_state=RANDOM_SEED))
    sample = (pd.concat(parts).sample(frac=1, random_state=RANDOM_SEED)
              .reset_index(drop=True))
    sample.insert(0, "label_id", [f"L{i:03d}" for i in range(1, len(sample) + 1)])
    return pool, sample


def add_validation(sheet, options: list, column: str, last_row: int) -> None:
    """Add a drop-down list to one column of the Labels sheet."""
    rule = DataValidation(
        type="list", formula1='"' + ",".join(options) + '"', allow_blank=True,
        showErrorMessage=True, errorTitle="Invalid value",
        error="Please choose a value from the drop-down list.")
    sheet.add_data_validation(rule)
    rule.add(f"{column}2:{column}{last_row}")


def write_workbook(sample: pd.DataFrame) -> None:
    """Write the Labels sheet and the Guide sheet."""
    book = Workbook()
    sheet = book.active
    sheet.title = "Labels"
    headers = ["label_id", "comment_id", "comment_text", "topic", "language_true"]
    sheet.append(headers)
    for cell in sheet[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
    sheet["D1"].fill = PatternFill(fill_type="solid", fgColor="C55A11")
    sheet["E1"].fill = PatternFill(fill_type="solid", fgColor="C55A11")

    for row_number, (_, row) in enumerate(sample.iterrows(), start=2):
        sheet.cell(row=row_number, column=1, value=row["label_id"])
        sheet.cell(row=row_number, column=2, value=row["comment_id"])
        text_cell = sheet.cell(row=row_number, column=3, value=row["text_clean"])
        text_cell.data_type = "s"          # never treat comment text as a formula
        text_cell.alignment = Alignment(wrap_text=True, vertical="top")
        for column in (4, 5):
            sheet.cell(row=row_number, column=column).fill = INPUT_FILL

    last_row = len(sample) + 1
    add_validation(sheet, [topic for topic, _, _ in TOPICS], "D", last_row)
    add_validation(sheet, [name for name, _, _ in LANGUAGES], "E", last_row)
    for column, width in {"A": 10, "B": 30, "C": 90, "D": 34, "E": 18}.items():
        sheet.column_dimensions[column].width = width
    sheet.freeze_panes = "A2"

    guide = book.create_sheet("Guide")
    guide.append(["Topic", "Meaning", "Example"])
    for item in TOPICS:
        guide.append(list(item))
    guide.append([])
    guide.append(["Language", "Rule", "Example"])
    for item in LANGUAGES:
        guide.append(list(item))
    guide.append([])
    guide.append(["General rules"])
    for rule in RULES:
        guide.append([rule])
    for row in guide.iter_rows():
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    for header_row in (1, len(TOPICS) + 3):
        for cell in guide[header_row]:
            cell.font = Font(bold=True)
    guide.cell(row=len(TOPICS) + len(LANGUAGES) + 5, column=1).font = Font(bold=True)
    for column, width in {"A": 34, "B": 70, "C": 52}.items():
        guide.column_dimensions[column].width = width

    book.save(OUTPUT_FILE)


def main() -> None:
    if OUTPUT_FILE.exists():
        raise SystemExit(f"{OUTPUT_FILE} already exists. Your labels are never "
                         "overwritten. Rename the file manually to re-run.")
    df = pd.read_csv(FLAGGED_FILE)
    df["text_clean"] = df["text_clean"].fillna("").astype(str)
    pool, sample = build_sample(df)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_workbook(sample)

    print("Label sheet summary")
    print(f"  Eligible substantive comments ({MIN_WORDS}-{MAX_WORDS} words): {len(pool)}")
    print(f"  Comments in the sheet: {len(sample)}")
    print(f"  Seed: {RANDOM_SEED}")
    print("\nSample by channel:")
    print(sample["channel"].value_counts().to_string())
    print(f"\nSaved: {OUTPUT_FILE}")
    print("Next: open the file in Excel, label both yellow columns, save, close Excel.")


if __name__ == "__main__":
    main()