"""Stage 02e: improved rule-based language tag (v2), tested on the hand labels.

Input : data/processed/comments_flagged.csv   (read only)
        data/processed/labels_handmade.csv    (read only)
Output: data/processed/comments_final.csv     (never overwritten)

Rule: Urdu letters dominate -> Urdu. One strong Roman Urdu word -> Roman Urdu.
Two weak (ambiguous) words -> Roman Urdu. No letters at all -> Other. Else English.
The tag is only used to DESCRIBE clusters. The clustering itself uses meaning.
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# ---------- Settings ----------
FLAGGED_FILE = Path("data/processed/comments_flagged.csv")
LABELS_FILE = Path("data/processed/labels_handmade.csv")
OUTPUT_FILE = Path("data/processed/comments_final.csv")
MIN_WORDS, MAX_WORDS = 3, 60
CLASSES = ["English", "Roman Urdu", "Urdu", "Other"]

URDU_SCRIPT = re.compile(r"[\u0600-\u06FF]")
LATIN_LETTERS = re.compile(r"[A-Za-z]")

# Words that are very unlikely in English text (spelling variants included).
STRONG_WORDS = set("""
hai hain hy hn hon hoon hun hoga hogi hoge hota hoti hote hua hui huwa tha thi thay
nahi nhi ni nai kya kia kiya kyun kyu kyon kaise kese kaisa kesa kasa kaisi kesi kasi
kitna kitne kitni ktna kab kahan kaha kidhar kon kaun konsa kaunsa konse kuch koi kisi
bhi bhe bohat bohot boht bahut bht buhat bahot bhut acha achi achay achha achy bilkul
sahi theek thik lena lene lein liya lya liye diya dia dena dein dedo karo karna karni
karen karain kare kr krna krta krte krden krdo kar karta karte karti gaya gya gia gai
gaye gayi aya aye aaya aayi ayega ayegi milega milta milti milay mila raha rahi rahe rha
rhi rhe rahay chahiye chahye chaye chahie chahiya chahta chahti abhi abi sirf zyada zada
ziada lekin magar phir bta bata btao batao batain batayen btaen bataen btain mera meri
mery mere mujhe mujhy mjhe mjhy humein hamay hamen hamari hamara tum tumhara aap ap apka
apki apke aapka aapki apna apni apny uska uski uske iska iski iske unka unki wala wali
walay wale wo woh ye yeh yahan yaha wahan wahin ka ki ke ko se sy ny ne mein py tak sath
saath saal aur jo jis jab toh tou bas yar yaar yr mukabila mehngi mehnga sasta sasti paisa
paise qeemat kmzor kamzor bekar bakwas zabardast shukria jaldi intezar intezaar bnao banao
bnaya banaya bnain banain dekh dekha dekhna dekho dekhe dekhi samajh samjh samjhao sunao
suno bolo
""".split())
# Words that also exist in English or are very common slang: two of them are needed.
WEAK_WORDS = set("ma pa pe par na ya ab ha ho main mai bhai".split())


def detect_language_v2(text: str) -> str:
    """Return Urdu, Roman Urdu, English or Other for one comment."""
    urdu_chars = len(URDU_SCRIPT.findall(text))
    latin_chars = len(LATIN_LETTERS.findall(text))
    if urdu_chars > latin_chars:
        return "Urdu"
    tokens = re.findall(r"[a-z']+", text.lower())
    if not tokens:
        return "Other"
    strong = sum(1 for token in tokens if token in STRONG_WORDS)
    weak = sum(1 for token in tokens if token in WEAK_WORDS)
    return "Roman Urdu" if strong >= 1 or weak >= 2 else "English"


def evaluate(merged: pd.DataFrame, column: str, title: str) -> None:
    """Print accuracy and recall for one detector version."""
    accuracy = (merged["true"] == merged[column]).mean() * 100
    print(f"{title:<28} accuracy {accuracy:.0f}%")
    for language in ("English", "Roman Urdu", "Urdu"):
        total = int((merged["true"] == language).sum())
        hit = int(((merged["true"] == language) & (merged[column] == language)).sum())
        print(f"    recall {language:<11} {hit:>2} of {total:>2}  ({hit / total * 100:.0f}%)")


def main() -> None:
    if OUTPUT_FILE.exists():
        raise SystemExit(f"{OUTPUT_FILE} already exists. Rename it manually to re-run.")
    flagged = pd.read_csv(FLAGGED_FILE)
    flagged["text_clean"] = flagged["text_clean"].fillna("").astype(str)
    labels = pd.read_csv(LABELS_FILE)

    flagged = flagged.rename(columns={"language_type": "language_v1"})
    flagged["language_type"] = flagged["text_clean"].apply(detect_language_v2)

    merged = labels.merge(flagged[["comment_id", "language_v1", "language_type"]],
                          on="comment_id")
    merged["true"] = merged["language_true"].replace({"Urdu script": "Urdu"})
    print("Language detector: old rules (v1) vs improved rules (v2), 100 hand labels")
    evaluate(merged, "language_v1", "v1 (old rules)")
    evaluate(merged, "language_type", "v2 (improved rules)")

    print("\nConfusion table for v2 (rows = your label, columns = v2 tag)")
    table = pd.crosstab(merged["true"], merged["language_type"]).reindex(
        index=CLASSES, columns=CLASSES, fill_value=0)
    print(table.to_string())

    wrong = merged[merged["true"] != merged["language_type"]]
    print(f"\nRemaining v2 mistakes: {len(wrong)}")
    for _, row in wrong.head(12).iterrows():
        print(f"  {row['label_id']} you={row['true']:<10} v2={row['language_type']:<10} "
              f"{str(row['comment_text'])[:70]}")

    # Estimated real mix: the label sample was stratified by the v1 tag.
    pool = flagged[(flagged["comment_kind"] == "substantive")
                   & flagged["word_count"].between(MIN_WORDS, MAX_WORDS)]
    weights = pool["language_v1"].value_counts(normalize=True)
    tags = [tag for tag in weights.index if (merged["language_v1"] == tag).any()]
    weights = weights[tags] / weights[tags].sum()
    v2_share = flagged["language_type"].value_counts(normalize=True) * 100
    print("\nLanguage mix of ALL clean comments (% of comments)")
    print(f"  {'Language':<12} {'v1 old':>8} {'v2 new':>8} {'Estimated real*':>16}")
    v1_share = flagged["language_v1"].value_counts(normalize=True) * 100
    for language in CLASSES:
        estimate = sum(weights[tag] * (merged[merged["language_v1"] == tag]["true"] == language).mean() * 100
                       for tag in tags)
        print(f"  {language:<12} {v1_share.get(language, 0.0):>7.1f}% "
              f"{v2_share.get(language, 0.0):>7.1f}% {estimate:>15.1f}%")
    print("  * rough estimate from the 100 labels (substantive comments, 3-60 words)")

    flagged.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {OUTPUT_FILE} ({len(flagged)} comments)")


if __name__ == "__main__":
    main()