"""Stage 06: improve the video labels (content type, brand) and compare performance (Q1).

Performance numbers come from video_features.csv. They are ratios to the typical video
of the SAME channel (1.0 = typical, 2.0 = twice as high), so big channels do not win
just because they are big.

Input : data/processed/video_features.csv              (read only)
Output: data/processed/videos_final.csv
        outputs/figures/content_type_performance.png
        outputs/other_titles.txt                       (never overwritten)
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr

sys.stdout.reconfigure(encoding="utf-8")

# ---------- Settings ----------
INPUT_FILE = Path("data/processed/video_features.csv")
OUTPUT_FILE = Path("data/processed/videos_final.csv")
FIGURE_FILE = Path("outputs/figures/content_type_performance.png")
OTHER_FILE = Path("outputs/other_titles.txt")
MIN_GROUP = 8                  # groups smaller than this are not compared

# The first matching rule wins. Each keyword is a regex that must start at a word start.
CONTENT_RULES = [
    ("News roundup", [r"ep \d+", r"vws extra", r"tech bol chal"]),
    ("Store offer", [r"offer\b", r"deal\b", r"sale\b", r"stock\b", r"clearance", r"pre-?order"]),
    ("Unboxing", [r"unbox\w*"]),
    ("Comparison", [r"vs\.?\b", r"versus", r"compar\w*"]),
    ("Review", [r"review", r"hands[- ]on", r"tests?\b", r"tested", r"opin\w*", r"this phone",
                r"long term", r"after \d+ (?:day|week|month|year)s?"]),
    ("Buying guide", [r"buying guide", r"best", r"under\b", r"top \d+", r"which phone",
                      r"kon ?sa", r"should you buy", r"worth", r"budget", r"paisa wasool",
                      r"box ?pack", r"recommend", r"don'?t buy", r"buy or not",
                      r"value for money", r"avoid", r"waste your money", r"stay away",
                      r"good for", r"bad smartphones", r"paisa barbaad"]),
    ("Launch news", [r"launch", r"first look", r"leak", r"upcoming", r"coming\b",
                     r"full details", r"new phones", r"is here\b", r"announced", r"revealed",
                     r"new model", r"new design", r"wait for", r"next big thing", r"is back"]),
    ("Price update", [r"price", r"rates?\b", r"mehng\w*", r"sast\w*"]),
    ("Explainer", [r"explained", r"features?\b", r"how to", r"kya hai", r"everything",
                   r"all you need", r"tips", r"tricks", r"tax\w*", r"settings", r"market",
                   r"pta", r"cpid", r"patch", r"refurbished", r"expos\w+", r"fraud", r"scam",
                   r"issues?\b", r"episode", r"why\b", r"is it true", r"factory"]),
]
PRICE_NUMBER = re.compile(r"\b\d{2,3},?\d{3}\b|\b\d{2,3}\s?k\b", re.IGNORECASE)

BRANDS = {
    "Samsung": [r"samsung", r"galaxy", r"z ?fold", r"z ?flip"],
    "Apple": [r"iphone", r"apple", r"ios"],
    "Google": [r"pixel", r"google"],
    "Xiaomi": [r"xiaomi", r"redmi"],
    "Poco": [r"poco"],
    "Oppo": [r"oppo", r"reno", r"find x\d*"],
    "Vivo": [r"vivo"],
    "iQOO": [r"iqoo"],
    "Realme": [r"realme"],
    "Infinix": [r"infinix"],
    "Tecno": [r"tecno", r"camon", r"pova", r"phantom"],
    "Motorola": [r"motorola", r"moto"],
    "Honor": [r"honor"],
    "OnePlus": [r"oneplus", r"one plus"],
    "Minor brands": [r"itel", r"nubia", r"red ?magic", r"huawei", r"zte", r"nokia",
                     r"sony", r"asus", r"rog", r"lenovo", r"tcl", r"hmd", r"nothing"],
}
METRICS = {"views_vs_channel": "Views", "like_rate_vs_channel": "Like rate",
           "comment_rate_vs_channel": "Comment rate"}


def detect_content_type(title: str) -> str:
    """Return the content type of a video from its title."""
    text = title.lower()
    for content_type, keywords in CONTENT_RULES:
        if any(re.search(rf"\b(?:{keyword})", text) for keyword in keywords):
            return content_type
    return "Price update" if PRICE_NUMBER.search(text) else "Other"


def detect_brand(title: str) -> str:
    """Return one brand, 'Multiple' if 2+ brands are named, else 'No brand in title'."""
    text = title.lower()
    found = [brand for brand, keywords in BRANDS.items()
             if any(re.search(rf"\b(?:{keyword})\b", text) for keyword in keywords)]
    if not found:
        return "No brand in title"
    return found[0] if len(found) == 1 else "Multiple"


def summarise(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Median performance of each group, as a multiple of the channel's typical video."""
    rows = []
    for name, part in df.groupby(column):
        rows.append({
            column: name, "videos": len(part),
            "views_x": 10 ** part["views_vs_channel"].median(),
            "like_rate_x": 10 ** part["like_rate_vs_channel"].median(),
            "comment_rate_x": 10 ** part["comment_rate_vs_channel"].median(),
            "engagement_pct": part["engagement_rate_pct"].median(),
        })
    return pd.DataFrame(rows).sort_values("views_x", ascending=False).round(2)


def significance(df: pd.DataFrame, column: str) -> None:
    """Print Kruskal-Wallis p-values: do the groups differ more than chance?"""
    groups = [g for _, g in df.groupby(column) if len(g) >= MIN_GROUP]
    if len(groups) < 2:
        print("  Not enough groups to test.")
        return
    for metric, label in METRICS.items():
        p_value = kruskal(*[g[metric] for g in groups]).pvalue
        print(f"  {label:<13} p = {p_value:.3f}")


def plot_content_types(table: pd.DataFrame, n_videos: int) -> None:
    """Three-panel bar chart of performance by content type."""
    table = table.sort_values("views_x")
    labels = [f"{t} (n = {n})" for t, n in zip(table["content_type"], table["videos"])]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
    panels = [("views_x", "Views"), ("like_rate_x", "Like rate"), ("comment_rate_x", "Comment rate")]
    for axis, (column, title) in zip(axes, panels):
        axis.barh(labels, table[column], color="#1F3864")
        axis.axvline(1.0, color="#C55A11", linewidth=1.2, linestyle="--")
        axis.set_title(title)
        axis.set_xlabel("Median ratio to the channel's typical video")
    axes[0].set_ylabel("Content type")
    fig.suptitle(f"Video performance by content type (n = {n_videos} videos; "
                 "dashed line = typical video of the same channel)")
    fig.tight_layout()
    FIGURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_FILE, dpi=200)
    plt.close(fig)


def main() -> None:
    if OUTPUT_FILE.exists() or FIGURE_FILE.exists() or OTHER_FILE.exists():
        raise SystemExit("Output files already exist. Rename them manually to re-run.")
    df = pd.read_csv(INPUT_FILE)
    df = df.rename(columns={"content_type": "content_type_v1", "brand": "brand_v1"})
    df["content_type"] = df["title"].apply(detect_content_type)
    df["brand"] = df["title"].apply(detect_brand)

    print("Labels before and after the improved rules (number of videos)")
    print(f"  Content type 'Other'      : {(df['content_type_v1'] == 'Other').sum()} -> "
          f"{(df['content_type'] == 'Other').sum()}")
    print(f"  Brand 'Other' / no brand  : {(df['brand_v1'] == 'Other').sum()} -> "
          f"{(df['brand'] == 'No brand in title').sum()}")

    big = df[df.groupby("content_type")["video_id"].transform("count") >= MIN_GROUP]
    types = summarise(big, "content_type")
    print(f"\nQ1: Which content type attracts attention? ({len(big)} of {len(df)} videos)")
    print("Values are multiples of the same channel's typical video (1.0 = typical).")
    print(types.to_string(index=False))
    print("\nDo content types differ more than chance? (Kruskal-Wallis test)")
    significance(big, "content_type")

    big_brands = df[df.groupby("brand")["video_id"].transform("count") >= MIN_GROUP]
    print(f"\nBy brand in the title ({big_brands['brand'].nunique()} brands with "
          f"{MIN_GROUP}+ videos)")
    print(summarise(big_brands, "brand").to_string(index=False))
    print("\nDo brands differ more than chance? (Kruskal-Wallis test)")
    significance(big_brands, "brand")

    visible = df[df["likes"] > 0]
    rho, p_value = spearmanr(visible["views_vs_channel"], visible["like_rate_vs_channel"])
    print(f"\nViews vs like rate (relative to channel), Spearman rho = {rho:.2f}, "
          f"p = {p_value:.3f}")

    others = df[df["content_type"] == "Other"]["title"].tolist()
    OTHER_FILE.parent.mkdir(parents=True, exist_ok=True)
    OTHER_FILE.write_text("\n".join(others), encoding="utf-8")
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    plot_content_types(types, len(big))
    print(f"\nSaved: {OUTPUT_FILE}, {FIGURE_FILE} and {OTHER_FILE} ({len(others)} titles)")


if __name__ == "__main__":
    main()