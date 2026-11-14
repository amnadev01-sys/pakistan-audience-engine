"""Stage 04b: profile the clusters for the chosen values of K.

For every cluster the report shows: size, channel mix (and which channel is
over-represented), language mix, hand-labelled topics, distinctive words and the
comments closest to the cluster centre. The full report goes to a text file.

Input : data/processed/embeddings.npy, embedding_ids.csv, cluster_assignments.csv,
        comments_final.csv, labels_handmade.csv   (read only)
Output: outputs/cluster_profiles.txt              (never overwritten)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

sys.stdout.reconfigure(encoding="utf-8")

# ---------- Settings ----------
EMBEDDINGS_FILE = Path("data/processed/embeddings.npy")
IDS_FILE = Path("data/processed/embedding_ids.csv")
ASSIGN_FILE = Path("data/processed/cluster_assignments.csv")
COMMENTS_FILE = Path("data/processed/comments_final.csv")
LABELS_FILE = Path("data/processed/labels_handmade.csv")
REPORT_FILE = Path("outputs/cluster_profiles.txt")
K_TO_PROFILE = [6, 10]
N_REPRESENTATIVE = 6
N_WORDS = 10
MIN_WORD_DOCS = 30        # a word must appear in at least this many comments overall


def distinctive_words(texts: pd.Series, mask: np.ndarray) -> list:
    """Return words that are much more common inside the cluster than overall."""
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b[a-zA-Z]{3,}\b", lowercase=True,
                                 min_df=MIN_WORD_DOCS, binary=True)
    matrix = vectorizer.fit_transform(texts)
    overall = np.asarray(matrix.mean(axis=0)).ravel()
    inside = np.asarray(matrix[mask].mean(axis=0)).ravel()
    docs_inside = inside * mask.sum()
    lift = np.where(docs_inside >= max(15, 0.03 * mask.sum()), inside / overall, 0)
    top = np.argsort(-lift)[:N_WORDS]
    names = vectorizer.get_feature_names_out()
    return [f"{names[i]} (x{lift[i]:.1f})" for i in top if lift[i] > 0]


def profile_one_k(k: int, df: pd.DataFrame, embeddings: np.ndarray,
                  labels: pd.Series, lines: list, summary: list) -> None:
    """Add the profile of every cluster for one value of K."""
    clusters = df[f"cluster_k{k}"].to_numpy()
    overall_channel = df["channel"].value_counts(normalize=True)
    lines.append("=" * 100)
    lines.append(f"K = {k}   (n = {len(df)} substantive comments)")
    lines.append("=" * 100)
    for cluster in range(k):
        mask = clusters == cluster
        part = df[mask]
        share = mask.mean() * 100
        channel_share = part["channel"].value_counts(normalize=True)
        lift = (channel_share / overall_channel[channel_share.index]).sort_values(ascending=False)
        top_channel = lift.index[0]
        language = part["language_type"].value_counts(normalize=True) * 100
        centre = embeddings[mask].mean(axis=0)
        nearest = np.argsort(-(embeddings[mask] @ centre))[:N_REPRESENTATIVE]
        hand = part["topic"].dropna().value_counts()

        lines.append(f"\n--- Cluster {cluster}: {mask.sum()} comments ({share:.1f}%) ---")
        lines.append("Channel mix : " + ", ".join(
            f"{name} {value * 100:.0f}%" for name, value in channel_share.head(3).items()))
        lines.append(f"Most over-represented channel: {top_channel} "
                     f"({channel_share[top_channel] * 100:.0f}% here vs "
                     f"{overall_channel[top_channel] * 100:.0f}% overall, x{lift[top_channel]:.1f})")
        lines.append("Language    : " + ", ".join(
            f"{name} {value:.0f}%" for name, value in language.items()))
        lines.append("Hand topics : " + (", ".join(f"{t} ({c})" for t, c in hand.items())
                                         if len(hand) else "none labelled here"))
        lines.append("Top words   : " + ", ".join(distinctive_words(df["text_clean"], mask)))
        lines.append("Closest comments to the cluster centre:")
        for text in part["text_clean"].iloc[nearest]:
            lines.append(f"   - {text[:110]}")
        summary.append({
            "k": k, "cluster": cluster, "comments": int(mask.sum()),
            "share_pct": round(share, 1), "top_channel": top_channel,
            "channel_lift": round(float(lift[top_channel]), 1),
            "roman_urdu_pct": round(float(language.get("Roman Urdu", 0.0)), 0),
        })


def main() -> None:
    if REPORT_FILE.exists():
        raise SystemExit(f"{REPORT_FILE} already exists. Rename it manually to re-run.")
    embeddings = np.load(EMBEDDINGS_FILE)
    ids = pd.read_csv(IDS_FILE)["comment_id"]
    assignments = pd.read_csv(ASSIGN_FILE)
    comments = pd.read_csv(COMMENTS_FILE).drop_duplicates("comment_id").set_index("comment_id")
    labels = pd.read_csv(LABELS_FILE).set_index("comment_id")["topic"]

    df = comments.loc[ids].reset_index()
    df["text_clean"] = df["text_clean"].fillna("").astype(str)
    df = df.merge(assignments, on="comment_id", how="left")
    df["topic"] = df["comment_id"].map(labels)
    if len(df) != len(embeddings):
        raise SystemExit("Row counts do not match. Stop and check the input files.")

    lines, summary = [], []
    for k in K_TO_PROFILE:
        profile_one_k(k, df, embeddings, labels, lines, summary)

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")

    table = pd.DataFrame(summary)
    for k in K_TO_PROFILE:
        print(f"\nCluster overview, K = {k}")
        print(table[table["k"] == k].drop(columns="k").to_string(index=False))
    print("\nChannel lift: x1.0 = same as overall; above x2.0 means one channel dominates the cluster.")
    print(f"\nFull report with comments and words saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()