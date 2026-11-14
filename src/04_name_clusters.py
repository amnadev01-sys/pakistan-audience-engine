"""Stage 04c: lock K = 10, attach cluster names and build the final clustered file.

Input : data/processed/cluster_assignments.csv, comments_final.csv   (read only)
Output: data/processed/cluster_names.csv        (names can be edited by hand later)
        data/processed/comments_clustered.csv   (all clean comments, cluster_id added)
        outputs/figures/cluster_sizes.png       (never overwritten)

cluster_id -1 means "flagged chatter": these comments were not clustered.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

sys.stdout.reconfigure(encoding="utf-8")

# ---------- Settings ----------
K = 10
ASSIGN_FILE = Path("data/processed/cluster_assignments.csv")
COMMENTS_FILE = Path("data/processed/comments_final.csv")
NAMES_FILE = Path("data/processed/cluster_names.csv")
OUTPUT_FILE = Path("data/processed/comments_clustered.csv")
FIGURE_FILE = Path("outputs/figures/cluster_sizes.png")

CLUSTER_INFO = {
    0: ("Short reactions and noise", "Noise"),
    1: ("Phone selection questions (Roman Urdu)", "Customer need"),
    2: ("Video praise (English)", "Appreciation"),
    3: ("Samsung and Motorola brand talk", "Customer need"),
    4: ("Gaming, performance and PTA requests (Roman Urdu)", "Customer need"),
    5: ("Buying decisions and availability (Roman Urdu)", "Customer need"),
    6: ("Device opinions and comparisons (English)", "Customer need"),
    7: ("Video praise (Roman Urdu)", "Appreciation"),
    8: ("Price talk", "Customer need"),
    9: ("Prayers and greetings for the creator", "Appreciation"),
}
TYPE_COLOURS = {"Customer need": "#1F3864", "Appreciation": "#8C8C8C", "Noise": "#C55A11"}


def plot_sizes(summary: pd.DataFrame, n_substantive: int) -> None:
    """Save a horizontal bar chart of cluster sizes, coloured by cluster type."""
    order = summary.sort_values(["cluster_type", "pct_of_substantive"], ascending=[True, True])
    fig, axis = plt.subplots(figsize=(10, 5.5))
    axis.barh(order["cluster_name"], order["pct_of_substantive"],
              color=order["cluster_type"].map(TYPE_COLOURS))
    for position, value in enumerate(order["pct_of_substantive"]):
        axis.text(value + 0.2, position, f"{value:.1f}%", va="center", fontsize=9)
    axis.set_xlabel("Share of substantive comments (%)")
    axis.set_ylabel("Comment cluster")
    axis.set_title(f"Audience comment clusters (K = {K}, n = {n_substantive:,} comments)")
    axis.legend(handles=[Patch(color=c, label=t) for t, c in TYPE_COLOURS.items()],
                title="Cluster type", loc="lower right")
    fig.tight_layout()
    FIGURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_FILE, dpi=200)
    plt.close(fig)


def main() -> None:
    if OUTPUT_FILE.exists() or FIGURE_FILE.exists():
        raise SystemExit("Output files already exist. Rename them manually to re-run.")
    if not NAMES_FILE.exists():
        pd.DataFrame([{"cluster_id": cid, "cluster_name": name, "cluster_type": kind}
                      for cid, (name, kind) in CLUSTER_INFO.items()]
                     ).to_csv(NAMES_FILE, index=False)
        print(f"Created {NAMES_FILE}")
    else:
        print(f"Using the existing {NAMES_FILE} (your edits are kept)")
    names = pd.read_csv(NAMES_FILE)

    comments = pd.read_csv(COMMENTS_FILE)
    assignments = pd.read_csv(ASSIGN_FILE)[["comment_id", f"cluster_k{K}"]]
    df = comments.merge(assignments, on="comment_id", how="left")
    df["cluster_id"] = df[f"cluster_k{K}"].fillna(-1).astype(int)
    df = df.drop(columns=[f"cluster_k{K}"])
    clustered = df[df["cluster_id"] >= 0]
    if len(clustered) != len(assignments):
        raise SystemExit("Some assigned comments were not found. Stop and check the files.")
    n_all, n_sub = len(df), len(clustered)

    rows = []
    for cid, part in clustered.groupby("cluster_id"):
        rows.append({
            "cluster_id": cid,
            "comments": len(part),
            "pct_of_substantive": round(len(part) / n_sub * 100, 1),
            "roman_urdu_pct": round((part["language_type"] == "Roman Urdu").mean() * 100),
            "avg_likes": round(part["like_count"].mean(), 1),
            "videos": part["video_id"].nunique(),
        })
    summary = pd.DataFrame(rows).merge(names, on="cluster_id")

    print(f"\nFinal clusters (K = {K}), n = {n_sub} substantive comments")
    show = summary[["cluster_id", "cluster_name", "cluster_type", "comments",
                    "pct_of_substantive", "roman_urdu_pct", "avg_likes", "videos"]].copy()
    show["cluster_name"] = show["cluster_name"].str.slice(0, 46)
    print(show.to_string(index=False))

    by_type = summary.groupby("cluster_type")["comments"].sum()
    print("\nShare by cluster type")
    print(f"  {'Type':<15} {'% of substantive':>17} {'% of all clean comments':>25}")
    for kind, count in by_type.sort_values(ascending=False).items():
        print(f"  {kind:<15} {count / n_sub * 100:>16.1f}% {count / n_all * 100:>24.1f}%")
    chatter = n_all - n_sub
    print(f"  {'Chatter':<15} {'(not clustered)':>17} {chatter / n_all * 100:>24.1f}%")

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    plot_sizes(summary, n_sub)
    print(f"\nSaved: {OUTPUT_FILE} ({n_all} comments) and {FIGURE_FILE}")


if __name__ == "__main__":
    main()