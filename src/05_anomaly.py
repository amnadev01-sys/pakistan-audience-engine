"""Stage 05: find unusual videos with Isolation Forest.

Video performance depends strongly on the channel, so every feature is measured
against the typical value of the SAME channel (channels with fewer than 5 videos use
the overall typical value). The audience mix is added for description only.

Input : data/raw/videos_all.csv, data/processed/comments_clustered.csv,
        data/processed/cluster_names.csv      (read only)
Output: data/processed/video_features.csv
        outputs/figures/video_anomalies.png   (never overwritten)
"""
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

sys.stdout.reconfigure(encoding="utf-8")

# ---------- Settings ----------
VIDEOS_FILE = Path("data/raw/videos_all.csv")
COMMENTS_FILE = Path("data/processed/comments_clustered.csv")
NAMES_FILE = Path("data/processed/cluster_names.csv")
FEATURES_FILE = Path("data/processed/video_features.csv")
FIGURE_FILE = Path("outputs/figures/video_anomalies.png")
SEEDS = [42, 7, 123]
FLAG_FRACTION = 0.05          # the 5% most unusual videos are flagged
MIN_CHANNEL_VIDEOS = 5
MIN_COMMENTS_FOR_MIX = 20
PRICE_CLUSTER_ID = 8          # "Price talk" in cluster_names.csv
N_SHOWN = 15
FEATURES = {
    "views_vs_channel": "views",
    "like_rate_vs_channel": "like rate",
    "comment_rate_vs_channel": "comment rate",
}


def add_channel_features(videos: pd.DataFrame) -> pd.DataFrame:
    """Add log10 ratios of views, like rate and comment rate to the channel typical value."""
    v = videos.copy()
    v["views"] = v["views"].clip(lower=1)
    # likes = 0 usually means hidden likes, so the like rate is treated as unknown
    v["like_rate"] = (v["likes"] / v["views"]).where(v["likes"] > 0).clip(lower=1e-5)
    v["comment_rate"] = (v["comments_count"] / v["views"]).clip(lower=1e-6)
    counts = v.groupby("channel")["video_id"].transform("count")
    for column in ("views", "like_rate", "comment_rate"):
        channel_typical = v.groupby("channel")[column].transform("median")
        typical = channel_typical.where(counts >= MIN_CHANNEL_VIDEOS, v[column].median())
        v[f"{column}_vs_channel"] = np.log10(v[column] / typical).fillna(0.0)
    return v


def add_audience_mix(v: pd.DataFrame) -> pd.DataFrame:
    """Add the share of customer-need and price comments for each video."""
    comments = pd.read_csv(COMMENTS_FILE)
    names = pd.read_csv(NAMES_FILE)
    need_ids = set(names.loc[names["cluster_type"] == "Customer need", "cluster_id"])
    sub = comments[comments["cluster_id"] >= 0]
    grouped = sub.groupby("video_id")["cluster_id"]
    mix = pd.DataFrame({
        "substantive_comments": grouped.size(),
        "need_share_pct": grouped.apply(lambda s: s.isin(need_ids).mean() * 100),
        "price_share_pct": grouped.apply(lambda s: (s == PRICE_CLUSTER_ID).mean() * 100),
    })
    mix.loc[mix["substantive_comments"] < MIN_COMMENTS_FOR_MIX,
            ["need_share_pct", "price_share_pct"]] = np.nan
    return v.merge(mix, left_on="video_id", right_index=True, how="left")


def score_videos(v: pd.DataFrame) -> pd.DataFrame:
    """Run Isolation Forest with 3 seeds and flag the most unusual videos."""
    x = v[list(FEATURES)].to_numpy()
    scores = []
    for seed in SEEDS:
        forest = IsolationForest(n_estimators=300, random_state=seed).fit(x)
        scores.append(-forest.score_samples(x))        # higher = more unusual
    v = v.copy()
    v["anomaly_score"] = np.mean(scores, axis=0)
    v["flagged"] = v["anomaly_score"] >= v["anomaly_score"].quantile(1 - FLAG_FRACTION)

    flagged_sets = [set(np.where(s >= np.quantile(s, 1 - FLAG_FRACTION))[0]) for s in scores]
    overlap = np.mean([len(a & b) / len(a | b) for a, b in combinations(flagged_sets, 2)])
    v.attrs["seed_overlap"] = overlap

    median = v[list(FEATURES)].median()
    spread = (v[list(FEATURES)] - median).abs().median() * 1.4826
    z = (v[list(FEATURES)] - median) / spread.replace(0, np.nan)
    main = z.abs().idxmax(axis=1)
    v["main_reason"] = [
        f"{FEATURES[col]} {'high' if z.loc[i, col] > 0 else 'low'} ({z.loc[i, col]:+.1f})"
        for i, col in main.items()]
    return v


def plot_anomalies(v: pd.DataFrame) -> None:
    """Scatter plot of views vs like rate (both relative to the channel)."""
    normal, flagged = v[~v["flagged"]], v[v["flagged"]]
    fig, axis = plt.subplots(figsize=(8, 5.5))
    axis.scatter(normal["views_vs_channel"], normal["like_rate_vs_channel"],
                 s=22, alpha=0.6, color="#8C8C8C", label=f"Normal videos ({len(normal)})")
    axis.scatter(flagged["views_vs_channel"], flagged["like_rate_vs_channel"],
                 s=60, marker="D", color="#C55A11", label=f"Flagged as unusual ({len(flagged)})")
    axis.axhline(0, color="#CCCCCC", linewidth=0.8)
    axis.axvline(0, color="#CCCCCC", linewidth=0.8)
    axis.set_xlabel("Views compared with the channel's typical video (log10 ratio)")
    axis.set_ylabel("Like rate compared with the channel's typical video (log10 ratio)")
    axis.set_title(f"Unusual video performance (Isolation Forest, n = {len(v)} videos)")
    axis.legend()
    fig.tight_layout()
    FIGURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_FILE, dpi=200)
    plt.close(fig)


def main() -> None:
    if FEATURES_FILE.exists() or FIGURE_FILE.exists():
        raise SystemExit("Output files already exist. Rename them manually to re-run.")
    videos = pd.read_csv(VIDEOS_FILE)
    v = score_videos(add_audience_mix(add_channel_features(videos)))
    flagged = v[v["flagged"]].sort_values("anomaly_score", ascending=False)

    print(f"Videos: {len(v)} | flagged as unusual: {len(flagged)} "
          f"(top {FLAG_FRACTION * 100:.0f}% by anomaly score)")
    print(f"Videos with hidden likes (treated as typical): {int((v['likes'] == 0).sum())}")
    print(f"Agreement between 3 random seeds (Jaccard overlap): {v.attrs['seed_overlap']:.2f}")
    show = flagged.head(N_SHOWN).copy()
    show["channel"] = show["channel"].str.slice(0, 14)
    show["title"] = show["title"].str.slice(0, 40)
    print(f"\nMost unusual videos (top {min(N_SHOWN, len(flagged))})")
    print(show[["channel", "title", "content_type", "views", "main_reason"]]
          .to_string(index=False))
    print("\nFlagged videos per channel (flagged / all videos)")
    per_channel = pd.DataFrame({"flagged": v.groupby("channel")["flagged"].sum(),
                                "all_videos": v.groupby("channel").size()})
    print(per_channel.sort_values("all_videos", ascending=False).to_string())
    print("\nAverage audience mix: flagged vs normal videos")
    print(v.groupby("flagged")[["need_share_pct", "price_share_pct"]].mean().round(1).to_string())

    v.to_csv(FEATURES_FILE, index=False, encoding="utf-8-sig")
    plot_anomalies(v)
    print(f"\nSaved: {FEATURES_FILE} and {FIGURE_FILE}")


if __name__ == "__main__":
    main()