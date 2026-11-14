"""Stage 06b: turn the evidence into rule-based marketing recommendations (Q5).

Every recommendation is a rule: a fixed action plus numbers that are computed from the
data. If a rule's condition is not met, the recommendation is simply not created.

Input : data/processed/comments_clustered.csv, cluster_names.csv, cluster_search.csv,
        videos_final.csv                          (read only)
Output: outputs/recommendations.csv               (never overwritten)
"""
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import kruskal

sys.stdout.reconfigure(encoding="utf-8")

# ---------- Settings ----------
COMMENTS_FILE = Path("data/processed/comments_clustered.csv")
NAMES_FILE = Path("data/processed/cluster_names.csv")
SEARCH_FILE = Path("data/processed/cluster_search.csv")
VIDEOS_FILE = Path("data/processed/videos_final.csv")
OUTPUT_FILE = Path("outputs/recommendations.csv")
K = 10
MIN_GROUP = 8
MIN_BRAND_VIDEOS = 10
GENERIC_BRANDS = {"Multiple", "No brand in title", "Minor brands"}

# One fixed action per customer-need cluster (cluster ids from cluster_names.csv).
NEED_ACTIONS = {
    1: "Publish budget-based 'which phone should I buy' guides (for example under 30k, "
       "50k and 100k) in Roman Urdu, and answer selection questions with a link to the "
       "matching guide.",
    3: "Create head-to-head brand content (for example Samsung vs Motorola) and put brand "
       "names in titles, because viewers already discuss these brands.",
    4: "Run a gaming-phone series with real FPS tests and a PTA / non-PTA checklist, and "
       "collect review requests in a pinned poll.",
    5: "Add where-to-buy details, store link, warranty and PTA status to every video "
       "description and pinned comment, written in Roman Urdu.",
    6: "Offer English comparison summaries (a spec table in the description) for viewers "
       "who compare devices.",
    8: "State the Pakistan price (PTA and non-PTA) in the first seconds of each video and "
       "keep a pinned price comment up to date.",
}


def p_text(p_value: float) -> str:
    """Format a p-value for text."""
    return "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"


def need_recommendations(comments: pd.DataFrame, names: pd.DataFrame,
                         n_videos: int, search: pd.DataFrame) -> list:
    """One recommendation for each customer-need cluster, ranked by size."""
    sub = comments[comments["cluster_id"] >= 0]
    row_k = search[search["k"] == K].iloc[0]
    limitation = (f"Clusters overlap (silhouette {row_k['silhouette']:.3f}) and K = {K} had a "
                  f"stability of {row_k['stability_ari']:.2f} across random seeds.")
    found = []
    for _, row in names[names["cluster_type"] == "Customer need"].iterrows():
        cluster_id = int(row["cluster_id"])
        part = sub[sub["cluster_id"] == cluster_id]
        if cluster_id in NEED_ACTIONS and not part.empty:
            found.append({"id": cluster_id, "name": row["cluster_name"],
                          "share": len(part) / len(sub) * 100,
                          "videos": part["video_id"].nunique(),
                          "roman": (part["language_type"] == "Roman Urdu").mean() * 100})
    found.sort(key=lambda item: -item["share"])
    return [{
        "priority": "High" if rank < 2 else "Medium",
        "area": f"Audience need: {item['name']}",
        "action": NEED_ACTIONS[item["id"]],
        "evidence": (f"{item['share']:.1f}% of substantive comments; present in "
                     f"{item['videos']} of {n_videos} videos; {item['roman']:.0f}% Roman Urdu."),
        "limitation": limitation,
    } for rank, item in enumerate(found)]


def format_recommendation(videos: pd.DataFrame) -> list:
    """Which content type gets the best audience response relative to its channel."""
    big = videos[videos.groupby("content_type")["video_id"].transform("count") >= MIN_GROUP]
    groups = [g for _, g in big.groupby("content_type")]
    if len(groups) < 2:
        return []
    p_like = kruskal(*[g["like_rate_vs_channel"] for g in groups]).pvalue
    p_views = kruskal(*[g["views_vs_channel"] for g in groups]).pvalue
    table = big.groupby("content_type").agg(
        videos=("video_id", "count"),
        like_x=("like_rate_vs_channel", lambda s: 10 ** s.median()))
    table = table[table["videos"] >= 20]
    if len(table) < 2 or p_like >= 0.05:
        return []
    best, worst = table["like_x"].idxmax(), table["like_x"].idxmin()
    views_note = "but not in views" if p_views >= 0.05 else "and also in views"
    return [{
        "priority": "High" if p_like < 0.01 else "Medium",
        "area": "Content format",
        "action": (f"Use {best} videos when the goal is audience response, and treat "
                   f"{worst} videos mainly as information or reach content."),
        "evidence": (f"Like rate is {table.loc[best, 'like_x']:.2f}x the channel's typical "
                     f"video for {best} (n = {table.loc[best, 'videos']}) versus "
                     f"{table.loc[worst, 'like_x']:.2f}x for {worst} (n = {table.loc[worst, 'videos']}). "
                     f"Content types differ in like rate ({p_text(p_like)}), {views_note} "
                     f"({p_text(p_views)})."),
        "limitation": "Content type comes from title keywords, and channel style is only "
                      "controlled by comparing each video with its own channel's typical video.",
    }]


def brand_recommendation(videos: pd.DataFrame) -> list:
    """Which brands bring reach, and whether the audience response keeps up."""
    named = videos[~videos["brand"].isin(GENERIC_BRANDS)]
    named = named[named.groupby("brand")["video_id"].transform("count") >= MIN_BRAND_VIDEOS]
    if named["brand"].nunique() < 3:
        return []
    p_views = kruskal(*[g["views_vs_channel"] for _, g in named.groupby("brand")]).pvalue
    table = named.groupby("brand").agg(
        videos=("video_id", "count"),
        views_x=("views_vs_channel", lambda s: 10 ** s.median()),
        like_x=("like_rate_vs_channel", lambda s: 10 ** s.median()),
        comment_x=("comment_rate_vs_channel", lambda s: 10 ** s.median()))
    top = table.sort_values("views_x", ascending=False).head(2)
    if p_views >= 0.05:
        return []
    lagging = (top["like_x"] < 1).any()
    action = (f"Use {' and '.join(top.index)} content for reach"
              + (", and add a clear call to action (a question or poll) because response per "
                 "view is lower." if lagging else "."))
    evidence = "; ".join(f"{brand}: {r.views_x:.2f}x views, {r.like_x:.2f}x like rate, "
                         f"{r.comment_x:.2f}x comment rate (n = {int(r.videos)})"
                         for brand, r in top.iterrows())
    return [{
        "priority": "Medium", "area": "Brand focus", "action": action,
        "evidence": f"{evidence}. Brands differ in views ({p_text(p_views)}).",
        "limitation": "Brand comes from the video title, and brand effects are mixed with "
                      "content type and launch timing.",
    }]


def launch_recommendation(videos: pd.DataFrame) -> list:
    """Brands whose videos are flagged as unusual much more often than average."""
    overall = videos["flagged"].mean() * 100
    named = videos[~videos["brand"].isin(GENERIC_BRANDS)]
    stats = named.groupby("brand").agg(videos=("video_id", "count"),
                                       flagged=("flagged", "sum"))
    stats = stats[stats["videos"] >= MIN_BRAND_VIDEOS]
    stats["rate"] = stats["flagged"] / stats["videos"] * 100
    top = stats[stats["rate"] >= 2 * overall].sort_values("rate", ascending=False).head(2)
    if top.empty:
        return []
    evidence = "; ".join(f"{brand}: {int(r.flagged)} of {int(r.videos)} videos ({r.rate:.0f}%) "
                         "flagged as unusual" for brand, r in top.iterrows())
    return [{
        "priority": "Medium", "area": "Launch timing",
        "action": (f"Plan launch-week content for {' and '.join(top.index)}: unusual spikes "
                   "cluster around these brands, so prepare unboxing, price and first-look "
                   "videos before the launch."),
        "evidence": f"{evidence}, versus {overall:.0f}% of all {len(videos)} videos.",
        "limitation": f"Only {int(videos['flagged'].sum())} videos were flagged, and a spike may "
                      "come from one launch event covered by several creators.",
    }]


def measurement_and_language(comments: pd.DataFrame, names: pd.DataFrame) -> list:
    """How to measure demand, and which language to write in."""
    type_of = names.set_index("cluster_id")["cluster_type"]
    kind = comments["cluster_id"].map(type_of).fillna("Chatter")
    shares = kind.value_counts(normalize=True) * 100
    need = comments[kind == "Customer need"]
    roman = (need["language_type"] == "Roman Urdu").mean() * 100
    english = (need["language_type"] == "English").mean() * 100
    return [{
        "priority": "Medium", "area": "Measurement",
        "action": "Measure audience demand only from customer-need comments; do not use "
                  "total comment counts as a demand signal.",
        "evidence": (f"Only {shares.get('Customer need', 0):.1f}% of {len(comments):,} clean "
                     f"comments express a customer need; {shares.get('Appreciation', 0):.1f}% "
                     f"are praise or greetings, {shares.get('Noise', 0):.1f}% are noise and "
                     f"{shares.get('Chatter', 0):.1f}% were flagged as chatter."),
        "limitation": "Cluster types were named by hand after reading cluster samples.",
    }, {
        "priority": "Medium", "area": "Language",
        "action": "Write titles, pinned comments and replies in Roman Urdu, and keep English "
                  "model names and keywords.",
        "evidence": (f"{roman:.0f}% of customer-need comments are Roman Urdu and "
                     f"{english:.0f}% are English."),
        "limitation": "Language comes from a rule-based tag (92% accurate on 100 hand labels).",
    }]


def main() -> None:
    if OUTPUT_FILE.exists():
        raise SystemExit(f"{OUTPUT_FILE} already exists. Rename it manually to re-run.")
    comments = pd.read_csv(COMMENTS_FILE)
    names = pd.read_csv(NAMES_FILE)
    search = pd.read_csv(SEARCH_FILE)
    videos = pd.read_csv(VIDEOS_FILE)

    recs = (need_recommendations(comments, names, len(videos), search)
            + format_recommendation(videos) + brand_recommendation(videos)
            + launch_recommendation(videos) + measurement_and_language(comments, names))
    table = pd.DataFrame(recs)
    table["order"] = table["priority"].map({"High": 0, "Medium": 1, "Low": 2})
    table = table.sort_values("order", kind="stable").drop(columns="order").reset_index(drop=True)
    table.insert(0, "rank", range(1, len(table) + 1))

    print(f"Marketing recommendations ({len(table)} rules fired)\n")
    for _, row in table.iterrows():
        print(f"{row['rank']}. [{row['priority']}] {row['area']}")
        print(f"   Action    : {row['action']}")
        print(f"   Evidence  : {row['evidence']}")
        print(f"   Limitation: {row['limitation']}\n")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()