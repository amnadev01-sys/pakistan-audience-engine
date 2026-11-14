"""Pakistan Social Audience & Campaign Intelligence Engine: Streamlit dashboard.

The app only reads the files produced by the pipeline (src/01 to src/06). It never
changes any data. Run from the project root:  streamlit run app/app.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import kruskal

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RECOMMENDATIONS_FILE = ROOT / "outputs" / "recommendations.csv"
MIN_GROUP = 8                      # smallest group that is compared statistically
BAR_COLOR, ACCENT_COLOR, MUTED_COLOR = "#2a6f97", "#d1495b", "#9aa5b1"
METRICS = {"Views": "views_vs_channel", "Like rate": "like_rate_vs_channel",
           "Comment rate": "comment_rate_vs_channel"}

st.set_page_config(page_title="Pakistan Audience Intelligence Engine", layout="wide")


# ---------- Data loading ----------
@st.cache_data
def load_data():
    """Load the processed files once."""
    videos = pd.read_csv(PROCESSED / "videos_final.csv")
    comments = pd.read_csv(PROCESSED / "comments_clustered.csv")
    names = pd.read_csv(PROCESSED / "cluster_names.csv")
    recs = pd.read_csv(RECOMMENDATIONS_FILE)
    return videos, comments, names, recs
@st.cache_data
def typical_scores(comment_ids: tuple, cluster_ids: tuple) -> pd.Series:
    """Cosine similarity of each comment to the centre of its own cluster."""
    embeddings = np.load(PROCESSED / "embeddings.npy")
    ids = pd.read_csv(PROCESSED / "embedding_ids.csv")["comment_id"]
    row_of = pd.Series(np.arange(len(ids)), index=ids)
    frame = pd.DataFrame({"comment_id": comment_ids, "cluster_id": cluster_ids})
    frame = frame[frame["comment_id"].isin(row_of.index)]
    scores = pd.Series(0.0, index=frame["comment_id"].to_numpy())
    for cluster_id, part in frame.groupby("cluster_id"):
        vectors = embeddings[row_of[part["comment_id"]].to_numpy()]
        centre = vectors.mean(axis=0)
        centre /= np.linalg.norm(centre)
        scores[part["comment_id"].to_numpy()] = vectors @ centre
    return scores


def p_text(p_value: float) -> str:
    """Format a p-value for display."""
    return "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"


def group_table(videos: pd.DataFrame, group_col: str, metric_col: str) -> pd.DataFrame:
    """Median of a metric relative to the video's own channel, shown as a multiplier."""
    table = videos.groupby(group_col).agg(
        videos=("video_id", "count"),
        multiplier=(metric_col, lambda s: 10 ** s.median()))
    return table.sort_values("multiplier", ascending=False)


def bar_chart(table: pd.DataFrame, title: str, xlabel: str) -> plt.Figure:
    """Horizontal bar chart of multipliers with sample sizes in the labels."""
    fig, axis = plt.subplots(figsize=(8, max(3, 0.45 * len(table) + 1)))
    labels = [f"{name} (n = {int(n)})" for name, n in zip(table.index, table["videos"])]
    colors = [BAR_COLOR if n >= MIN_GROUP else MUTED_COLOR for n in table["videos"]]
    axis.barh(labels, table["multiplier"], color=colors)
    axis.axvline(1.0, color="black", linewidth=1, linestyle="--")
    axis.invert_yaxis()
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


# ---------- Page 1: content ----------
def page_content(videos: pd.DataFrame) -> None:
    st.header("1. Which content attracts attention?")
    st.caption("Each video is compared with the typical video of its own channel, so large "
               "channels do not dominate. A value of 1.0x means a typical video for that channel.")
    left, right = st.columns(2)
    channels = left.multiselect("Channels", sorted(videos["channel"].unique()),
                                default=sorted(videos["channel"].unique()))
    metric_name = right.selectbox("Metric", list(METRICS), index=1)
    group_name = st.radio("Group videos by", ["Content type", "Brand in title"], horizontal=True)
    data = videos[videos["channel"].isin(channels)]
    if data.empty:
        st.warning("Select at least one channel.")
        return

    group_col = "content_type" if group_name == "Content type" else "brand"
    metric_col = METRICS[metric_name]
    table = group_table(data, group_col, metric_col)
    st.pyplot(bar_chart(table, f"{metric_name} relative to channel, by {group_name.lower()}",
                        "Median multiplier of the channel's typical video (1.0x = typical)"))
    st.caption(f"Videos shown: {len(data)}. Grey bars have fewer than {MIN_GROUP} videos "
               "and should not be interpreted.")

    large = data.groupby(group_col)["video_id"].transform("count") >= MIN_GROUP
    groups = [g[metric_col].dropna() for _, g in data[large].groupby(group_col)]
    if len(groups) >= 2:
        p_value = kruskal(*groups).pvalue
        verdict = "a statistically detectable difference" if p_value < 0.05 else "no clear difference"
        st.info(f"Kruskal-Wallis test across groups with at least {MIN_GROUP} videos: "
                f"{p_text(p_value)} ({verdict}). Six related tests were run in the full "
                "analysis, so a strict threshold of about 0.008 applies there.")
    st.dataframe(table.round(2).rename_axis(group_name).rename(
        columns={"videos": "Videos", "multiplier": "Multiplier"}))


# ---------- Page 2: audience needs ----------
def page_needs(comments: pd.DataFrame, names: pd.DataFrame) -> None:
    st.header("2. What does the audience want?")
    st.caption("Comments were converted to sentence embeddings and grouped with KMeans (K = 10). "
               "Group names were assigned after reading sample comments.")
    left, right = st.columns(2)
    channels = left.multiselect("Channels", sorted(comments["channel"].unique()),
                                default=sorted(comments["channel"].unique()), key="needs_ch")
    languages = right.multiselect("Comment language", sorted(comments["language_type"].unique()),
                                  default=sorted(comments["language_type"].unique()))
    data = comments[comments["channel"].isin(channels)
                    & comments["language_type"].isin(languages) & (comments["cluster_id"] >= 0)]
    if data.empty:
        st.warning("No comments match the selected filters.")
        return

    info = names.set_index("cluster_id")
    summary = data.groupby("cluster_id").agg(
        comments=("comment_id", "count"),
        videos=("video_id", "nunique"),
        roman_urdu=("language_type", lambda s: (s == "Roman Urdu").mean() * 100))
    summary["share"] = summary["comments"] / len(data) * 100
    summary["name"] = info.loc[summary.index, "cluster_name"]
    summary["type"] = info.loc[summary.index, "cluster_type"]
    summary = summary.sort_values("share", ascending=False)

    fig, axis = plt.subplots(figsize=(8, 0.45 * len(summary) + 1.5))
    colors = [BAR_COLOR if t == "Customer need" else MUTED_COLOR for t in summary["type"]]
    axis.barh(summary["name"], summary["share"], color=colors)
    axis.invert_yaxis()
    axis.set_title(f"Audience groups in substantive comments (n = {len(data):,})")
    axis.set_xlabel("Share of comments (%)")
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("Blue: customer needs. Grey: appreciation and noise.")

    show = summary[["name", "type", "share", "comments", "videos", "roman_urdu"]].rename(columns={
        "name": "Group", "type": "Type", "share": "Share (%)", "comments": "Comments",
        "videos": "Videos", "roman_urdu": "Roman Urdu (%)"})
    st.dataframe(show.round(1), hide_index=True)

    st.subheader("Example comments")
    choice = st.selectbox("Group", summary["name"].tolist())
    cluster_id = summary.index[summary["name"] == choice][0]
    full = comments[comments["cluster_id"] >= 0]
    scores = typical_scores(tuple(full["comment_id"]), tuple(full["cluster_id"]))
    examples = data[data["cluster_id"] == cluster_id].copy()
    examples["typicality"] = examples["comment_id"].map(scores)
    examples = examples.sort_values("typicality", ascending=False)
    st.caption("The comments closest to the centre of the group, so they describe the group "
               "better than the most-liked comments, which are often viral reactions.")
    st.dataframe(examples[["text_clean", "language_type", "like_count"]].head(8).rename(columns={
        "text_clean": "Comment", "language_type": "Language", "like_count": "Likes"}),
        hide_index=True)


# ---------- Page 3: unusual videos ----------
def page_unusual(videos: pd.DataFrame) -> None:
    st.header("3. Which marketing patterns are unusual?")
    st.caption("An Isolation Forest compares each video with its own channel (views, like rate "
               "and comment rate) and flags the top 5%. Flagged videos are leads for "
               "investigation, not proof of success or failure.")
    channels = st.multiselect("Channels", sorted(videos["channel"].unique()),
                              default=sorted(videos["channel"].unique()), key="unusual_ch")
    data = videos[videos["channel"].isin(channels)]
    if data.empty:
        st.warning("Select at least one channel.")
        return
    flagged = data[data["flagged"]]
    normal = data[~data["flagged"]]

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(10 ** normal["views_vs_channel"], 10 ** normal["like_rate_vs_channel"],
                 s=14, color=MUTED_COLOR, alpha=0.6, label=f"Typical (n = {len(normal)})")
    axis.scatter(10 ** flagged["views_vs_channel"], 10 ** flagged["like_rate_vs_channel"],
                 s=40, color=ACCENT_COLOR, label=f"Flagged (n = {len(flagged)})")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_title("Views and like rate relative to each channel's typical video")
    axis.set_xlabel("Views (multiple of channel typical, log scale)")
    axis.set_ylabel("Like rate (multiple of channel typical, log scale)")
    axis.legend()
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    col1.metric("Videos shown", len(data))
    col2.metric("Flagged as unusual", len(flagged))
    st.subheader("Flagged videos")
    st.dataframe(flagged.sort_values("anomaly_score", ascending=False)[
        ["title", "channel", "content_type", "brand", "views", "main_reason"]].rename(columns={
            "title": "Title", "channel": "Channel", "content_type": "Content type",
            "brand": "Brand", "views": "Views", "main_reason": "Main reason"}),
        hide_index=True)


# ---------- Page 4: audience groups ----------
def page_groups(comments: pd.DataFrame, names: pd.DataFrame) -> None:
    st.header("4. Which audience needs should the brand address?")
    st.caption("Which customer needs dominate on each channel, and in which language the "
               "audience speaks. Only customer-need groups are shown.")
    need_ids = names.loc[names["cluster_type"] == "Customer need", "cluster_id"]
    name_of = names.set_index("cluster_id")["cluster_name"]
    need = comments[comments["cluster_id"].isin(need_ids)].copy()
    need["need"] = need["cluster_id"].map(name_of)

    top_channels = need["channel"].value_counts()
    top_channels = top_channels[top_channels >= 100].index.tolist()
    channels = st.multiselect("Channels (with at least 100 need comments)", top_channels,
                              default=top_channels[:8], key="groups_ch")
    if not channels:
        st.warning("Select at least one channel.")
        return
    mix = pd.crosstab(need["channel"], need["need"], normalize="index").loc[channels] * 100

    fig, axis = plt.subplots(figsize=(9, 0.5 * len(mix) + 2.5))
    left = pd.Series(0.0, index=mix.index)
    palette = plt.cm.tab10.colors
    for i, column in enumerate(mix.columns):
        axis.barh(mix.index, mix[column], left=left, label=column, color=palette[i % 10])
        left += mix[column]
    axis.invert_yaxis()
    axis.set_title(f"Customer-need mix by channel (n = {int(need['channel'].isin(channels).sum()):,} comments)")
    axis.set_xlabel("Share of the channel's need comments (%)")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=8)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Language of each need")
    language = pd.crosstab(need["need"], need["language_type"], normalize="index") * 100
    st.dataframe(language.round(0).rename_axis("Need"))
    st.caption("Percentages add to 100 per row. Language comes from a rule-based tag "
               "(92% accurate on 100 hand labels).")


# ---------- Page 5: recommendations ----------
def page_recommendations(recs: pd.DataFrame) -> None:
    st.header("5. What should the marketing team do next?")
    st.caption("Every recommendation is a rule with computed evidence. A rule appears only "
               "when its condition is met in the data.")
    priorities = st.multiselect("Priority", ["High", "Medium", "Low"], default=["High", "Medium"])
    data = recs[recs["priority"].isin(priorities)]
    for _, row in data.iterrows():
        with st.expander(f"{row['rank']}. [{row['priority']}] {row['area']}", expanded=row["priority"] == "High"):
            st.markdown(f"**Action:** {row['action']}")
            st.markdown(f"**Evidence:** {row['evidence']}")
            st.markdown(f"**Limitation:** {row['limitation']}")
    st.download_button("Download recommendations (CSV)", recs.to_csv(index=False).encode("utf-8-sig"),
                       "recommendations.csv", "text/csv")


# ---------- Main ----------
def main() -> None:
    try:
        videos, comments, names, recs = load_data()
    except FileNotFoundError as error:
        st.error(f"A pipeline output is missing: {error.filename}. Run the pipeline scripts first.")
        return
    st.sidebar.title("Audience Intelligence Engine")
    page = st.sidebar.radio("Question", [
        "1. Content that attracts attention", "2. What the audience wants",
        "3. Unusual videos", "4. Audience needs to address", "5. Recommendations"])
    st.sidebar.caption(f"Data: {len(videos)} YouTube videos from {videos['channel'].nunique()} "
                       f"Pakistani smartphone channels and {len(comments):,} cleaned comments.")
    if page.startswith("1"):
        page_content(videos)
    elif page.startswith("2"):
        page_needs(comments, names)
    elif page.startswith("3"):
        page_unusual(videos)
    elif page.startswith("4"):
        page_groups(comments, names)
    else:
        page_recommendations(recs)


main()