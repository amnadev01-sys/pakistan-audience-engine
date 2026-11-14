# Pakistan Social Audience & Campaign Intelligence Engine

A clustering-based analytics pipeline and Streamlit dashboard that turns public YouTube
comments and video metrics from Pakistani smartphone channels into evidence-based
marketing decisions.

## Business problem

Marketing teams receive thousands of comments such as "Price please?", "Where can I buy
this?" or "PTA approved hai?". Counting positive and negative comments does not show
which customer needs sit inside this feedback. This project groups similar audience
behaviour, flags unusual videos and converts the findings into concrete recommendations.

## What the dashboard answers

| Question | Method | Dashboard page |
|---|---|---|
| Which content attracts attention? | Channel-relative engagement, Kruskal-Wallis tests | 1 |
| What does the audience want? | Sentence embeddings + KMeans (K = 10) | 2 |
| Which patterns are unusual? | Isolation Forest on channel-relative metrics | 3 |
| Which audience needs should the brand address? | Need mix by channel and language | 4 |
| What should marketing do next? | Rule-based recommendations with computed evidence | 5 |

## Data

- **Source:** official YouTube Data API v3 (public videos and comments only).
- **Scope:** 352 videos from 10 Pakistani smartphone-review channels, 31,376 collected
  comments, 29,561 after cleaning, 24,683 substantive comments used for clustering.
- **Channel selection:** channels were verified using the channel country field and
  Pakistan-specific comment vocabulary; channels with Indian audience signals were removed.
- **Not included:** Instagram, TikTok and Facebook were not used because their data is
  not available through permitted public access. YouTube does not expose share counts.
- **Redistribution:** the collected comments are not included in this repository.
  Re-collect them with your own API key (see below).

## Method

1. **Collection:** videos per channel, then up to 50 top-relevance and 50 newest comments
   per video, to reduce top-comment bias.
2. **Cleaning:** removal of emoji-only, very short, promotional and duplicate comments.
   Greeting and subscribe chatter is flagged, not deleted.
3. **Hand labels:** 100 comments labelled manually (topic and language) for validation.
4. **Language tag:** rule-based English / Roman Urdu / Urdu detector (92% agreement with
   the hand labels). The tag describes clusters and is not an input to clustering.
5. **Clustering:** `paraphrase-multilingual-MiniLM-L12-v2` embeddings and KMeans.
   K = 10 was chosen using silhouette, seed stability and cluster sizes.
6. **Profiling:** channel lift, language mix, distinctive words and the comments nearest
   to each cluster centre; clusters were named by hand from these.
7. **Anomalies:** Isolation Forest on log ratios of views, like rate and comment rate
   relative to each video's own channel; the top 5% are flagged.
8. **Content analysis:** rule-based content type and brand from titles; group
   differences tested with Kruskal-Wallis.
9. **Recommendations:** each rule fires only when its condition is met in the data.

## Key findings

- **Audience needs:** 53.8% of clean comments express a customer need. The largest need
  groups are buying decisions and availability (13.6% of substantive comments), device
  opinions and comparisons (11.8%), Samsung and Motorola brand talk (11.5%), gaming and
  PTA requests (11.2%) and phone selection questions (10.8%).
- **Language:** about 62% of need comments are in Roman Urdu.
- **Content:** content type does not change views (p = 0.519) but changes like rate
  (p < 0.001). Buying guides reach 1.13x the channel's typical like rate, explainers 0.81x.
- **Brands:** Apple and Motorola videos receive about 1.40x typical views, but Apple's
  like and comment rates are lower than typical.
- **Unusual videos:** 18 of 352 videos were flagged, mostly launch and unboxing videos
  for Tecno and Oppo.

The full list of 11 recommendations is in `outputs/recommendations.csv` and on page 5
of the dashboard.

## Limitations

- **Sample size:** 352 videos from 10 channels and one platform. Several content-type
  groups have only 10 to 11 videos and are shown for context only.
- **Cluster quality:** clusters overlap (silhouette 0.022; K = 10 stability 0.78 across
  seeds) and partly separate by language. Agreement with hand topics is weak (AMI 0.05
  to 0.07). Cluster names are interpretations; for example, the Roman Urdu buying group
  also contains viewer-to-creator conversation.
- **Content type and brand:** derived from title keywords, so some videos are
  misclassified. Effects are associations, not causes.
- **Language tag:** 92% agreement on only 100 hand labels, so the true accuracy is
  uncertain. Roman Urdu has no standard spelling.
- **Multiple tests:** six related tests were run, so a strict threshold of about
  p < 0.008 is applied in the analysis.
- **Platform scope:** results describe YouTube smartphone audiences and may not transfer
  to fashion or other categories or to other platforms.

## Repository structure

```
app/            Streamlit dashboard (reads processed files only)
src/            Pipeline scripts, one stage per file
data/           raw/ and processed/ (not tracked in Git)
outputs/        Figures, cluster profiles and recommendations
```

Pipeline order: `01_find_channels` -> `01_verify_channels` -> `01_collect_videos` ->
`01_collect_comments` -> `02_clean` -> `02b_flag_chatter` -> `02c_make_label_sheet` ->
`02d_import_labels` -> `02e_fix_language` -> `03_embed_comments` -> `04_choose_k` ->
`04_profile_clusters` -> `04_name_clusters` -> `05_anomaly` -> `06_content_analysis` ->
`06_recommend`.

Scripts never overwrite existing outputs; move an old output to a backup folder before
re-running a stage. Hand labels are never regenerated.

## Setup

Requires Python 3.12.

```powershell
git clone https://github.com/amnadev01-sys/pakistan-audience-engine.git
cd pakistan-audience-engine
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the project root (never commit it):

```
YOUTUBE_API_KEY=your_key_here
```

## Run

Run the pipeline stages in the order above, for example `python src\02_clean.py`, then
start the dashboard from the project root:

```powershell
streamlit run app\app.py
```

## Tech stack

Python, pandas, NumPy, SciPy, scikit-learn, sentence-transformers, matplotlib,
Streamlit, YouTube Data API v3.