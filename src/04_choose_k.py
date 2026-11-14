"""Stage 04a: choose the number of clusters (K) for the comment embeddings.

For each K the script runs KMeans with 3 random seeds and reports:
  - silhouette score (how well separated the groups are, sample of comments)
  - stability (average agreement between the 3 seeds, adjusted Rand index)
  - size of the smallest and largest cluster
  - agreement with the 100 hand-labelled topics (rough check only)

Input : data/processed/embeddings.npy, embedding_ids.csv, labels_handmade.csv (read only)
Output: data/processed/cluster_search.csv, data/processed/cluster_assignments.csv
        outputs/figures/k_selection.png   (never overwritten)
"""
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             silhouette_score)

sys.stdout.reconfigure(encoding="utf-8")

# ---------- Settings ----------
EMBEDDINGS_FILE = Path("data/processed/embeddings.npy")
IDS_FILE = Path("data/processed/embedding_ids.csv")
LABELS_FILE = Path("data/processed/labels_handmade.csv")
SEARCH_FILE = Path("data/processed/cluster_search.csv")
ASSIGN_FILE = Path("data/processed/cluster_assignments.csv")
FIGURE_FILE = Path("outputs/figures/k_selection.png")
K_VALUES = [6, 8, 10, 12]
SEEDS = [42, 7, 123]
N_INIT = 5
SILHOUETTE_SAMPLE = 5000


def plot_results(results: pd.DataFrame, n_comments: int) -> None:
    """Save a two-panel chart: silhouette and stability for each K."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    panels = [("silhouette", "Silhouette score (higher is better)"),
              ("stability_ari", "Stability across 3 seeds (1.0 = identical)")]
    for axis, (column, label) in zip(axes, panels):
        axis.plot(results["k"], results[column], marker="o", color="#1F3864")
        axis.set_xlabel("Number of clusters (K)")
        axis.set_ylabel(label)
        axis.set_xticks(results["k"])
        axis.grid(alpha=0.3)
    fig.suptitle(f"Choosing K for comment clusters (n = {n_comments:,} comments)")
    fig.tight_layout()
    FIGURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_FILE, dpi=200)
    plt.close(fig)


def main() -> None:
    if SEARCH_FILE.exists() or ASSIGN_FILE.exists() or FIGURE_FILE.exists():
        raise SystemExit("Cluster search files already exist. Rename them manually to re-run.")
    embeddings = np.load(EMBEDDINGS_FILE)
    ids = pd.read_csv(IDS_FILE)["comment_id"]
    if len(ids) != len(embeddings):
        raise SystemExit("embedding_ids.csv and embeddings.npy have different lengths.")
    n_comments = len(ids)

    labels = pd.read_csv(LABELS_FILE).set_index("comment_id")["topic"]
    labelled = np.where(ids.isin(labels.index))[0]
    topics = ids.iloc[labelled].map(labels).to_numpy()
    print(f"Comments: {n_comments} | hand-labelled comments found: {len(labelled)}")

    rows, assignments = [], {"comment_id": ids}
    for k in K_VALUES:
        fits = [KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit_predict(embeddings)
                for seed in SEEDS]
        stability = np.mean([adjusted_rand_score(a, b) for a, b in combinations(fits, 2)])
        main_labels = fits[0]
        silhouette = silhouette_score(embeddings, main_labels, random_state=42,
                                      sample_size=min(SILHOUETTE_SAMPLE, n_comments - 1))
        sizes = np.bincount(main_labels) / n_comments * 100
        rows.append({
            "k": k, "silhouette": round(silhouette, 3),
            "stability_ari": round(stability, 3),
            "smallest_cluster_pct": round(sizes.min(), 1),
            "largest_cluster_pct": round(sizes.max(), 1),
            "agreement_with_topics": round(
                adjusted_mutual_info_score(topics, main_labels[labelled]), 3),
        })
        assignments[f"cluster_k{k}"] = main_labels
        print(f"  K = {k} done")

    results = pd.DataFrame(rows)
    print("\nCluster search results")
    print(results.to_string(index=False))
    print("\nagreement_with_topics: adjusted mutual information with the 100 hand labels "
          "(0 = none, 1 = perfect; small sample, rough guide only)")

    results.to_csv(SEARCH_FILE, index=False)
    pd.DataFrame(assignments).to_csv(ASSIGN_FILE, index=False)
    plot_results(results, n_comments)
    print(f"\nSaved: {SEARCH_FILE}, {ASSIGN_FILE} and {FIGURE_FILE}")


if __name__ == "__main__":
    main()