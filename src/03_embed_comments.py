"""Stage 03a: turn substantive comments into meaning vectors (embeddings).

Input : data/processed/comments_final.csv   (read only)
        data/processed/labels_handmade.csv  (read only, used for a quick sanity check)
Output: data/processed/embeddings.npy       (one row per comment, never overwritten)
        data/processed/embedding_ids.csv    (comment_id of each row)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding="utf-8")

# ---------- Settings ----------
COMMENTS_FILE = Path("data/processed/comments_final.csv")
LABELS_FILE = Path("data/processed/labels_handmade.csv")
EMBEDDINGS_FILE = Path("data/processed/embeddings.npy")
IDS_FILE = Path("data/processed/embedding_ids.csv")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 64
NEIGHBOURS = 3
PROBE_TOPICS = ["Price", "Warranty / PTA / Authenticity",
                "Availability / Where to buy", "Launch / Release date"]


def show_neighbours(df: pd.DataFrame, embeddings: np.ndarray, labels: pd.DataFrame) -> None:
    """Print the closest comments (by meaning) for one hand-labelled example per topic."""
    row_of = {cid: i for i, cid in enumerate(df["comment_id"])}
    print("\nSanity check: nearest comments by meaning (cosine similarity)")
    for topic in PROBE_TOPICS:
        example = labels[labels["topic"] == topic]
        example = example[example["comment_id"].isin(row_of)]
        if example.empty:
            continue
        index = row_of[example.iloc[0]["comment_id"]]
        scores = embeddings @ embeddings[index]
        order = [i for i in np.argsort(-scores) if i != index][:NEIGHBOURS]
        print(f"\n  [{topic}] {df.loc[index, 'text_clean'][:85]}")
        for i in order:
            print(f"     {scores[i]:.2f}  {df.loc[i, 'text_clean'][:85]}")


def main() -> None:
    if EMBEDDINGS_FILE.exists() or IDS_FILE.exists():
        raise SystemExit("Embedding files already exist. Rename them manually to re-run.")
    comments = pd.read_csv(COMMENTS_FILE)
    comments["text_clean"] = comments["text_clean"].fillna("").astype(str)
    df = comments[comments["comment_kind"] == "substantive"].reset_index(drop=True)
    print(f"Substantive comments to embed: {len(df)} (chatter is excluded)")

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        df["text_clean"].tolist(), batch_size=BATCH_SIZE, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    print(f"\nEmbedding matrix shape: {embeddings.shape}")
    print(f"Missing values: {int(np.isnan(embeddings).sum())}")
    print(f"Average vector length: {np.linalg.norm(embeddings, axis=1).mean():.3f} (should be 1.000)")

    np.save(EMBEDDINGS_FILE, embeddings)
    df[["comment_id"]].to_csv(IDS_FILE, index=False)
    show_neighbours(df, embeddings, pd.read_csv(LABELS_FILE))
    print(f"\nSaved: {EMBEDDINGS_FILE} and {IDS_FILE}")


if __name__ == "__main__":
    main()