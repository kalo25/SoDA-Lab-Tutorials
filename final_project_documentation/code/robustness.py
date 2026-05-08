"""
Robustness and validation checks for the BERTopic pipeline.

Performs two checks:

  1. Multi-seed BERTopic stability: refits the pipeline under several
     UMAP random states and reports topic-word Jaccard similarity,
     Adjusted Rand Index, and outlier-rate variation against the baseline.

  2. NMF cross-method validation: fits a deterministic NMF model on a
     TF-IDF representation of the same corpus and aligns each BERTopic
     topic to its best-matching NMF topic by top-10-word Jaccard.

Results are printed to stdout and saved as CSVs under outputs/.
"""

import os
from collections import Counter

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import (CountVectorizer,
                                             ENGLISH_STOP_WORDS,
                                             TfidfVectorizer)
from sklearn.metrics import (adjusted_rand_score,
                             normalized_mutual_info_score)
from umap import UMAP


# --------------------------------------------------------------------- #
# Configuration                                                         #
# --------------------------------------------------------------------- #

INPUT_PATH = "data/iran_cons_full_labelled.csv"
OUTPUTS_DIR = "outputs"
BASELINE_SEED = 19880106
ALT_SEEDS = [19880107, 19880108, 19880109, 19880110]

# Same custom stopword list used in the main pipeline. Keep this in
# sync with analysis.py.
CUSTOM_STOPWORDS = [
    "said", "says", "told", "according", "new", "news", "just", "like",
    "time", "day", "year", "ago", "latest", "ve", "advertisement", "ad",
    "video", "com",
    "00", "10", "16", "20", "27", "28", "2024", "2025", "2026",
    "monday", "apr", "april", "march", "oct", "sep", "nov",
    "ap", "fox", "davis", "charles", "linsey", "nightline", "kimmel",
    "king", "birth",
]

os.makedirs(OUTPUTS_DIR, exist_ok=True)


# --------------------------------------------------------------------- #
# 1. Load corpus                                                        #
# --------------------------------------------------------------------- #

print("Loading and filtering data...")
df = pd.read_csv(INPUT_PATH)
df = df.dropna(subset=["full_text", "label"])
df = df.drop_duplicates(subset=["full_text"])
df = df[df["label"].isin(["liberal", "conservative"])].copy()
df = df.reset_index(drop=True)

docs = df["full_text"].tolist()
final_stopwords = list(ENGLISH_STOP_WORDS.union(CUSTOM_STOPWORDS))
print(f"Corpus: {len(df)} articles")


# --------------------------------------------------------------------- #
# 2. Check 1: multi-seed BERTopic stability                             #
# --------------------------------------------------------------------- #

print("\n" + "=" * 70)
print("CHECK 1: MULTI-SEED BERTOPIC STABILITY")
print("=" * 70)

all_seeds = [BASELINE_SEED] + ALT_SEEDS
seed_results = {}

for seed in all_seeds:
    print(f"  Fitting BERTopic with random_state={seed}...")
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                      metric="cosine", random_state=seed)
    hdbscan_model = HDBSCAN(min_cluster_size=10, metric="euclidean",
                            cluster_selection_method="eom",
                            prediction_data=True)
    vectorizer_model = CountVectorizer(stop_words=final_stopwords)

    m = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        language="english",
        verbose=False,
    )
    t, _ = m.fit_transform(docs)
    t_arr = np.array(t)

    # Collect top-10 words per substantive topic
    top_words = {}
    for tid in m.get_topics():
        if tid == -1:
            continue
        top_words[tid] = [w for w, _ in m.get_topic(tid)][:10]

    seed_results[seed] = {
        "topics": t_arr,
        "top_words": top_words,
        "n_topics": len(top_words),
        "outlier_rate": (t_arr == -1).mean(),
    }


# Compare each alternative seed against the baseline
base = seed_results[BASELINE_SEED]
rows = []
for seed in all_seeds:
    r = seed_results[seed]
    if seed == BASELINE_SEED:
        rows.append({
            "seed": seed,
            "n_topics": r["n_topics"],
            "outlier_rate": r["outlier_rate"],
            "avg_best_jaccard_vs_base": np.nan,
            "ARI_vs_base": np.nan,
            "NMI_vs_base": np.nan,
        })
        continue

    # Best-match Jaccard: for each baseline topic, find the alternate
    # seed's topic with the highest top-10 word overlap.
    best_scores = []
    for ta_words in base["top_words"].values():
        sa = set(ta_words)
        best = 0.0
        for tb_words in r["top_words"].values():
            sb = set(tb_words)
            score = len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0
            if score > best:
                best = score
        best_scores.append(best)

    rows.append({
        "seed": seed,
        "n_topics": r["n_topics"],
        "outlier_rate": r["outlier_rate"],
        "avg_best_jaccard_vs_base": np.mean(best_scores),
        "ARI_vs_base": adjusted_rand_score(base["topics"], r["topics"]),
        "NMI_vs_base": normalized_mutual_info_score(base["topics"],
                                                    r["topics"]),
    })

summary = pd.DataFrame(rows)
print("\n--- Per-seed summary ---")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
summary.to_csv(f"{OUTPUTS_DIR}/robustness_multiseed.csv", index=False)


# Detailed topic-by-topic alignment for inspection
print("\n--- Topic alignment detail (baseline vs. each alternative seed) ---")
for seed in ALT_SEEDS:
    print(f"\n  Seed {BASELINE_SEED} -> Seed {seed}:")
    alt_top_words = seed_results[seed]["top_words"]
    for ta, ta_words in sorted(base["top_words"].items()):
        sa = set(ta_words)
        best_tb, best_score = None, 0.0
        for tb, tb_words in alt_top_words.items():
            sb = set(tb_words)
            score = len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0
            if score > best_score:
                best_score, best_tb = score, tb
        base_words = ", ".join(ta_words[:5])
        alt_words = (", ".join(alt_top_words[best_tb][:5])
                     if best_tb is not None else "-")
        print(f"    T{ta} [{base_words}] -> T{best_tb} [{alt_words}]  "
              f"(Jaccard={best_score:.2f})")


# Headline numbers
valid = summary["avg_best_jaccard_vs_base"].dropna()
aris = summary["ARI_vs_base"].dropna()
print("\n--- Stability summary ---")
print(f"  Mean topic-word Jaccard across seeds: {valid.mean():.3f}")
print(f"  Mean ARI across seeds:                {aris.mean():.3f}")
print("  Reference: Jaccard > 0.5 indicates the same topic; "
      "ARI > 0.5 is strong agreement, > 0.3 moderate.")


# --------------------------------------------------------------------- #
# 3. Check 2: NMF cross-method validation                               #
# --------------------------------------------------------------------- #

print("\n" + "=" * 70)
print("CHECK 2: CROSS-METHOD VALIDATION (NMF vs. BERTOPIC)")
print("=" * 70)

# Use the baseline BERTopic top words computed above
bert_topics = base["top_words"]
n_topics = len(bert_topics)

tfidf = TfidfVectorizer(stop_words=final_stopwords, max_features=2000,
                        min_df=2, max_df=0.95)
X = tfidf.fit_transform(docs)
feature_names = tfidf.get_feature_names_out()

nmf = NMF(n_components=n_topics, random_state=BASELINE_SEED,
          max_iter=500, init="nndsvd")
nmf.fit(X)

nmf_topics = {}
print(f"\n--- NMF top words (k={n_topics}, deterministic) ---")
for k in range(n_topics):
    top_idx = nmf.components_[k].argsort()[-10:][::-1]
    nmf_topics[k] = [feature_names[i] for i in top_idx]
    print(f"  NMF Topic {k}: {nmf_topics[k]}")


# Align each BERTopic cluster to its best-matching NMF topic
print("\n--- BERTopic -> best-matching NMF topic ---")
align_rows = []
for bt_id, bt_words in sorted(bert_topics.items()):
    sa = set(bt_words)
    best_nmf, best_score = None, 0.0
    for nmf_id, nmf_words in nmf_topics.items():
        sb = set(nmf_words)
        score = len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0
        if score > best_score:
            best_score, best_nmf = score, nmf_id
    align_rows.append({
        "BERTopic_ID": bt_id,
        "BERTopic_top5": ", ".join(bt_words[:5]),
        "Best_NMF_ID": best_nmf,
        "NMF_top5": ", ".join(nmf_topics[best_nmf][:5]),
        "Jaccard": best_score,
    })

align = pd.DataFrame(align_rows)
print(align.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
align.to_csv(f"{OUTPUTS_DIR}/robustness_nmf_alignment.csv", index=False)


coverage = (align["Jaccard"] >= 0.2).mean()
mean_jacc = align["Jaccard"].mean()
print("\n--- Cross-method summary ---")
print(f"  Mean BERTopic-to-NMF Jaccard: {mean_jacc:.3f}")
print(f"  Share of BERTopic topics with Jaccard >= 0.2: {coverage:.0%}")

print(f"\nDone. Robustness CSVs saved to {OUTPUTS_DIR}/.")
