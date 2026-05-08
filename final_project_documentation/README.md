# Constructing Public Opinion: A Case Study of the US-Israel Strikes on Iran

Code and analysis pipeline for a SoDA 501 (Social Data Analytics) final
project at Penn State, Spring 2026. The project examines how liberal and
conservative US news outlets framed the 2026 US-Israel strikes on Iran,
using BERTopic-based topic modeling and named-entity co-occurrence
networks across 432 news articles.

Authors: Kadir Cihan Duran, Kawain Lo

## Running the pipeline

The main analysis and the robustness checks are separate entry points.

### 1. Main analysis

This produces, in `outputs/`:

* `topic_info.csv` - the BERTopic topic table (id, count, name, top words)
* `topic_counts_by_stance.csv` - pivot of raw topic counts by stance
* `topic_percentages_by_stance.csv` - within-stance percentages on the
  categorized base (Topic -1 excluded)
* `network_overall.gexf` - full entity co-occurrence network
* `network_overall_with_communities.gexf` - same network with Louvain
  community ids attached as node attributes
* `network_<gpe|person|org|norp|event>.gexf` - homogeneous sub-networks
  by entity type
* `network_<liberal|conservative>.gexf` - sub-networks by media stance

And in `figures/`:

* `fig1_topic_distribution.png` - topic sizes (excluding outliers)
* `fig2_topics_by_stance.png` - heatmap of raw counts
* `fig3_topic_percentages.png` - within-stance percentages

GEXF files can be opened in Gephi for visualization.

### 2. Robustness checks

Runs two checks and writes their results to `outputs/`:

* **Multi-seed BERTopic stability** (`robustness_multiseed.csv`):
  refits the pipeline under five UMAP random states and reports
  per-seed topic-word Jaccard similarity, Adjusted Rand Index, and
  outlier-rate variation against the baseline.
* **NMF cross-method validation** (`robustness_nmf_alignment.csv`):
  fits a deterministic NMF model with the same number of substantive
  topics as BERTopic and aligns each BERTopic cluster to its
  best-matching NMF topic by top-10-word Jaccard.

## Reproducibility

The baseline run uses a fixed UMAP random state (`19880106`) so that the
BERTopic results are reproducible across machines. Spot-checking against the
multi-seed robustness check is the recommended way to confirm that any
given run has not drifted.

The named-entity extraction and co-occurrence-graph construction are
deterministic given the same spaCy model version. The Louvain
community-detection step uses a fixed `random_state` for the same
reason.

## Methods overview

The pipeline implements two complementary analyses on the same corpus.

The **topic-modeling** branch uses BERTopic with sentence-transformer
embeddings (`all-MiniLM-L6-v2`), UMAP dimensionality reduction
(`n_components=5`, fixed seed), HDBSCAN density-based clustering
(`min_cluster_size=10`), and a class-based TF-IDF representation. A
project-specific stopword list is layered on top of sklearn's English
stopwords to suppress publication-date fragments and scraped HTML
boilerplate. Documents that HDBSCAN does not assign to any cluster are
collected in a residual outlier topic (-1) and excluded from
proportional comparisons; reported percentages are computed on the
categorized base only.

The **network** branch extracts five spaCy entity types
(`GPE`, `PERSON`, `ORG`, `NORP`, `EVENT`) from each article,
deduplicates within-article repeats to avoid self-loops, and builds a
weighted undirected co-occurrence graph in which an edge between two
entities is weighted by the number of articles in which they appear
together. The same construction is applied to (a) the full corpus,
(b) each entity type in isolation, and (c) each media-stance subset
(liberal-only and conservative-only). Louvain modularity optimization
is run on the full network to surface community structure.

For full methodological detail and the rationale for these choices, see
the project paper.

