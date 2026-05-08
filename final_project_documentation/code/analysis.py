"""
US-Iran Conflict News Coverage: Topic Modeling and Network Analysis
SoDA 501 Final Project, Spring 2026

Reads data/iran_cons_full_labelled.csv and writes results to outputs/
and figures/.
"""

import os
from collections import Counter, defaultdict
from itertools import combinations

import community as community_louvain
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
import spacy
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from umap import UMAP


# --------------------------------------------------------------------- #
# Configuration                                                         #
# --------------------------------------------------------------------- #

INPUT_PATH = "data/iran_cons_full_labelled.csv"
OUTPUTS_DIR = "outputs"
FIGURES_DIR = "figures"
RANDOM_SEED = 19880106
SPACY_MODEL = "en_core_web_sm"
ENTITY_LABELS = ["GPE", "PERSON", "ORG", "NORP", "EVENT"]

CUSTOM_STOPWORDS = [
    # Generic news boilerplate
    "said", "says", "told", "according", "new", "news", "just", "like",
    "time", "day", "year", "ago", "latest", "ve", "advertisement", "ad",
    "video", "com",
    # Date and timestamp fragments that surface in scraped HTML
    "00", "10", "16", "20", "27", "28", "2024", "2025", "2026",
    "monday", "apr", "april", "march", "oct", "sep", "nov",
    # Wire-service bylines and broadcast page boilerplate identified
    # during exploratory inspection
    "ap", "fox", "davis", "charles", "linsey", "nightline", "kimmel",
    "king", "birth",
]

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# --------------------------------------------------------------------- #
# 1. Load and filter the corpus                                         #
# --------------------------------------------------------------------- #

print("Loading and filtering data...")
df = pd.read_csv(INPUT_PATH)
df = df.dropna(subset=["full_text", "label"])
df = df.drop_duplicates(subset=["full_text"])
df = df[df["label"].isin(["liberal", "conservative"])].copy()
df = df.reset_index(drop=True)

docs = df["full_text"].tolist()
classes = df["label"].tolist()

n_lib = (df["label"] == "liberal").sum()
n_con = (df["label"] == "conservative").sum()
print(f"Corpus: {len(df)} articles (liberal={n_lib}, conservative={n_con})")


# --------------------------------------------------------------------- #
# 2. Build the stopword list                                            #
# --------------------------------------------------------------------- #

final_stopwords = list(ENGLISH_STOP_WORDS.union(CUSTOM_STOPWORDS))


# --------------------------------------------------------------------- #
# 3. Fit BERTopic with a fixed UMAP seed                                #
# --------------------------------------------------------------------- #

print("Fitting BERTopic...")
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                  metric="cosine", random_state=RANDOM_SEED)
hdbscan_model = HDBSCAN(min_cluster_size=10, metric="euclidean",
                        cluster_selection_method="eom", prediction_data=True)
vectorizer_model = CountVectorizer(stop_words=final_stopwords)

topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    language="english",
    verbose=True,
)
topics, _ = topic_model.fit_transform(docs)
df["Topic"] = topics

topic_info = topic_model.get_topic_info()
topic_info.to_csv(f"{OUTPUTS_DIR}/topic_info.csv", index=False)

n_substantive = sum(1 for tid in topic_model.get_topics() if tid != -1)
n_outliers = sum(1 for t in topics if t == -1)
print(f"Found {n_substantive} substantive topics, {n_outliers} outliers "
      f"({100 * n_outliers / len(docs):.1f}%)")


# --------------------------------------------------------------------- #
# 4. Topic-modeling figures                                             #
# --------------------------------------------------------------------- #

valid_topics = topic_info[topic_info["Topic"] != -1]
valid_ids = valid_topics["Topic"].tolist()

# Figure 1: topic distribution
plt.figure(figsize=(12, 6))
sns.barplot(data=valid_topics, x="Count", y="Name", palette="viridis")
plt.title("Topics Discovered by BERTopic (Excluding Outliers)")
plt.xlabel("Number of Documents")
plt.ylabel("Topic Name")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig1_topic_distribution.png", dpi=150)
plt.close()

# Figure 2: raw topic frequency by stance
print("Computing topics-by-stance...")
topics_per_class = topic_model.topics_per_class(docs, classes=classes)
tpc_filtered = topics_per_class[topics_per_class["Topic"].isin(valid_ids)]
pivot = tpc_filtered.pivot(index="Topic", columns="Class",
                          values="Frequency").fillna(0)
pivot.to_csv(f"{OUTPUTS_DIR}/topic_counts_by_stance.csv")

plt.figure(figsize=(10, 8))
sns.heatmap(pivot, annot=True, fmt="g", cmap="Blues", linewidths=0.5)
plt.title("Raw Topic Frequency by Media Stance")
plt.xlabel("Media Stance")
plt.ylabel("Topic ID")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig2_topics_by_stance.png", dpi=150)
plt.close()

# Figure 3: within-stance percentages on the categorized base
counts = (df.groupby(["label", "Topic"]).size()
            .reset_index(name="Count"))
counts = counts[counts["Topic"] != -1]
totals = counts.groupby("label")["Count"].sum()
counts["Percentage"] = counts.apply(
    lambda row: 100 * row["Count"] / totals[row["label"]], axis=1
)
plot_data = counts[counts["Topic"].isin(valid_ids)]
plot_data.to_csv(f"{OUTPUTS_DIR}/topic_percentages_by_stance.csv", index=False)

plt.figure(figsize=(12, 6))
sns.barplot(x="Topic", y="Percentage", hue="label", data=plot_data,
            palette=["#E15759", "#4E79A7"])
plt.title("Percentage of Coverage Dedicated to Top Topics")
plt.xlabel("Topic ID")
plt.ylabel("Percentage of Categorized Coverage (%)")
plt.legend(title="Stance")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig3_topic_percentages.png", dpi=150)
plt.close()


# --------------------------------------------------------------------- #
# 5. Named-entity extraction                                            #
# --------------------------------------------------------------------- #

print(f"Loading spaCy model: {SPACY_MODEL}")
nlp = spacy.load(SPACY_MODEL)

print(f"Extracting entities from {len(df)} articles...")
entities_typed = []
for text in df["full_text"]:
    doc = nlp(text)
    seen = {(ent.text, ent.label_) for ent in doc.ents
            if ent.label_ in ENTITY_LABELS}
    entities_typed.append(list(seen))
df["entities"] = entities_typed


# --------------------------------------------------------------------- #
# 6. Overall co-occurrence network                                      #
# --------------------------------------------------------------------- #

print("Building overall network...")
edge_counter = Counter()
for entity_list in df["entities"]:
    if len(entity_list) < 2:
        continue
    names = sorted({t for t, _ in entity_list})
    edge_counter.update(combinations(names, 2))

G_overall = nx.Graph()
for (a, b), w in edge_counter.items():
    G_overall.add_edge(a, b, weight=w)

print("\n--- OVERALL NETWORK ---")
print(f"Nodes:   {G_overall.number_of_nodes()}")
print(f"Edges:   {G_overall.number_of_edges()}")
print(f"Density: {nx.density(G_overall):.4f}")

centrality = nx.degree_centrality(G_overall)
top_nodes = sorted(centrality.items(), key=lambda kv: kv[1], reverse=True)[:10]
print("Top 10 hubs:")
for node, score in top_nodes:
    print(f"  {node:<35}  centrality = {score:.4f}")

top_edges = sorted(G_overall.edges(data=True),
                   key=lambda e: e[2]["weight"], reverse=True)[:10]
print("Top 10 edges:")
for u, v, d in top_edges:
    print(f"  {u} -- {v}  (weight = {d['weight']})")


# --------------------------------------------------------------------- #
# 7. Per-entity-type sub-networks                                       #
# --------------------------------------------------------------------- #

for label in ENTITY_LABELS:
    edge_counter = Counter()
    for entity_list in df["entities"]:
        filtered = [t for t, lbl in entity_list if lbl == label]
        if len(filtered) < 2:
            continue
        edge_counter.update(combinations(sorted(set(filtered)), 2))

    G = nx.Graph()
    for (a, b), w in edge_counter.items():
        G.add_edge(a, b, weight=w)

    print(f"\n--- {label} NETWORK ---")
    if G.number_of_nodes() == 0:
        print("(empty, skipping)")
        continue
    print(f"Nodes:   {G.number_of_nodes()}")
    print(f"Edges:   {G.number_of_edges()}")
    print(f"Density: {nx.density(G):.4f}")

    centrality = nx.degree_centrality(G)
    top_nodes = sorted(centrality.items(), key=lambda kv: kv[1],
                       reverse=True)[:10]
    print("Top 10 hubs:")
    for node, score in top_nodes:
        print(f"  {node:<35}  centrality = {score:.4f}")

    top_edges = sorted(G.edges(data=True),
                       key=lambda e: e[2]["weight"], reverse=True)[:10]
    print("Top 10 edges:")
    for u, v, d in top_edges:
        print(f"  {u} -- {v}  (weight = {d['weight']})")

    nx.write_gexf(G, f"{OUTPUTS_DIR}/network_{label.lower()}.gexf")


# --------------------------------------------------------------------- #
# 8. Per-stance sub-networks                                            #
# --------------------------------------------------------------------- #

for stance in ["liberal", "conservative"]:
    edge_counter = Counter()
    for entity_list in df.loc[df["label"] == stance, "entities"]:
        if len(entity_list) < 2:
            continue
        names = sorted({t for t, _ in entity_list})
        edge_counter.update(combinations(names, 2))

    G = nx.Graph()
    for (a, b), w in edge_counter.items():
        G.add_edge(a, b, weight=w)

    print(f"\n--- {stance.upper()} NETWORK ---")
    print(f"Nodes:   {G.number_of_nodes()}")
    print(f"Edges:   {G.number_of_edges()}")
    print(f"Density: {nx.density(G):.4f}")

    centrality = nx.degree_centrality(G)
    top_nodes = sorted(centrality.items(), key=lambda kv: kv[1],
                       reverse=True)[:10]
    print("Top 10 hubs:")
    for node, score in top_nodes:
        print(f"  {node:<35}  centrality = {score:.4f}")

    nx.write_gexf(G, f"{OUTPUTS_DIR}/network_{stance}.gexf")


# --------------------------------------------------------------------- #
# 9. Louvain community detection on the overall network                 #
# --------------------------------------------------------------------- #

print("\nRunning Louvain community detection...")
partition = community_louvain.best_partition(G_overall, weight="weight",
                                             random_state=RANDOM_SEED)
overall_centrality = nx.degree_centrality(G_overall)
nx.set_node_attributes(G_overall, partition, "community")
nx.set_node_attributes(G_overall, overall_centrality, "centrality")

communities = defaultdict(list)
for node, comm_id in partition.items():
    communities[comm_id].append((node, overall_centrality[node]))
for comm_id in communities:
    communities[comm_id].sort(key=lambda kv: kv[1], reverse=True)

print(f"Detected {len(communities)} communities.")
for comm_id in sorted(communities, key=lambda c: -len(communities[c])):
    members = communities[comm_id]
    print(f"\nCommunity {comm_id} (n={len(members)}, top 10 hubs):")
    for node, score in members[:10]:
        print(f"  {node:<35}  centrality = {score:.4f}")

nx.write_gexf(G_overall,
              f"{OUTPUTS_DIR}/network_overall_with_communities.gexf")
print(f"\nPipeline complete. Outputs in {OUTPUTS_DIR}/ and {FIGURES_DIR}/.")
