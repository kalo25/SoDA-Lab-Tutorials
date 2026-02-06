#-----------------------------------------------------------------------------------
#Week 4 Text as Data HW Assignment
#Kawain Lo
#-----------------------------------------------------------------------------------

#see accompanying .txt file in github folder for conceptual question response and interpretations of the homework exercises

#Setup

#pip install numpy pandas matplotlib seaborn scikit-learn gensim sentence-transformers bertopic umap-learn hdbscan wordcloud pyLDAvis

import os
import re
import random
import tarfile
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import ast
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
import hdbscan
from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.lda_model
import seaborn as sns
from sklearn.manifold import TSNE

#-----------------------------------------------------------------------------------
#PROMPT 1: LDA Topic Model
#Load the movie corpus (e.g., data_raw/week_movie_corpus.csv) and create a document--term matrix using CountVectorizer.
  #Fit an LDA model with a chosen number of topics (e.g., K=6).
  #Report the top 8--12 words for each topic.
  #Assign each document a dominant topic and create a plot showing the number of documents per dominant topic.
  #In 3--8 sentences, interpret at least two topics and explain how preprocessing choices (e.g., stopwords, min_df, token pattern) affect topic quality.

#--------------------------------------------------------------------------------

df = pd.read_csv("C:/Users/karra/Desktop/Coding_work/soda_501/04_text_as_data/demo/data_raw/week_movie_corpus.csv")

vectorizer = CountVectorizer(
    lowercase=True,
    stop_words="english", #removes all meaningless filler words
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",  # words with >=2 letters
    min_df=5
)

X_counts = vectorizer.fit_transform(df["text"])
vocab = vectorizer.get_feature_names_out()

print("\n--- Document-term matrix (counts) ---")
print("Shape:", X_counts.shape)  # (n_docs, n_terms)
print("Vocabulary size:", len(vocab))
print("Example vocab terms:", vocab[:20])

# Top terms by total count (quick diagnostic)
term_totals = np.asarray(X_counts.sum(axis=0)).ravel() #identifies how many times a specific word (from your vocab list) is found
top_idx = term_totals.argsort()[::-1][:15]
top_terms = pd.DataFrame({"term": vocab[top_idx], "total_count": term_totals[top_idx]})
print("\n--- Top terms by total count ---")
print(top_terms)


n_topics = 10
lda = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=123,
    learning_method="batch"
)
lda.fit(X_counts)

# Topic-word distributions
topic_word = lda.components_  # shape: (K, n_terms)

print("\n--- LDA topics: top words ---") #tells python to print you the top 12 words (by frequency) in your data
n_top_words = 12
for k in range(n_topics):
    top_word_idx = topic_word[k].argsort()[::-1][:n_top_words]
    words = vocab[top_word_idx]
    weights = topic_word[k][top_word_idx]
    print(f"\nTopic {k}:")
    for w, wt in zip(words, weights):
        print(f"  {w:15s} {wt:,.2f}")

# Document-topic proportions
doc_topic = lda.transform(X_counts)  # shape: (n_docs, K)
df_lda = df.copy()
df_lda["lda_topic"] = doc_topic.argmax(axis=1)
df_lda["lda_topic_prob"] = doc_topic.max(axis=1)

print("\n--- LDA: dominant topic counts ---")
print(df_lda["lda_topic"].value_counts().sort_index())

df_lda.to_csv("C:/Users/karra/Desktop/Coding_work/soda_501/04_text_as_data/problem_set/LDA_homework.csv", index=False)

topic_counts = df_lda["lda_topic"].value_counts().sort_index()
plt.figure(figsize=(8, 4))
plt.bar(topic_counts.index.astype(str), topic_counts.values)
plt.title("LDA: Dominant Topic Counts (Movie Plots)")
plt.xlabel("Dominant topic")
plt.ylabel("Number of documents")
plt.tight_layout()
plt.savefig("C:/Users/karra/Desktop/Coding_work/soda_501/04_text_as_data/problem_set/LDA_homework_plot.png", dpi=200)
plt.show()
plt.close()


lda_display = pyLDAvis.lda_model.prepare(
    lda, 
    X_counts, 
    vectorizer, 
    mds='tsne'
)
pyLDAvis.save_html(lda_display, 'C:/Users/karra/Desktop/Coding_work/soda_501/04_text_as_data/problem_set/lda_homework_visualization.html')

#------------------------------------------------------------------------------
#PROMPT 2:BERTopic: transformer embeddings --> clustering --> topic summaries.
#Using the BERTopic section of the tutorial:
  #Fit a BERTopic model on the movie plots and generate a topic summary table.
  #Create a plot showing the number of documents assigned to each topic (excluding outliers if present).
  #Report (i) the number of topics discovered (excluding outliers) and (ii) the share of documents assigned to the outlier topic (Topic = -1), if applicable.
  #In 3--10 sentences, compare BERTopic to LDA on this dataset. Discuss topic interpretability, sensitivity to preprocessing, and computational cost. Identify one setting where BERTopic is likely to outperform LDA and one where LDA may be preferable.
#-------------------------------------------------------------------------------

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(df["text"].tolist(), show_progress_bar=True)

print("\n--- Transformer embedding matrix ---")
print("Shape:", embeddings.shape)

umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=123
)

hdbscan_model = hdbscan.HDBSCAN( #this model is used to find/identify clusters of words
    min_cluster_size=5,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True
)

topic_model = BERTopic( #combines the two models used above
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    calculate_probabilities=True,
    verbose=True
)

topics, probs = topic_model.fit_transform(df["text"].tolist(), embeddings) #gives probabilities of topics

df_bert = df.copy()
df_bert["bertopic_topic"] = topics
df_bert["bertopic_max_prob"] = np.max(probs, axis=1)

print("\n--- BERTopic: topic counts ---")
print(pd.Series(topics).value_counts().sort_index()) #generates total number of topics in the dataset (by groups)
#one specific topic always represents random noise in the dataset (labelled as "-1", indicates that the words could not be labelled as part of any specific topic)

topic_info = topic_model.get_topic_info()
print("\n--- BERTopic: topic info (head) ---")
print(topic_info.head(10))

df_bert.to_csv("C:/Users/karra/Desktop/Coding_work/soda_501/04_text_as_data/demo/data_processed/week_with_bertopic.csv", index=False)
topic_info.to_csv("C:/Users/karra/Desktop/Coding_work/soda_501/04_text_as_data/demo/outputs/week_bertopic_topic_info.csv", index=False)

topic_counts_bt = topic_info.loc[topic_info["Topic"] != -1, ["Topic", "Count"]]
plt.figure(figsize=(8, 4))
plt.bar(topic_counts_bt["Topic"].astype(str), topic_counts_bt["Count"])
plt.title("BERTopic: Topic Counts (Excluding Outliers)")
plt.xlabel("Topic")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("C:/Users/karra/Desktop/Coding_work/soda_501/04_text_as_data/demo/figures/week_bertopic_topic_counts.png", dpi=200)
plt.show()
plt.close()

fig_hierarchy = topic_model.visualize_hierarchy()
fig_hierarchy.show()
fig_hierarchy.write_html("C:/Users/karra/Desktop/Coding_work/soda_501/04_text_as_data/demo/outputs/topic_hierarchy.html")

fig_docs = topic_model.visualize_documents(df["text"].tolist(), embeddings=embeddings)
fig_docs.show()
fig_docs.write_html("C:/Users/karra/Desktop/Coding_work/soda_501/04_text_as_data/demo/outputs/document_map.html")


#---------------------------------------------------------------------------
#Extension assignment
#In-Class Extension: Multinomial Outcome (Required for Participation).}
#In the tutorial, the outcome variable 'y_outcome' is binary. Extend this analysis to a multinomial outcome using movie genres.
  #Redefine the outcome variable so that each document belongs to one of several genre categories (e.g., Action, Comedy, Drama, Horror, Other).
  #Replace the regression model with an appropriate multinomial model.
  #Evaluate performance using accuracy and a confusion matrix.
#-------------------------------------------------------------
tokenized_docs = []
for text in df["text"].tolist():
    tokens = re.findall(r"[a-z]+", text.lower())
    tokenized_docs.append(tokens)

print("\n--- Tokenization check ---")
print("Example tokens:", tokenized_docs[0][:20])

w2v = Word2Vec(
    sentences=tokenized_docs,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=1,  # skip-gram
    seed=123
)

print("\n--- Word2Vec vocabulary size ---")
print(len(w2v.wv.index_to_key))

doc_vectors = [] ##the following section of code converts word vectors into document vectors (groups them all together into fewer objects)
for tokens in tokenized_docs:
    tokens_in_vocab = []
    for t in tokens:
        if t in w2v.wv:
            tokens_in_vocab.append(t)

    if len(tokens_in_vocab) == 0:
        doc_vec = np.zeros(w2v.vector_size)
    else:
        vecs = w2v.wv[tokens_in_vocab]
        doc_vec = vecs.mean(axis=0)

    doc_vectors.append(doc_vec)

doc_vectors = np.vstack(doc_vectors)

print("\n--- Document embedding matrix ---")
print("Shape:", doc_vectors.shape)

true_genre = df["true_topic"].tolist()

# Map genre labels to integers
genre_to_label = {"action": 0, "comedy": 1, "drama": 2, "horror": 3, "other": 4}
y_multiclass = list(map(genre_to_label.get, true_genre))

# Split data
X_train, X_test, y_train, y_test = train_test_split(embeddings, y_multiclass)

# Fit multinomial model

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


multinomial_logit = LogisticRegression(
    solver='lbfgs',             
    max_iter=500,
    random_state=123
)

multinomial_logit.fit(X_train, y_train) 

# Predict and evaluate
y_pred = multinomial_logit.predict(X_test)

#computing accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#computing confusion matrix
from sklearn.metrics import confusion_matrix
import numpy as np

# Sample true and predicted labels (for example, fraud detection)
cm = confusion_matrix(y_multiclass, y_pred)
print(cm)

#generates a regression plot
plt.figure(figsize=(8, 6))
sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.xlabel("Actual Outcome (y_test)")
plt.ylabel("Predicted Outcome (y_pred)")
plt.title("Multinomial Regression: Actual vs. Predicted Values")
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("C:/Users/karra/Desktop/Coding_work/soda_501/04_text_as_data/problem_set/word2vec_regression_hw.png", dpi=200)
plt.show()

