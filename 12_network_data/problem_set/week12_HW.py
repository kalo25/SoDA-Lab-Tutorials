###################################################################################
#WEEK 12 HOMEWORK--NETWORK ANALYSIS
#Kawain Lo
###################################################################################
#CONCEPTUAL QUESTION
#------------------------------------------------------------------
#Graph Neural Networks (GNN's) combine node features and network structure
#via message passing. Explain conceptually why using the graph can improve prediction
#beyond a feature-only model. Then, identify one way a GNN can fail or mislead in
#social science settings (ex: homophily-driven overconfidence, label
#leakage, missing data, boundary specification). 
#---------------------------------------------------------------------
#RESPONSE BELOW: 
#A GNN uses both graph structure and node features to predict what a network looks like by going through an
#iterative process of each node exchanging information with its neighbors to determine its true position
#in the network. This process also accounts for the issue of dependencies/high correlation among nodes within
#the network. On the other hand, a feature-only model solely relies on static data that contains 
#no information on the structure of the overall network and cannot account for dependencies among nodes.
#A GNN is good for predicting a network that reflects the data it is trained on. This does not mean
#that a GNN's output reflects the real world. For example, definitions of nodes, ties, weights, etc. 
#in a network are entirely created by the researcher when specifying parameters/boundaries. If the researcher
#makes an incorrect assumption or mis-labels an attribute, the GNN will reproduce these mistakes in
#its own model. 
####################################################################################
#SETUP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as sp

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_rand_score

import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import date

# Reproducibility
np.random.seed(123)
torch.manual_seed(123)
#######################################################################################
#PROMPT 1: GRAPH CONSTRUCTION AND CENTRALITY
#using the provided Python script, generate the synthetic stochastic bloack model (SBM)
#network and save (or print) a basic summary: number of nodes, number of edges,
#and density.
#------------------------------------------------------------------------------------
block_sizes = [400, 350, 250] 
num_blocks = len(block_sizes)
num_nodes = sum(block_sizes)
P = np.array([
    [0.06, 0.01, 0.005],  
    [0.01, 0.05, 0.008],
    [0.005, 0.008, 0.04]
])

G = nx.stochastic_block_model(block_sizes, P, seed=123)

# Basic graph summary
num_edges = G.number_of_edges()
print("Synthetic graph summary:")
print("  Nodes:", num_nodes) 
print("  Edges:", num_edges)

#Create an edge list (u, v) and reconstruct the graph from the edge list. 
true_labels = np.concatenate([
    np.zeros(block_sizes[0], dtype=int),
    np.ones(block_sizes[1], dtype=int),
    2 * np.ones(block_sizes[2], dtype=int) 
])


num_features = 8

centers = np.array([
    [2, 0, 0, 1, 0, 0, 1, 0],   # community 0 center
    [0, 2, 0, 0, 1, 0, 0, 1],   # community 1 center
    [0, 0, 2, 0, 0, 1, 1, 0]    # community 2 center
], dtype=float)

noise_sd = 0.8

X = np.zeros((num_nodes, num_features), dtype=float) #true labels (0, 1, 2) are treated as outcome vars
X[true_labels == 0] = centers[0] + np.random.normal(0, noise_sd, size=(np.sum(true_labels == 0), num_features))
X[true_labels == 1] = centers[1] + np.random.normal(0, noise_sd, size=(np.sum(true_labels == 1), num_features))
X[true_labels == 2] = centers[2] + np.random.normal(0, noise_sd, size=(np.sum(true_labels == 2), num_features))

edge_list = pd.DataFrame(list(G.edges()), columns=["u", "v"]) #edge-list = link between a sender and receiver
print("\nEdge list (first 10 rows):")
print(edge_list.head(10))

G2 = nx.from_pandas_edgelist(edge_list, source="u", target="v", create_using=nx.Graph()) #converting edge list table into a graph

print("\nReconstructed graph summary:")
print("  Nodes:", G2.number_of_nodes())
print("  Edges:", G2.number_of_edges())


plt.figure(figsize=(10, 10))

# Layout for the graph
pos = nx.spring_layout(G2, seed=123, k=0.15)

# Draw faint edges first
nx.draw_networkx_edges(
    G2,
    pos,
    alpha=0.08,
    width=0.4
)

# Draw nodes on top, colored by true community
nx.draw_networkx_nodes(
    G2,
    pos,
    node_color=true_labels,
    node_size=18,
    alpha=0.9
)

plt.title("Synthetic network spaghetti plot")
plt.axis("off")
plt.tight_layout()
plt.show() 

#Compute three centrality measures: 
#Create one figure that shows the distribution of each centrality measure (3 histograms)
    #degree or degree centrality
deg = np.array([d for _, d in G2.degree()])
deg_summary = pd.Series(deg).describe()
print("\nDegree summary:")
print(deg_summary)

top_10_deg = sorted(G2.degree(), key=lambda x: x[1], reverse=True)[:10]
for node, degree in top_10_deg:
    print(node, degree)

plt.figure()
plt.hist(deg, bins=40)
plt.title("Degree distribution (synthetic SBM)") #degree = how many nodes each node is connected to (how many ties exist for each node)
plt.xlabel("Degree")
plt.ylabel("Count of nodes") #graph shows the distribution of the number of ties that nodes have
plt.tight_layout() #this is NOT a natural distribution b/c network ties are generally highly skewed in real life
plt.show()

    #approximate between centrality (using sampling),
betw = nx.betweenness_centrality(G2, k=200, seed=123)
betw_values = np.array(list(betw.values()))

top_10_bet = sorted(betw.items(), key=lambda x: x[1], reverse=True)[:10]

for node, score in top_10_bet:
    print(node, score)

plt.figure()
plt.hist(betw_values, bins=40)
plt.title("Approximate betweenness centrality distribution")
plt.xlabel("Betweenness (approx)") #which node connects separate groups/communities (which node acts as a middleman)
plt.ylabel("Count of nodes")
plt.tight_layout()
plt.show()

    #eigenvector centrality (weighted average for betweenness--nodes are considered important if they are connected to other important nodes)
eig = nx.eigenvector_centrality_numpy(G2)
eig_values = np.array(list(eig.values()))

top_10_eig = sorted(eig.items(), key=lambda x:x[1], reverse=True)[:10]
for node, score in top_10_eig:
    print(node, score)

plt.figure()
plt.hist(eig_values, bins=40)
plt.title("Eigenvector centrality distribution")
plt.xlabel("Eigenvector centrality")
plt.ylabel("Count of nodes")
plt.tight_layout()
plt.show()

#-------------------------------------------
#Report the top 10 nodes under each measure and discuss in 4-8 sentences: do the rankings
#agree? What kind of "importance" does each metric capture in this synthetic network?
#-------------------------------------------
#Degree Centrality: The top 10 nodes are #394, 352, 129, 5, 390, 197, 291, 236, 530, and 285,
#with number of ties ranging from 41 to 46. This measure simply tells us the number of ties
#each node has.

#Between Centrality: The top 10 nodes are #530, 352, 570, 252, 129, 394, 280, 390, 291, and 179, in order
#from highest to lowest betweenness score. This measure tells us which nodes are the "middlemen"--they
#are situated on the shortest paths between other nodes. These nodes likely serve as bridges between groups/clusters.

#Eigenvector Centrality: The top 10 nodes are #394, 352, 5, 285, 146, 323, 129, 291, 197, and 390,
#in order from highest to lowest eigenvalue. This measure tells us which nodes are connected to the 
#"most important" nodes; i.e., the most influential or powerful nodes in the network.

#There is not much overlap between the three measures. The only nodes in common across all three
# measures are 394, 390, 291, and 352.

#-----------------------------------
############################################################################################
#PROMPT 2: COMMUNITY DETECTION AND EVALUATION
#########################################################################################
#Run Louvain community detection and report:
louvain_comms = nx.algorithms.community.louvain_communities(G2, seed=123) #measures latent structures in a community network

louvain_labels = np.zeros(num_nodes, dtype=int)

comm_id = 0
for comm in louvain_comms:
    for node in comm:
        louvain_labels[node] = comm_id
    comm_id = comm_id + 1

    #the number of detected communities #there are 3 communities 
print("\nCommunity detection summary:")
print("  Number of Louvain communities:", len(louvain_comms))

    #the sizes of the communities #the community sizes are 400 nodes, 350 nodes, and 250 nodes respectively
louvain_sizes = [len(c) for c in louvain_comms]
print("  Louvain community sizes (first 10):", louvain_sizes[:10])

#compute the ARI (Adjusted Rand Index) comparing Louvain communities to the known SBM
#ground-truth labels
ari = adjusted_rand_score(true_labels, louvain_labels)
print("\nAdjusted Rand Index (Louvain vs truth):", ari) #ARI = 1.0

#create ONE visualization that communicates the community structure result (like a bar plot
#of community sizes)
plt.figure()
plt.bar(range(len(louvain_sizes)), louvain_sizes)
plt.title("Louvain Community Sizes")
plt.xlabel("Community ID")
plt.ylabel("Number of Nodes")
plt.xticks(range(len(louvain_sizes)))
plt.tight_layout()
plt.show()

#-----------------------------------------------------------
#In 5-8 sentences, interpret what the ARI means and give one reason why community detection
#might split, or merge true blocks (even when the data are generated from an SBM)
#------------------------------------------------------------
#ARI is a measure that compares the Louvain-generated communities to the actual "ground truth" communities
#in the network we collected data from (or synthesized data from). In this case, the ARI value is 1.0,
#indicating a perfect match with the ground truth/SBM network. This was possible because we generated
#the synthetic SBM network using clear parameters and limited variation, which produced a dataset with neatly
#defined communities/groups. In a real-life case, Louvain might be much less accurate at identifying discrete
#communities because ties might be looser, the distance between nodes in specific clusters might be smaller, 
#or ties may vary in weight and importance. These variations would all muddle the network structure, making
#it more difficult to detect which groups are distinct and which groups should be merged. 

##################################################################################################
#PROMPT 3: Node Classification: features-only baseline v.s. GCN model
#Using the node features and ground truth labels in the tutorial:
########################################################################################################
#Fit a features-only baseline model (logistic regression) and report validation and test accuracy
perm = np.random.permutation(num_nodes)

train_size = int(0.60 * num_nodes)
val_size   = int(0.20 * num_nodes)
test_size  = num_nodes - train_size - val_size

train_idx = perm[:train_size]
val_idx   = perm[train_size:train_size + val_size]
test_idx  = perm[train_size + val_size:]

y = true_labels.copy()
lr = LogisticRegression(max_iter=200)
lr.fit(X[train_idx], y[train_idx])

yhat_val_lr  = lr.predict(X[val_idx])
yhat_test_lr = lr.predict(X[test_idx])

val_acc_lr  = accuracy_score(y[val_idx], yhat_val_lr)
test_acc_lr = accuracy_score(y[test_idx], yhat_test_lr)

print("\nBaseline (features-only) logistic regression:")
print("  Validation accuracy:", val_acc_lr) 
print("  Test accuracy:", test_acc_lr)
            #There are two validation accuracy scores b/c of training data model vs testing data (predictive) model.
            #Validation accuracy = 0.945. Testing accuracy = 0.975

#train the 2-layer GCN-style model from the tutorial and report validation and test accuracy

A = nx.to_scipy_sparse_array(
    G2,
    nodelist=np.arange(num_nodes),
    format="csr",
    dtype=np.float32
)
print("\nAdjacency matrix:") #this is a 1000x1000 matrix because the original network data had 1000 nodes
print("  Shape:", A.shape)
print("  Nonzeros:", A.nnz)

I = sp.eye(num_nodes, format="csr", dtype=np.float32)
A_tilde = A + I

deg_tilde = np.array(A_tilde.sum(axis=1)).flatten()
deg_inv_sqrt = 1.0 / np.sqrt(deg_tilde)

D_inv_sqrt = sp.diags(deg_inv_sqrt.astype(np.float32), format="csr")
A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt

X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.long)

# Convert A_norm to a torch sparse tensor
A_norm_coo = A_norm.tocoo()
A_indices = torch.tensor(
    np.vstack((A_norm_coo.row, A_norm_coo.col)),
    dtype=torch.long
)
A_values = torch.tensor(A_norm_coo.data, dtype=torch.float32)
A_norm_t = torch.sparse_coo_tensor(A_indices, A_values, size=(num_nodes, num_nodes)).coalesce()
train_idx_t = torch.tensor(train_idx, dtype=torch.long)
val_idx_t   = torch.tensor(val_idx, dtype=torch.long)
test_idx_t  = torch.tensor(test_idx, dtype=torch.long)

hidden_dim = 16
num_classes = num_blocks

lin1 = nn.Linear(num_features, hidden_dim)
lin2 = nn.Linear(hidden_dim, num_classes)

optimizer = torch.optim.Adam(list(lin1.parameters()) + list(lin2.parameters()), lr=0.01, weight_decay=5e-4)

epochs = 30

epoch_list = []
loss_history = []
train_acc_history = []
val_acc_history = []
test_acc_history = []

for epoch in range(1, epochs + 1):

    # Forward pass
    H0 = lin1(X_t)
    H1 = torch.sparse.mm(A_norm_t, H0)
    H1 = torch.relu(H1)

    Z0 = lin2(H1)
    logits = torch.sparse.mm(A_norm_t, Z0)

    # Loss on training nodes only
    loss = F.cross_entropy(logits[train_idx_t], y_t[train_idx_t])

    # Backprop + update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Accuracy (train/val/test)
    preds = torch.argmax(logits, dim=1)

    train_acc = (preds[train_idx_t] == y_t[train_idx_t]).float().mean().item()
    val_acc   = (preds[val_idx_t]   == y_t[val_idx_t]).float().mean().item()
    test_acc  = (preds[test_idx_t]  == y_t[test_idx_t]).float().mean().item()

#reporting validation and test accuracy:
    print("Epoch", epoch, "| loss:", float(loss.detach().cpu().numpy()),
          "| train_acc:", train_acc, "| val_acc:", val_acc, "| test_acc:", test_acc)
    

    # Save values for plotting after training
    epoch_list.append(epoch)
    loss_history.append(float(loss.detach().cpu().numpy()))
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)
    test_acc_history.append(test_acc)


#create a confusion table/confusion matrix for the test set for the GCN model

preds_np = preds.detach().cpu().numpy()

test_truth = y[test_idx]
test_pred_gcn = preds_np[test_idx]

test_acc_gcn = accuracy_score(test_truth, test_pred_gcn)

print("\nModel comparison:")
print("  Baseline LR test accuracy:", test_acc_lr) #accuracy: 0.975
print("  GCN test accuracy:", test_acc_gcn) #accuracy: 0.99

confusion_like = pd.crosstab(
    pd.Series(test_truth, name="True"),
    pd.Series(test_pred_gcn, name="Predicted")
)

print("\nGCN confusion table (test set):")
print(confusion_like)
print(confusion_like)
#--------------------------------------------------------------
#in 6-10 sentences, compare the baseline and GCN results. Your discussions must include
        #why the GCN can outperform the baseline in this setting
        #one reason the baseline might be competitive (or even better) in some settings,
        #one caution about interpreting "high accuracy" as scientific validity in a social network prediction task.
#---------------------------------------------------------------
#The GCN model is more accurate than the baseline logistic regression model because it takes more characteristics
#into account when generating the network structure. A GCN factors both network structure and between-node
#homophily (in which nodes with similar characteristics are more likely to cluster together) into the prediction,
#which means it can capture a more accurate representation of the original network.
#The baseline logistic regression may be more accurate in cases where the network has little or no
#homophily. Since the GCN assumes that the network has homophily, it may artifically inflate the density of ties
#or misrepresent the connections between nodes in the model output.
#High accuracy does not necessarily equate high validity--they are two separate concepts. For example,
#a predictive model can generate outcome probabilities with 99% accuracy, but the output it gives would
#be useless if it was trained on flawed data in the first place. Scientific validity requires that a dataset
#accurately reflect real-world conditions, among other criteria. In this example, if data was not collected via
#random sampling, one demographic group may be overrepresented or one social network may be considered
#more influential than it actually is in real life. A predictive model would take this flawed data and turn
#it into flawed outputs. 