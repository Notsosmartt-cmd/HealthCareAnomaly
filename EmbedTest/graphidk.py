import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# 1. Load & preprocess
df = pd.read_csv("diabetes.csv")
X = df.drop(columns=["Outcome"])
X_scaled = MinMaxScaler().fit_transform(X)

# 2. Embed & build graph (as before)
pca = PCA(n_components=8)
emb = pca.fit_transform(X_scaled)
sim = cosine_similarity(emb)
threshold = 0.7

G = nx.Graph()
G.add_nodes_from(range(len(emb)))
for i, j in zip(*np.where(sim >= threshold)):
    if i < j:
        G.add_edge(i, j, weight=sim[i,j])

nbrs = NearestNeighbors(n_neighbors=5).fit(emb)
_, inds = nbrs.kneighbors(emb)
for i, neigh in enumerate(inds):
    for j in neigh[1:]:
        G.add_edge(i, j)

# 3. Pick a layout for the full graph
pos = nx.spring_layout(G, seed=42)  # can also try kamada_kawai_layout, spectral_layout, etc.

# 4. Draw it
plt.figure(figsize=(10,10))
nx.draw_networkx_edges(G, pos, alpha=0.1, edge_color='gray')
nx.draw_networkx_nodes(
    G, pos,
    node_size=30,
    node_color=[df.loc[n, "Outcome"] for n in G.nodes()],
    cmap=plt.cm.coolwarm,
    alpha=0.8
)
plt.title("Patient Similarity Graph (spring layout)\nNode color = Outcome (0=No, 1=Yes)")
plt.axis('off')
plt.show()
