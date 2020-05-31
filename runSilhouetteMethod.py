from scipy.spatial.distance import cdist, pdist
from sklearn import cluster
from sklearn.cluster import KMeans
from simsearch import SimSearch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import silhouette_score

# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
(ksearch, ssearch) = SimSearch.load(save_dir='mhc_corpus/')

X=ssearch.index.index

sil= np.array([])
sil2= []

k=100
kmax=255

# be automatically 3.
for k in range(190, kmax+1,5):
  kmeans = KMeans(n_clusters = k).fit(X)
  labels = kmeans.labels_
  sil2.append(silhouette_score(X, labels, metric = 'euclidean'))


sil=np.array(sil2)

plt.plot(range(190, kmax + 1, 5), sil)  # for line graph
plt.show()

index_max = np.argmax(sil)
print(index_max)