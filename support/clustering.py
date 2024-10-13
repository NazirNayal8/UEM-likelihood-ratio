


def cluster_kmeans(vectors, num_clusters):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(vectors)
    return kmeans.cluster_centers_, kmeans.labels_

# write a function that applies hdbscan clustering on the vectors
def cluster_hdbscan(vectors, min_cluster_size=15):
    import hdbscan
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    clusterer.fit(vectors)
    return clusterer.labels_

# write a function that applies dbscan clustering on the vectors
def cluster_dbscan(vectors, eps=0.5, min_samples=5):
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(vectors)
    return clustering.labels_

# write a function that applies hierarchical clustering on the vectors
def cluster_hierarchical(vectors, distance_threshold=10):
    from sklearn.cluster import AgglomerativeClustering
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold).fit(vectors)
    return clustering.labels_



    