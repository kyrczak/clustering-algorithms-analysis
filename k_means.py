import numpy as np

def initialize_centroids_kmeans_pp(data, k):
    centroids = data[np.random.choice(data.shape[0],1,replace=False)]
    for i in range(1,k):
        distance = list(map(lambda x: np.min(np.sum((x-centroids) ** 2, axis=1)),data))
        centroids = np.append(centroids,[data[np.argmax(distance)]],axis=0)
    return centroids

def assign_to_cluster(data, centroid):
    assignments = []
    for i in range(data.shape[0]):
        assignments.append(np.argmin(np.sum((data[i]-centroid)**2, axis=1)))
    return np.asarray(assignments)

def update_centroids(data, assignments):
    centroids = []
    for i in range(max(assignments)+1):
        if np.any(assignments == i):
            centroids.append(np.mean(data[assignments==i],axis=0))
    return np.asarray(centroids)

def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :])**2))

def k_means(data, num_centroids):
    # centroids initizalization
    centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    assignments  = assign_to_cluster(data, centroids)

    for _ in range(100): # max number of iteration = 100
        #print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments): # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)         

