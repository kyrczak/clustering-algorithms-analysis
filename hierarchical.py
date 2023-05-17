import numpy as np
from sklearn.preprocessing import normalize



def count_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

def create_distance_matrix(data):
    distance_matrix = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            distance_matrix[i][j] = count_distance(data[i], data[j])
    return np.array(distance_matrix)

def find_smallest_distance(distance_matrix):
    masked_matrix = np.ma.masked_equal(distance_matrix, 0)
    min_indices = np.where(masked_matrix == np.min(masked_matrix))[0]
    return min_indices

def update_clusters(assignments, new_cluster, index):
    assignments[index] = new_cluster
    for i in range(assignments.shape[0]):
        if assignments[i] == index:
            assignments[i] = new_cluster


def update_distance_matrix(distance_matrix, index):
    for i in range(distance_matrix.shape[0]):
        distance_matrix[i][index] = 0
        distance_matrix[index][i] = 0
def hierarchical(data, clusters):
    assignments = np.arange(0, data.shape[0])
    current_clusters = len(np.unique(assignments))
    data_scaled = normalize(data)
    distance_matrix = create_distance_matrix(data_scaled)
    while current_clusters > clusters:
        x, y = find_smallest_distance(distance_matrix)
        update_clusters(assignments, min(x, y), max(x, y))
        update_distance_matrix(distance_matrix, max(x, y))
        current_clusters = len(np.unique(assignments))
    return assignments