import numpy as np
from sklearn.preprocessing import normalize


def count_distance(a, b, weights):
    a_weighted = np.multiply(a, weights)
    b_weighted = np.multiply(b, weights)
    return np.sqrt(np.sum((a_weighted - b_weighted)**2))


def create_distance_matrix(data, weights):
    distance_matrix = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            distance_matrix[i][j] = count_distance(data[i], data[j], weights)
    return np.array(distance_matrix)


def find_smallest_distance(distance_matrix):
    masked_matrix = np.ma.masked_equal(distance_matrix, 0)
    min_value_index = np.unravel_index(np.argmin(masked_matrix), masked_matrix.shape)
    return min_value_index


def update_clusters(assignments, new_cluster, index):
    assignments[index] = new_cluster
    for i in range(assignments.shape[0]):
        if assignments[i] == index:
            assignments[i] = new_cluster


def update_distance_matrix(distance_matrix, index):
    for i in range(distance_matrix.shape[0]):
        distance_matrix[i][index] = 0
        distance_matrix[index][i] = 0



def fix_indexes(assignments):
    unique_indexes = list(set(assignments))
    x = len(unique_indexes)
    indexes = {index: i for i, index in enumerate(unique_indexes)}
    changed_indexes = [indexes[index] for index in assignments]
    return changed_indexes


def hierarchical(data, clusters, weights):
    assignments = np.arange(0, data.shape[0])
    current_clusters = len(np.unique(assignments))
    data_scaled = normalize(data)
    distance_matrix = create_distance_matrix(data_scaled, weights)
    while current_clusters > clusters:
        x, y = find_smallest_distance(distance_matrix)
        update_clusters(assignments, min(x, y), max(x, y))
        update_distance_matrix(distance_matrix, max(x, y))
        current_clusters = len(np.unique(assignments))
    assignments = fix_indexes(assignments)
    return assignments
