import numpy as np
from matplotlib import pyplot as plt

def calculate_distance(data, point):
    return np.sqrt(np.sum((data - point)**2, axis=1))

# Define method for calculating the epsilon value for given data for DBSCAN
def calculate_epsilon(data, minpoints):
    distances = []
    for i in range(data.shape[0]):
        distances.append(np.sort(calculate_distance(data, data[i]))[minpoints])
    plt.plot(np.sort(distances))
    plt.show()
    print(np.sort(distances)[minpoints])
    return np.sort(distances)[minpoints]

def dbscan(data, eps, minpoints):
    cluster = 0
    assignments = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        if assignments[i] != 0:
            continue
        distances = calculate_distance(data, data[i])
        neighbors = np.where(distances <= eps)[0]
        if len(neighbors) < minpoints:
            assignments[i] = -1
        else:
            cluster += 1
            assignments[i] = cluster
            for j in neighbors:
                if assignments[j] == -1:
                    assignments[j] = cluster
                if assignments[j] != 0:
                    continue
                assignments[j] = cluster
                distances = calculate_distance(data, data[j])
                new_neighbors = np.where(distances <= eps)[0]
                if len(new_neighbors) >= minpoints:
                    neighbors = np.append(neighbors, new_neighbors)
    return assignments