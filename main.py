from k_means import k_means
from dbscan import dbscan, calculate_epsilon
import pandas as pd
import numpy as np

def load_wine():
    data = pd.read_csv("data/wine.data", names=["class", "alcohol", "malic_acid", "ash", "alcalnity", "magnesium", "total_phenols", "flavanoids", "nonflavanoids_phenols", "proanthocyanins", "color_intensity", "hue", "od_of_diluted_wines", "proline"])
    #print(data)
    classes = data["class"].to_numpy()
    features = data.drop("class", axis=1).to_numpy()
    return features, classes

def load_yeast():
    data = pd.read_csv("data/yeast.data", delim_whitespace=True, names=["name", "mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc", "class"])
    #print(data)
    classes = data["class"].to_numpy()
    features = data.drop(["class", "name"], axis=1).to_numpy()
    return features, classes

def load_ecoli():
    data = pd.read_csv("data/ecoli.data", delim_whitespace=True, names=["name", "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "class"])
    #print(data)
    classes = data["class"].to_numpy()
    features = data.drop(["class", "name"], axis=1).to_numpy()
    return features, classes

def load_iris():
    data = pd.read_csv("data/iris.data", names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])
    #print(data)
    classes = data["class"].to_numpy()
    features = data.drop("class", axis=1).to_numpy()
    return features, classes

def evaluate(clusters, labels):
    for cluster in np.unique(clusters):
        labels_in_cluster = labels[clusters==cluster]
        print(f"Cluster: {cluster}")
        for label_type in np.unique(labels):
            print(f"Num of {label_type}: {np.sum(labels_in_cluster==label_type)}")
        print("\n")
    

def kmeans_clustering(data, num_clusters):
    features, classes = data
    intra_class_variance = []
    for i in range(100):
        assignments, centroids, error = k_means(features, num_clusters)
        intra_class_variance.append(error)
    evaluate(assignments, classes)
    print(f"Mean intra-class variance: {np.mean(intra_class_variance)}")

def dbscan_clustering(data, eps):
    features, classes = data
    num_of_points = features.shape[1] # number of features
    assignments = dbscan(features, eps, num_of_points)
    evaluate_dbscane(assignments, classes)

def evaluate_dbscane(assignments, classes):
    for cluster in np.unique(assignments):
        labels_in_cluster = classes[assignments==cluster]
        print(f"Cluster: {cluster}")
        for label_type in np.unique(classes):
            print(f"Num of {label_type}: {np.sum(labels_in_cluster==label_type)}")
        print("\n")

if __name__=="__main__":
    # print("K-means++ Wine data set")
    # kmeans_clustering(data=load_wine(), num_clusters=3)
    # print("K-means++ Yeast data set")
    # kmeans_clustering(data=load_yeast(), num_clusters=10)
    # print("K-means++ E-coli data set")
    # kmeans_clustering(data=load_ecoli(), num_clusters=8)

    print("DBSCAN Wine data set")
    dbscan_clustering(data=load_wine(),eps = 425)
    # print("DBSCAN Yeast data set")
    # dbscan_clustering(data=load_yeast(), eps = 0.5)
    # print("DBSCAN E-coli data set")
    # dbscan_clustering(data=load_ecoli(), eps = 0.53)
