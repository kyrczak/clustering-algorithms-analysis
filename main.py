from k_means import k_means
import pandas as pd
import numpy as np

def load_wine():
    data = pd.read_csv("data/wine.data", names=["class","alcohol", "malic_acid", "ash", "alcalnity", "magnesium", "total_phenols", "flavanoids", "nonflavanoids_phenols", "proanthocyanins", "color_intensity", "hue", "od_of_diluted_wines", "proline"])
    #print(data)
    classes = data["class"].to_numpy()
    features = data.drop("class",axis=1).to_numpy()
    return features,classes

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
    

def clustering(kmeans_pp):
    data = load_wine()
    features, classes = data
    intra_class_variance = []
    for i in range(100):
        assignments, centroids, error = k_means(features, 3, kmeans_pp)
        evaluate(assignments, classes)
        intra_class_variance.append(error)
    print(f"Mean intra-class variance: {np.mean(intra_class_variance)}")

if __name__=="__main__":
    print("K-means++")
    clustering(kmeans_pp=True)
    #print("Forgy")
    #clustering(kmeans_pp=False)
