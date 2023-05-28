from k_means import k_means
from dbscan import dbscan
from hierarchical import hierarchical
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


def load_wine():
    data = pd.read_csv("data/wine.data", names=["class", "alcohol", "malic_acid", "ash", "alcalnity", "magnesium", "total_phenols", "flavanoids", "nonflavanoids_phenols", "proanthocyanins", "color_intensity", "hue", "od_of_diluted_wines", "proline"])
    #print(data)
    classes = data["class"].to_numpy()
    features = data.drop("class", axis=1).to_numpy()
    weights = generate_weights(data)
    return features, classes, weights


def load_yeast():
    data = pd.read_csv("data/yeast.data", delim_whitespace=True, names=["name", "mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc", "class"])
    #print(data)
    classes = data["class"].to_numpy()
    features = data.drop(["class", "name"], axis=1).to_numpy()
    weight_data = data.drop(["name"], axis=1)
    weights = generate_weights(weight_data)
    return features, classes, weights


def load_ecoli():
    data = pd.read_csv("data/ecoli.data", delim_whitespace=True, names=["name", "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "class"])
    #print(data)
    classes = data["class"].to_numpy()
    features = data.drop(["class", "name"], axis=1).to_numpy()
    weight_data = data.drop(["name"], axis=1)
    weights = generate_weights(weight_data)
    return features, classes, weights


def generate_weights(data):
    classes = data['class'].unique()
    feature_weights = {}
    feature_diffs = data.groupby('class').max().sub(data.groupby('class').min())
    min_diffs = feature_diffs.min()
    # Przypisz największą wagę dla cechy o najmniejszej różnicy
    max_weight = 1.0
    for feature in min_diffs.index:
        feature_weights[feature] = max_weight
        max_weight -= 1 / len(min_diffs)
    return list(feature_weights.values())


def evaluate(clusters, labels):
    for cluster in np.unique(clusters):
        labels_in_cluster = labels[clusters==cluster]
        print(f"Cluster: {cluster}")
        for label_type in np.unique(labels):
            print(f"Num of {label_type}: {np.sum(labels_in_cluster==label_type)}")
        print("\n")


def plot_clusters(data, models, results):
    pca = PCA(n_components=2)  # dimensionality reduction
    reduced_features = pca.fit_transform(data)

    le = LabelEncoder()  # converting string class names to numbers
    models_encoded = le.fit_transform(models)

    plt.subplot(121)
    num_clusters = len(np.unique(models_encoded))
    cmap_colors = colors.ListedColormap(['C{}'.format(i) for i in range(num_clusters)])  # more distinctive colours
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=models_encoded, cmap=cmap_colors)
    plt.title('Model')

    plt.subplot(122)
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=results, cmap=cmap_colors)
    plt.title('Result')
    plt.suptitle("Comparison", fontsize=16)
    plt.tight_layout()
    plt.show()


def kmeans_clustering(data, num_clusters):
    features, classes, weights = data
    intra_class_variance = []
    for i in range(100):
        assignments, centroids, error = k_means(features, num_clusters)
        intra_class_variance.append(error)
    evaluate(assignments, classes)
    plot_clusters(features, classes, assignments)
    print(f"Mean intra-class variance: {np.mean(intra_class_variance)}")


def dbscan_clustering(data, eps):
    features, classes, weights = data
    num_of_points = features.shape[1] # number of features
    assignments = dbscan(features, eps, num_of_points)
    evaluate(assignments, classes)
    plot_clusters(features, classes, assignments)


def hierarchical_clustering(data, clusters):
    features, classes, weights = data
    assignments = hierarchical(features, clusters, weights)
    evaluate(assignments, classes)
    plot_clusters(features, classes, assignments)


if __name__=="__main__":
    print("Clustering Algorithms for 3 Datasets:\n")
    # print("K-means++ Wine data set")
    # kmeans_clustering(data=load_wine(), num_clusters=3)
    # print("K-means++ Yeast data set")
    # kmeans_clustering(data=load_yeast(), num_clusters=10)
    # print("K-means++ E-coli data set")
    # kmeans_clustering(data=load_ecoli(), num_clusters=8)

    # print("DBSCAN Wine data set")
    # dbscan_clustering(data=load_wine(),eps = 425)
    # print("DBSCAN Yeast data set")
    # dbscan_clustering(data=load_yeast(), eps = 0.65)
    # print("DBSCAN E-coli data set")
    # dbscan_clustering(data=load_ecoli(), eps = 0.53)

    # print("Hierarchical Wine data set")
    # hierarchical_clustering(data=load_wine(), clusters=3)
    # print("Hierarchical Yeast data set")
    # hierarchical_clustering(data=load_yeast(), clusters=10)
    # print("Hierarchical E-coli data set")
    # hierarchical_clustering(data=load_ecoli(), clusters=8)

