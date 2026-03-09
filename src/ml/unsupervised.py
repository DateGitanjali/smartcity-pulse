"""
Unsupervised ML — SmartCity Pulse
KMeans clustering, DBSCAN, Isolation Forest, and PCA.
"""

# import pickle to save models
import pickle

# import numpy for math
import numpy as np

# import pandas for data manipulation
import pandas as pd

# import KMeans for clustering similar weather days
from sklearn.cluster import KMeans

# import DBSCAN for density based clustering and outlier detection
from sklearn.cluster import DBSCAN

# import Isolation Forest for anomaly detection
from sklearn.ensemble import IsolationForest

# import PCA to reduce dimensions for visualization
from sklearn.decomposition import PCA

# import silhouette score to evaluate clustering quality
from sklearn.metrics import silhouette_score

# import matplotlib for plotting
import matplotlib.pyplot as plt

# import mlflow for experiment tracking
import mlflow

# import os to create folders
import os

# import sys to add file paths
import sys

# import logging
import logging

# create logger for this file
logger = logging.getLogger(__name__)

def run_kmeans(X: pd.DataFrame) -> pd.Series:
    """
    Groups weather readings into clusters of similar days.

    Args:
        X: Feature matrix from preprocessor.py

    Returns:
        Series with cluster label for each row
    """

    # log that kmeans is starting
    logger.info("Running KMeans clustering...")

    # dictionary to store silhouette scores for each k value
    scores_dict = {}

    # try k from 2 to 6 and find best number of clusters
    for k in range(2, 7):

        # create kmeans model with k clusters
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)

        # fit model and get cluster labels for each row
        labels = kmeans.fit_predict(X)

        # calculate silhouette score — higher means better clusters
        score = silhouette_score(X, labels)

        # store score with k as key
        scores_dict[k] = score

        # log score for this k
        logger.info(f"k={k} silhouette score: {score:.4f}")

    # find k that gave highest silhouette score
    best_k = max(scores_dict, key=scores_dict.get)

    # log best k found
    logger.info(f"Best k: {best_k} with score: {scores_dict[best_k]:.4f}")

    # set mlflow experiment name
    mlflow.set_experiment("smartcity-unsupervised")

    # start mlflow run for kmeans
    with mlflow.start_run(run_name="kmeans"):

        # create final kmeans model with best k
        final_kmeans = KMeans(
            n_clusters=best_k,
            n_init=10,
            random_state=42
        )

        # fit model and get final cluster labels
        labels = final_kmeans.fit_predict(X)

        # calculate final silhouette score using the function
        final_score = silhouette_score(X, labels)

        # log number of clusters to mlflow
        mlflow.log_param("n_clusters", best_k)

        # log silhouette score to mlflow
        mlflow.log_metric("silhouette_score", final_score)

        # save model to disk
        with open("models/kmeans.pkl", "wb") as f:
            pickle.dump(final_kmeans, f)

        # print best k value
        print(f"\n✅ KMeans Clustering")
        print(f"   Best k           : {best_k}")
        print(f"   Silhouette Score : {final_score:.4f}")

        # get unique cluster labels and their counts
        unique, counts = np.unique(labels, return_counts=True)

        # print size of each cluster
        print(f"   Cluster sizes    :")
        for cluster, count in zip(unique, counts):
            print(f"      Cluster {cluster} : {count} weather readings")

    # return cluster labels as pandas Series
    return pd.Series(labels, name="cluster")
def run_dbscan(X: pd.DataFrame) -> pd.Series:
    """
    Finds clusters of any shape and detects outliers.

    Args:
        X: Feature matrix from preprocessor.py

    Returns:
        Series with cluster label for each row
        (-1 means outlier/anomaly)
    """

    # log that dbscan is starting
    logger.info("Running DBSCAN...")

    # set mlflow experiment
    mlflow.set_experiment("smartcity-unsupervised")

    # start mlflow run
    with mlflow.start_run(run_name="dbscan"):

        # create DBSCAN model
        # eps=0.5 — maximum distance between two points to be neighbors
        # min_samples=3 — minimum points to form a dense region
        dbscan = DBSCAN(eps=0.5, min_samples=3)

        # fit model and get labels
        # label -1 means the point is an outlier
        labels = dbscan.fit_predict(X)

        # count how many clusters found
        # -1 is outliers so we exclude it from cluster count
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # count outliers — points with label -1
        n_outliers = list(labels).count(-1)

        # log to mlflow
        mlflow.log_param("eps", 0.5)
        mlflow.log_param("min_samples", 3)
        mlflow.log_metric("n_clusters", n_clusters)
        mlflow.log_metric("n_outliers", n_outliers)

        # print results
        print(f"\n✅ DBSCAN")
        print(f"   Clusters found : {n_clusters}")
        print(f"   Outliers found : {n_outliers}")

        # show what each label means
        print(f"   Label breakdown:")
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, counts):
            # label -1 is special — means outlier
            if label == -1:
                print(f"      Outliers  : {count} readings")
            else:
                print(f"      Cluster {label} : {count} readings")

        # log cluster info
        logger.info(f"DBSCAN found {n_clusters} clusters and {n_outliers} outliers")

    # return labels as pandas Series
    return pd.Series(labels, name="dbscan_cluster")
def run_isolation_forest(X: pd.DataFrame) -> pd.Series:
    """
    Detects anomalies in weather data.
    Finds unusual weather readings that don't fit normal patterns.

    Args:
        X: Feature matrix from preprocessor.py

    Returns:
        Series with 1=normal, -1=anomaly for each row
    """

    # log that isolation forest is starting
    logger.info("Running Isolation Forest...")

    # set mlflow experiment
    mlflow.set_experiment("smartcity-unsupervised")

    # start mlflow run
    with mlflow.start_run(run_name="isolation_forest"):

        # create Isolation Forest model
        # contamination=0.1 means we expect 10% of data to be anomalies
        # n_estimators=100 means build 100 isolation trees
        # random_state=42 for reproducibility
        iso_forest = IsolationForest(
            contamination=0.1,
            n_estimators=100,
            random_state=42
        )

        # fit model and predict
        # output: 1=normal, -1=anomaly
        labels = iso_forest.fit_predict(X)

        # count normal and anomaly points
        n_normal   = list(labels).count(1)
        n_anomaly  = list(labels).count(-1)

        # calculate anomaly percentage
        anomaly_pct = (n_anomaly / len(labels)) * 100

        # log to mlflow
        mlflow.log_param("contamination", 0.1)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("n_anomalies", n_anomaly)
        mlflow.log_metric("anomaly_pct", anomaly_pct)

        # save model to disk — useful for real time anomaly detection
        with open("models/isolation_forest.pkl", "wb") as f:
            pickle.dump(iso_forest, f)

        # print results
        print(f"\n✅ Isolation Forest")
        print(f"   Total readings : {len(labels)}")
        print(f"   Normal         : {n_normal}")
        print(f"   Anomalies      : {n_anomaly}")
        print(f"   Anomaly %      : {anomaly_pct:.1f}%")

        # get anomaly scores — more negative = more anomalous
        scores = iso_forest.decision_function(X)

        # find the most anomalous reading
        most_anomalous_idx = scores.argmin()
        print(f"   Most anomalous reading index: {most_anomalous_idx}")

        # log info
        logger.info(f"Found {n_anomaly} anomalies ({anomaly_pct:.1f}%)")

    # return labels as pandas Series
    return pd.Series(labels, name="anomaly")
def run_pca(X: pd.DataFrame, cluster_labels: pd.Series) -> None:
    """
    Reduces 9 features to 2 dimensions for visualization.
    Creates a scatter plot showing weather clusters.

    Args:
        X: Feature matrix from preprocessor.py
        cluster_labels: Cluster labels from run_kmeans()
    """

    # log that pca is starting
    logger.info("Running PCA...")

    # set mlflow experiment
    mlflow.set_experiment("smartcity-unsupervised")

    # start mlflow run
    with mlflow.start_run(run_name="pca"):

        # create PCA model
        # n_components=2 means reduce to 2 dimensions
        pca = PCA(n_components=2)

        # fit and transform — reduces 9 features to 2
        X_pca = pca.fit_transform(X)

        # explained variance — how much info is kept after reduction
        explained_var = pca.explained_variance_ratio_

        # total variance explained by 2 components
        total_variance = sum(explained_var)

        # log to mlflow
        mlflow.log_param("n_components", 2)
        mlflow.log_metric("explained_variance_pc1", explained_var[0])
        mlflow.log_metric("explained_variance_pc2", explained_var[1])
        mlflow.log_metric("total_variance_explained", total_variance)

        # save pca model to disk
        with open("models/pca.pkl", "wb") as f:
            pickle.dump(pca, f)

        # print results
        print(f"\n✅ PCA")
        print(f"   PC1 variance explained : {explained_var[0]:.4f}")
        print(f"   PC2 variance explained : {explained_var[1]:.4f}")
        print(f"   Total variance kept    : {total_variance:.4f}")

        # ── Create visualization ─────────────────────────
        # create figure with size 10x7 inches
        plt.figure(figsize=(10, 7))

        # get unique cluster labels
        unique_clusters = cluster_labels.unique()

        # color map for clusters
        colors = ["red", "blue", "green", "orange", "purple"]

        # plot each cluster with different color
        for i, cluster in enumerate(sorted(unique_clusters)):

            # get rows belonging to this cluster
            mask = cluster_labels == cluster

            # plot those rows as scatter points
            plt.scatter(
                X_pca[mask, 0],    # PC1 values
                X_pca[mask, 1],    # PC2 values
                c=colors[i],       # color for this cluster
                label=f"Cluster {cluster}",
                alpha=0.7,         # slight transparency
                s=100              # dot size
            )

        # add title and labels
        plt.title(
            f"Weather Clusters — PCA Visualization\n"
            f"Total variance explained: {total_variance:.1%}",
            fontsize=14
        )
        plt.xlabel(f"PC1 ({explained_var[0]:.1%} variance)", fontsize=12)
        plt.ylabel(f"PC2 ({explained_var[1]:.1%} variance)", fontsize=12)

        # add legend
        plt.legend(fontsize=11)

        # add grid for readability
        plt.grid(True, alpha=0.3)

        # save plot to file
        plt.savefig("models/pca_clusters.png", dpi=150, bbox_inches="tight")

        # show plot
        plt.show()

        # log image to mlflow
        mlflow.log_artifact("models/pca_clusters.png")

        # log info
        logger.info(f"PCA complete. Variance explained: {total_variance:.4f}")
# runs only when you run this file directly
if __name__ == "__main__":

    # add pipeline folder to path
    sys.path.append("src/pipeline")

    # import data fetcher to get live weather data
    from data_fetcher import CityDataFetcher

    # import preprocessor to clean and engineer features
    from preprocessor import WeatherPreprocessor

    # fetch live weather data
    print("Fetching live weather data...")
    fetcher = CityDataFetcher()
    df = fetcher.fetch_weather("Mumbai")

    # preprocess data — no target needed for unsupervised
    print("Preprocessing data...")
    preprocessor = WeatherPreprocessor()
    X, _ = preprocessor.prepare(df)

    # print data shape
    print(f"\nData ready:")
    print(f"X shape : {X.shape}")

    # create models folder
    os.makedirs("models", exist_ok=True)

    # run KMeans clustering
    print("\n" + "="*50)
    print("RUNNING KMEANS CLUSTERING")
    print("="*50)
    cluster_labels = run_kmeans(X)

    # run DBSCAN
    print("\n" + "="*50)
    print("RUNNING DBSCAN")
    print("="*50)
    dbscan_labels = run_dbscan(X)

    # run Isolation Forest
    print("\n" + "="*50)
    print("RUNNING ISOLATION FOREST")
    print("="*50)
    anomaly_labels = run_isolation_forest(X)

    # run PCA visualization
    print("\n" + "="*50)
    print("RUNNING PCA VISUALIZATION")
    print("="*50)
    run_pca(X, cluster_labels)

    # final summary
    print("\n" + "="*50)
    print("UNSUPERVISED LEARNING COMPLETE")
    print("="*50)
    print(f"\nKMeans clusters    : {cluster_labels.nunique()}")
    print(f"DBSCAN outliers    : {(dbscan_labels == -1).sum()}")
    print(f"Anomalies detected : {(anomaly_labels == -1).sum()}")
    print(f"\nPCA plot saved to  : models/pca_clusters.png")