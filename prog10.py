import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Function to load and preprocess the dataset
def load_and_preprocess_data():
    # Load the Wisconsin Breast Cancer dataset
    data = load_breast_cancer()

    # Extract features (X) and labels (y)
    X = data.data
    y = data.target

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# Function to apply K-Means clustering
def apply_kmeans(X, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)

    # Return the cluster centers and labels
    return kmeans.cluster_centers_, kmeans.labels_

# Function to visualize the clustering result
def visualize_clusters(X, labels, centers):
    plt.figure(figsize=(8, 6))

    # Plot the data points, colored by cluster label
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)

    # Plot the cluster centers
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label='Centroids')

    plt.title('K-Means Clustering on Wisconsin Breast Cancer Dataset')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

# Main function to execute the program
def main():
    # Load and preprocess the data
    X, y = load_and_preprocess_data()

    # Apply K-Means clustering (using 2 clusters for simplicity)
    centers, labels = apply_kmeans(X, n_clusters=2)

    # Visualize the clustering results
    visualize_clusters(X, labels, centers)

# Run the program
if __name__ == "__main__":
    main()
