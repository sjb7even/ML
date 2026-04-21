import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# Load the Breast Cancer dataset
def load_and_preprocess_data():
    cancer_data = load_breast_cancer()
    X = cancer_data.data
    y = cancer_data.target
    feature_names = cancer_data.feature_names
    target_names = cancer_data.target_names
    return X, y, feature_names, target_names

# Train a Decision Tree classifier
def train_decision_tree(X_train, y_train):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf

# Plot the Decision Tree
def plot_decision_tree(clf, feature_names):
    plt.figure(figsize=(12, 8))
    plot_tree(clf, filled=True, feature_names=feature_names, class_names=["Malignant", "Benign"], rounded=True, proportion=True)
    plt.title("Decision Tree for Breast Cancer Classification")
    plt.show()

# Classify a new sample
def classify_new_sample(clf, sample):
    sample = np.array(sample).reshape(1, -1)
    prediction = clf.predict(sample)
    return prediction

# Main function to demonstrate Decision Tree Algorithm
def main():
    # Load and preprocess data
    X, y, feature_names, target_names = load_and_preprocess_data()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Decision Tree model
    clf = train_decision_tree(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Print performance metrics
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Plot the decision tree
    plot_decision_tree(clf, feature_names)

    # Classify a new sample (e.g., random sample from the test set)
    sample = X_test[0]  # You can replace this with any sample
    prediction = classify_new_sample(clf, sample)
    print(f"\nClassified sample: {'Benign' if prediction == 1 else 'Malignant'}")

# Run the main function
if __name__ == "__main__":
    main()