# Requires Internet
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Function to load and preprocess the Olivetti Face Dataset
def load_and_preprocess_data():
    faces_data = datasets.fetch_olivetti_faces(shuffle=True, random_state=42)
    X = faces_data.data  # 4096 features for each face (64x64 pixel images flattened)
    y = faces_data.target  # Labels for each person (40 classes)
    return X, y

# Function to split the data into training and testing sets
def split_data(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# Function to train a Naive Bayes classifier
def train_naive_bayes(X_train, y_train):
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)
    return nb_classifier

# Function to predict and calculate accuracy
def predict_and_evaluate(nb_classifier, X_test, y_test):
    y_pred = nb_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return y_pred, accuracy

# Function to visualize a few test samples with predicted labels
def visualize_predictions(X_test, y_pred, n_samples=5):
    plt.figure(figsize=(10, 5))
    for i in range(n_samples):
        plt.subplot(1, n_samples, i + 1)
        plt.imshow(X_test[i].reshape(64, 64), cmap='gray')
        plt.title(f"Pred: {y_pred[i]}")
        plt.axis('off')
    plt.show()

# Main function to run the program
def main():
    # Load and preprocess data
    X, y = load_and_preprocess_data()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train the Naive Bayes classifier
    nb_classifier = train_naive_bayes(X_train, y_train)

    # Predict and evaluate the model
    y_pred, accuracy = predict_and_evaluate(nb_classifier, X_test, y_test)
    print(f"Accuracy of the Naive Bayes Classifier on the test set: {accuracy * 100:.2f}%")

    # Visualize predictions on the test set
    visualize_predictions(X_test, y_pred)

# Run the program
if __name__ == "__main__":
    main()