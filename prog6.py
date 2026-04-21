import numpy as np
import matplotlib.pyplot as plt
# Generate synthetic data
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = np.sin(X) + np.random.normal(scale=0.2, size=X.shape)  # Add some noise
# Reshape X for matrix operations
X = X[:, np.newaxis]
# X=X.reshape(-1,1) second method to reshape the array
# Locally Weighted Regression function
def locally_weighted_regression(X_train, y_train, query_point, tau):
    # Compute weights using Gaussian kernel
    weights = np.exp(-np.sum((X_train - query_point)**2, axis=1) / (2 * tau**2))
    # Create a diagonal weight matrix
    W = np.diag(weights)
    # Compute the weighted normal equation
    X_bias = np.hstack([np.ones_like(X_train), X_train])  # Add bias term
    theta = np.linalg.inv(X_bias.T @ W @ X_bias) @ X_bias.T @ W @ y_train
    # Predict for the query point
    query_point_bias = np.array([1, query_point])  # Use scalar query_point
    prediction = query_point_bias @ theta
    return prediction
# Predict for multiple points using LWR
def predict_lwr(X_train, y_train, X_test, tau):
    predictions = np.array([locally_weighted_regression(X_train, y_train, x[0], tau) for x in X_test])
    return predictions
# Hyperparameter: Bandwidth (tau)
tau = 0.5
# Predict values for the test set
y_pred = predict_lwr(X, y, X, tau)
# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label="Data Points", color="blue", s=10)
plt.plot(X, y_pred, label=f"LWR Prediction (tau={tau})", color="red", linewidth=2)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Locally Weighted Regression (LWR)")
plt.legend()
plt.grid(True)
plt.show()