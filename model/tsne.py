import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

def compute_pairwise_distances(X):
    # Compute the squared Euclidean distance between each pair of points
    return np.sum((X[None, :] - X[:, None])**2, axis=2)

def compute_conditional_probabilities(distances, sigmas):
    # Compute the conditional probabilities based on distances and sigmas
    exp_term = np.exp(-distances / (2 * np.square(sigmas.reshape((-1, 1)))))
    np.fill_diagonal(exp_term, 0.)
    exp_term += 1e-8  # Prevent division by zero
    return exp_term / exp_term.sum(axis=1).reshape([-1, 1])

def compute_perplexity(conditional_probs):
    # Compute the perplexity of the conditional probability matrix
    entropy = -np.sum(conditional_probs * np.log2(conditional_probs), axis=1)
    return 2 ** entropy

def binary_search(func, target, tolerance=1e-10, max_iterations=1000, lower_bound=1e-20, upper_bound=10000):
    # Perform a binary search to find the sigma that achieves the target perplexity
    for _ in range(max_iterations):
        guess = (upper_bound + lower_bound) / 2.
        value = func(guess)

        if value > target:
            upper_bound = guess
        else:
            lower_bound = guess

        if np.abs(value - target) <= tolerance:
            return guess

    warnings.warn(f"Binary search couldn't find the target, returning {guess} with value {value}")
    return guess

def find_optimal_sigmas(distances, target_perplexity):
    # Find the optimal sigmas for each point to achieve the target perplexity
    optimal_sigmas = np.zeros(distances.shape[0])
    for i in range(distances.shape[0]):
        func = lambda sigma: compute_perplexity(compute_conditional_probabilities(distances[i:i+1, :], np.array([sigma])))
        optimal_sigmas[i] = binary_search(func, target_perplexity)
    return optimal_sigmas

def compute_joint_probabilities(Y):
    # Compute the joint probabilities of the low-dimensional points
    distances = compute_pairwise_distances(Y)
    numerator = 1 / (1 + distances)
    np.fill_diagonal(numerator, 0.)
    return numerator / np.sum(numerator)

def compute_gradient(P, Q, Y):
    # Compute the gradient of the Kullback-Leibler divergence
    (n, num_dims) = Y.shape
    pq_diff = P - Q
    y_diff = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)

    distances = compute_pairwise_distances(Y)
    aux_term = 1 / (1 + distances)
    return 4 * (np.expand_dims(pq_diff, 2) * y_diff * np.expand_dims(aux_term, 2)).sum(1)

def momentum(t):
    # Define the momentum term based on the iteration number
    return 0.5 if t < 250 else 0.8

def compute_joint_probabilities_high_dim(X, perplexity):
    # Compute the joint probabilities for the high-dimensional data
    N = X.shape[0]
    distances = compute_pairwise_distances(X)
    sigmas = find_optimal_sigmas(distances, perplexity)
    conditional_probs = compute_conditional_probabilities(distances, sigmas)
    return (conditional_probs + conditional_probs.T) / (2. * N)

def tsne(X, y_dim=2, num_iterations=1000, learning_rate=500, perplexity=30):
    # Perform t-SNE to reduce the dimensionality of the dataset
    N = X.shape[0]
    P = compute_joint_probabilities_high_dim(X, perplexity)

    Y = []
    y = np.random.normal(loc=0.0, scale=1e-4, size=(N, y_dim))
    Y.append(y); Y.append(y)

    for t in range(num_iterations):
        Q = compute_joint_probabilities(Y[-1])
        grad = compute_gradient(P, Q, Y[-1])
        y = Y[-1] - learning_rate * grad + momentum(t) * (Y[-1] - Y[-2])
        Y.append(y)
        if t % 10 == 0:
            Q = np.maximum(Q, 1e-12)
    return y

# Load the digits dataset
X, y = load_digits(return_X_y=True)

# Run t-SNE on the dataset
result = tsne(X, num_iterations=1000, learning_rate=200, perplexity=40)

# Plot the results
plt.scatter(result[:, 0], result[:, 1], s=20, c=y)
plt.show()
