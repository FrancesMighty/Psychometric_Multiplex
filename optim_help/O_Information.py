import numpy as np


def entropy(data: np.ndarray) -> float:
    """
    Shannon entropy (bits) of discrete 1D array.
    """
    data = np.asarray(data).ravel()
    _, counts = np.unique(data, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs), dtype=np.float64)


def joint_entropy(data: np.ndarray) -> float:
    """
    Joint entropy (bits) of multiple discrete variables.
    """
    data = np.asarray(data)
    _, counts = np.unique(data, axis=0, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs), dtype=np.float64)


def o_information(X: np.ndarray) -> float:
    """
    Compute O-information for DISCRETE variables (bits).
    """
    X = np.asarray(X)
    n_vars = X.shape[1]

    # Precompute
    H_joint = joint_entropy(X)
    H_individuals = np.array([entropy(X[:, i]) for i in range(n_vars)])
    H_excluding = np.array([joint_entropy(np.delete(X, i, axis=1)) for i in range(n_vars)])

    return (n_vars - 2) * H_joint + H_individuals.sum() - H_excluding.sum()
