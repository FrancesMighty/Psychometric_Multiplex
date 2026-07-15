import numpy as np


def gaussian_entropy(cov: np.ndarray) -> float:
    """
    Gaussian entropy (up to additive constant).
    Uses log-det for numerical stability.
    """
    sign, logdet = np.linalg.slogdet(cov)

    if sign <= 0:
        raise ValueError("Covariance matrix is not positive definite")

    return 0.5 * logdet


def covariance_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute covariance matrix (variables in columns).
    """
    return np.cov(X, rowvar=False)


def o_information(X: np.ndarray) -> float:
    """
    Compute O-information for continuous variables (Gaussian assumption).

    Parameters:
        X : shape (n_samples, n_variables)
    """
    X = np.asarray(X)
    n_vars = X.shape[1]

    # Full covariance
    cov = covariance_matrix(X)
    H_joint = gaussian_entropy(cov)

    # Individual entropies
    H_individuals = np.zeros(n_vars)
    for i in range(n_vars):
        var_i = cov[i, i]
        if var_i <= 0:
            raise ValueError("Non-positive variance detected")
        H_individuals[i] = 0.5 * np.log(var_i)

    # Entropy excluding each variable
    H_excluding = np.zeros(n_vars)
    for i in range(n_vars):
        cov_minus_i = np.delete(np.delete(cov, i, axis=0), i, axis=1)
        H_excluding[i] = gaussian_entropy(cov_minus_i)

    return (n_vars - 2) * H_joint + H_individuals.sum() - H_excluding.sum()