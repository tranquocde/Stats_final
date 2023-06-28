"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture
import common
def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component
    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture
    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, _ = X.shape
    K, _ = mixture.mu.shape
    log_post = np.zeros((n, K))
    ll = 0
    for i in range(n):
        mask = X[i]!=0
        for j in range(K):
            log_likelihood = log_gaussian(X[i,mask] , mixture.mu[j,mask] , mixture.var[j])
            log_post[i,j] = np.log(mixture.p[j] + 1e-16) + log_likelihood
        total = logsumexp(log_post[i, :])
        log_post[i,:] -= total
        ll += total
    return np.exp(log_post) , ll
        


def log_gaussian(X:np.ndarray , mu:np.ndarray , var:float):
    d = len(X)
    log_prob = -d / 2.0 * np.log(2 * np.pi * var)
    log_prob -= 0.5 * ((X - mu)**2).sum() / var
    return log_prob  

def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    # raise NotImplementedError
    n, d = X.shape
    _, K = post.shape
    n_hat = post.sum(axis=0)
    p = n_hat / n
    mu = mixture.mu.copy()
    var = np.zeros(K)
    for j in range(K):
        sse, weight = 0, 0
        for l in range(d):
            mask = (X[:, l] != 0)
            n_sum = post[mask, j].sum()
            if (n_sum >= 1):
                # Updating mean
                mu[j, l] = (X[mask, l] @ post[mask, j]) / n_sum
            # Computing variance
            sse += ((mu[j, l] - X[mask, l])**2) @ post[mask, j]
            weight += n_sum
        var[j] = sse / weight
        if var[j] < min_variance:
            var[j] = min_variance
    return GaussianMixture(mu, var, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    # raise NotImplementedError
    old_log_likelihood = None
    new_log_likelihood = None
    while old_log_likelihood is None or new_log_likelihood - old_log_likelihood > 1e-6*np.abs(new_log_likelihood):
        old_log_likelihood = new_log_likelihood
        post , new_log_likelihood = estep(X,mixture)
        mixture = mstep(X,post,mixture,min_variance=0.25)
    return mixture , post , new_log_likelihood


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
        """Fills an incomplete matrix according to a mixture model
         Args:
             X: (n, d) array of incomplete data (incomplete entries =0)
             mixture: a mixture of gaussians
         Returns
             np.ndarray: a (n, d) array with completed data
        """
        n, d = X.shape
        X_pred = X.copy()
        K, _ = mixture.mu.shape
        for i in range(n):
            mask = X[i, :] != 0
            mask0 = X[i, :] == 0
            post = np.zeros(K)
            for j in range(K):
                log_likelihood = log_gaussian(X[i, mask], mixture.mu[j, mask],
                                            mixture.var[j])
                post[j] = np.log(mixture.p[j]) + log_likelihood
            post = np.exp(post - logsumexp(post))
            X_pred[i, mask0] = np.dot(post, mixture.mu[:, mask0]) # X_cu(i) = sum_j(p(j|i) * mu(j))
        return X_pred

