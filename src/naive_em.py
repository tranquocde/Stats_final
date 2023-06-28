"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from src.common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    # raise NotImplementedError
    n,_ = X.shape
    K,_ = mixture.mu.shape
    post = np.zeros((n,K))
    log_likelihood = 0
    for i in range(n):
        for j in range(K):
            prob = gaussian(X[i] , mixture.mu[j] , mixture.var[j])
            post[i,j] = mixture.p[j] * prob
        sum = post[i,:].sum()
        post[i,:] /= sum
        log_likelihood += np.log(sum)
    return post , log_likelihood



def gaussian(x: np.ndarray, mean: np.ndarray, var: float) -> float:
    """Computes the probablity of vector x under a normal distribution
    Args:
        x: (d, ) array holding the vector's coordinates
        mean: (d, ) mean of the gaussian
        var: variance of the gaussian
    Returns:
        float: the probability
    """
    d = len(x)
    log_prob = -d / 2.0 * np.log(2 * np.pi * var)
    log_prob -= 0.5 * ((x - mean)**2).sum() / var
    return np.exp(log_prob)

def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    # raise NotImplementedError
    n,d = X.shape
    _,K = post.shape
    n_hat = post.sum(axis=0)
    p = n_hat / n        #(K,)
    var = np.zeros(K)     #(K,)
    mu = np.zeros((K,d))       #(K,d)
    for j in range(K):
         # Computing mean
         mu[j, :] = (X * post[:, j,None]).sum(axis=0) / n_hat[j]
         # Computing variance
         sse = ((mu[j] - X)**2).sum(axis=1) @ post[:, j]
         var[j] = sse / (d * n_hat[j])
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
        mixture = mstep(X,post)
    return mixture , post , new_log_likelihood

