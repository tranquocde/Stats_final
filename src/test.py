import numpy as np
import src.em as em
import src.common as common

X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

K = 12
n, d = X.shape
seed = 0

# TODO: Your code here
mixture, post = common.init(X, K,  seed)
mixture, post, ln_like = em.run(X, mixture, post)

