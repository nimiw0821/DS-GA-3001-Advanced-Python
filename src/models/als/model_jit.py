### ALS with optimization using conjugate gradient
import numpy as np
from numba import jit

@jit(nopython=True)
def implicit_als_cg_jit(data, alpha_val = 15, iterations=20, lambda_val=0.1, features=20):
    '''
    data is numpy array
    return recomendation vector
    '''
    Cui = data * alpha_val
    user_size, item_size = Cui.shape
    
    np.random.seed(123)
    X = np.random.rand(user_size, features) * 0.01
    Y = np.random.rand(item_size, features) * 0.01

    Cui, Ciu = Cui, Cui.T

    for iteration in range(iterations):
        ### print not working in numba compilier
        # print('iteration {} of {}'.format(iteration+1, iterations))
        least_squares_cg(Cui, X, Y, lambda_val)
        least_squares_cg(Ciu, Y, X, lambda_val)
    
    return X, Y

##### helper functiions for implicit_als_cg
@jit(nopython=True)
def nonzeros(m, row):
    items_user = m[row]
    idx = items_user.nonzero()[0]
    return np.stack((idx, items_user[idx]), 1)
        
@jit(nopython=True)
def least_squares_cg(Cui, X, Y, lambda_val, cg_steps=3):
    users, features = X.shape

    YtY = Y.T.dot(Y) + lambda_val * np.eye(features)

    for u in range(users):

        x = X[u]
        r = -YtY.dot(x)
        
        pair = nonzeros(Cui, u)
        for i in range(pair.shape[0]):
            r += (pair[i,1] - (pair[i,1] - 1) * Y[int(pair[i,0])].dot(x)) * Y[int(pair[i,0])]

        p = r.copy()
        rsold = r.dot(r)

        for it in range(cg_steps):
            Ap = YtY.dot(p)
            pair = nonzeros(Cui, u)
            for i in range(pair.shape[0]):
                Ap += (pair[i,1] - 1) * Y[int(pair[i,0])].dot(p) * Y[int(pair[i,0])]

            alpha = rsold / p.dot(Ap)
            x += alpha * p
            r -= alpha * Ap

            rsnew = r.dot(r)
            p = r + (rsnew / rsold) * p
            rsold = rsnew

        X[u] = x