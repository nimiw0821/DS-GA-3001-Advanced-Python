import numpy as np
cimport numpy as np

## Cython implementation of ALS
def implicit_als_cg_cython(np.ndarray[double, ndim = 2] data, int alpha_val = 15, \
                           int iterations=20, double lambda_val=0.1, int features=20):
    '''
    data is numpy array
    return recomendation vector
    '''
    cdef np.ndarray[double, ndim = 2] Cui = data * alpha_val
    cdef int user_size = Cui.shape[0]
    cdef int item_size = Cui.shape[1]
    
    np.random.seed(123)
    cdef np.ndarray[double, ndim = 2] X = np.random.rand(user_size, features) * 0.01
    cdef np.ndarray[double, ndim = 2] Y = np.random.rand(item_size, features) * 0.01

    cdef np.ndarray[double, ndim = 2] Ciu = Cui.T

    for iteration in range(iterations):
        ### print not working in numba compilier
        print('iteration {} of {}'.format(iteration+1, iterations))
        least_squares_cg(Cui, X, Y, lambda_val)
        least_squares_cg(Ciu, Y, X, lambda_val)
    
    return X, Y

##### helper functiions for implicit_als_cg
def nonzeros(np.ndarray[double, ndim = 2] m, int row):
    cdef np.ndarray items_user = m[row]
    cdef np.ndarray idx = items_user.nonzero()[0]
    return np.stack((idx, items_user[idx]), 1)
        
def least_squares_cg(np.ndarray[double, ndim = 2] Cui, np.ndarray[double, ndim = 2] X, \
                     np.ndarray[double, ndim = 2] Y, double lambda_val, int cg_steps=3):
    cdef int users = X.shape[0]
    cdef int features = X.shape[1]

    cdef np.ndarray[double, ndim = 2] YtY = Y.T.dot(Y) + lambda_val * np.eye(features)
    
    # stating the type of the variables in for-loops allows 
    # for a more optimized conversion to a C loop
    cdef int u
    cdef int it
    
    ## type of varaibles in the loop
    cdef np.ndarray x
    cdef np.ndarray r
    cdef int i
    cdef double confidence
    cdef np.ndarray p
    cdef double rsold
    cdef np.ndarray Ap
    cdef double alpha
    cdef double rsnew
    
    for u in range(users):
        
        x = X[u]
        r = -YtY.dot(x)
        
        for i, confidence in nonzeros(Cui, u):
            r += (confidence - (confidence - 1) * Y[i].dot(x)) * Y[i]
        p = r.copy()
        rsold = r.dot(r)

        for it in range(cg_steps):
            Ap = YtY.dot(p)
            
            for i, confidence in nonzeros(Cui, u):
                Ap += (confidence - 1) * Y[i].dot(p) * Y[i]
            
            alpha = rsold / p.dot(Ap)
            x += alpha * p
            r -= alpha * Ap

            rsnew = r.dot(r)
            p = r + (rsnew / rsold) * p
            rsold = rsnew

        X[u] = x