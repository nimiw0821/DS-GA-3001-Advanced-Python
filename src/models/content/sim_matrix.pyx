import numpy as np
cimport numpy as np

def create_sim_matrix_cython(tfidf_matrix):
    '''
    Create the pairwise similarity matrix for all the items from scratch
    using nested for loops (very slow).
    '''
    cdef np.ndarray X = tfidf_matrix.toarray()
    cdef np.ndarray Y = X.T
    cdef int X_h = len(X)
    cdef int Y_w = len(Y[0])
    cdef int Y_h = len(Y)
    cdef np.ndarray[double, ndim=2] sim_mat = np.zeros((X_h, X_h))
    cdef int i
    cdef int j
    cdef int k
    for i in range(X_h):
        for j in range(Y_w):
            for k in range(Y_h):
                sim_mat[i][j] += X[i][k] * Y[k][j]
    return sim_mat

    
