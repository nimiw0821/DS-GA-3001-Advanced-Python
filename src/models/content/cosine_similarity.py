from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.metrics import pairwise_distances
import numpy as np
import operator


def create_sim_matrix_kernel(tfidf_matrix):
    '''
    Create the pairwise similarity matrix for all the items
    using linear_kernel from sklearn since the input matrix is normalized.
    Ref: https://scikit-learn.org/stable/modules/metrics.html
    '''
    sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
    return sim_matrix


def create_sim_matrix_cosine(tfidf_matrix):
    '''
    Create the pairwise similarity matrix for all the items
    using cosine_similarity from sklearn.
    '''
    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return sim_matrix


def create_sim_matrix_distance(tfidf_matrix):
    '''
    Create the pairwise similarity matrix for all the items
    using pairwise_distances from sklearn and scipy.
    '''
    sim_matrix = 1 - pairwise_distances(tfidf_matrix, metric = 'cosine')
    return sim_matrix


def create_sim_matrix_dot(tfidf_matrix):
    '''
    Create the pairwise similarity matrix for all the items using numpy.
    '''
    sim_matrix = np.dot(tfidf_matrix, tfidf_matrix.T)
    return sim_matrix


def create_sim_matrix_product(tfidf_matrix):
    '''
    Create the pairwise similarity matrix for all the items using *.
    '''
    sim_matrix = tfidf_matrix * tfidf_matrix.T
    return sim_matrix


def create_sim_matrix_operator(tfidf_matrix):
    '''
    Create the pairwise similarity matrix for all the items using operator.
    '''
    sim_matrix = operator.mul(tfidf_matrix, tfidf_matrix.T)
    return sim_matrix


def create_sim_matrix_scratch(tfidf_matrix):
    '''
    Create the pairwise similarity matrix for all the items from scratch
    using nested for loops (very slow).
    '''
    # return the dense ndarray representation of the sparse matrix
    tfidf_matrix = tfidf_matrix.toarray()
    X, Y = tfidf_matrix, tfidf_matrix.T
    sim_matrix = np.empty((len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(Y[0])):
            for k in range(len(Y)):
                sim_matrix[i][j] += X[i][k] * Y[k][j]
    return sim_matrix
