### copy from Shirley's content_based_recommender.py file
### compare different cosine similarity method
### can be used across the models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np


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


# def create_sim_matrix_product(tfidf_matrix):
#     '''
#     Create the pairwise similarity matrix for all the items using numpy.
#     '''
#     # return the dense ndarray representation of the sparse matrix
#     tfidf_matrix = tfidf_matrix.toarray()
#     return np.dot(tfidf_matrix, tfidf_matrix.T)


def create_sim_matrix_numpy(tfidf_matrix):
    '''
    Create the pairwise similarity matrix for all the items using numpy.
    '''
    # return the dense ndarray representation of the sparse matrix
    tfidf_matrix = tfidf_matrix.toarray()
    return np.inner(tfidf_matrix, tfidf_matrix)


def create_sim_matrix_scratch_np(tfidf_matrix):
    '''
    Create the pairwise similarity matrix for all the items from scratch
    using nested for loops.
    Note: very, very slow.
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


# for memory based part
def cos_similarity_1(user_item, kind, epsilon=1e-9): 
    # epsilon -> small number for handling dived-by-zero errors
    if kind == 'user':
        sim = user_item.dot(user_item.T) + epsilon  #matrix X matrix itself.T
    elif kind == 'item':
        sim = user_item.T.dot(user_item) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

#MPI
# def create_sim_matrix_MPI():
