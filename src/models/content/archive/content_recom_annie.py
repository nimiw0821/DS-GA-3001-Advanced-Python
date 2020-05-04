from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.metrics import pairwise_distances
import operator
import pandas as pd
import numpy as np
import time
import os


def preprocess(file_path, feature):
    '''
    Load the data file and create tfidf matrix
    '''
    df = pd.read_csv(file_path, header = 0)
    df[feature] = df[feature].str.replace('|', ' ')
    tfidf = TfidfVectorizer(analyzer = 'word')
    tfidf_matrix = tfidf.fit_transform(df[feature])
    return df, tfidf_matrix



def sim_matrix_kernel(tfidf_matrix):
    '''
    create similarity matrix using liner_kernel
    '''
    sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
    return sim_matrix



def sim_matrix_cosine(tfidf_matrix):
    '''
    create similarity matrix using cosine_similarity
    '''
    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return sim_matrix



def sim_matrix_pairwise_dist(tfidf_matrix):
    '''
    create similarity matrix using pairwise_distances
    '''
    sim_matrix = 1 - pairwise_distances(tfidf_matrix, metric = 'cosine')
    return sim_matrix



def sim_matrix_dot(tfidf_matrix):
    '''
    create similarity matrix using numpy
    '''
    sim_matrix = np.dot(tfidf_matrix, tfidf_matrix.T)
    return sim_matrix



def sim_matrix_product(tfidf_matrix):
    '''
    create the similarity matrix using *
    '''
    sim_matrix = tfidf_matrix * tfidf_matrix.T
    return sim_matrix



def sim_matrix_operator(tfidf_matrix):
    '''
    reate the similarity matrix using operator
    '''
    sim_matrix = operator.mul(tfidf_matrix, tfidf_matrix.T)
    return sim_matrix



def get_target_indices(sim_matrix, idx, n):
    if isinstance(sim_matrix[idx], np.ndarray):
        sim_scores = np.argsort(sim_matrix[idx])
    else:
        sim_scores = np.argsort(sim_matrix[idx].toarray()[0])
    target_indices = sim_scores[-(n+1)::][::-1].tolist()
    if idx in target_indices: target_indices.remove(idx)
    return target_indices



def content_recommender(df, sim_matrix, title, n):
    '''
    Recommends a list of similar items for the user given a specific item
    and the number of recommendations.
    '''
    idx_title = df['title'] # pd Series
    # find the index of the given item
    idx = idx_title[idx_title == title].index[0]
    target_indices = get_target_indices(sim_matrix, idx, n)
    return list(idx_title.iloc[target_indices].values)


if __name__ == '__main__':
    data_path = os.path.dirname(os.path.abspath('../..'))
    file_path = os.path.join(data_path, 'data', 'movies.csv')
    df, tfidf_matrix = preprocess(file_path, 'genres')
    
    sim_mat_fns = [sim_matrix_kernel, 
                sim_matrix_cosine, 
                sim_matrix_dot, 
                sim_matrix_pairwise_dist, 
                sim_matrix_product, 
                sim_matrix_operator]

    sim_names = ['linear_kernal', 'cosine_similarity', 'numpy.dot',
                'pairwise_distance', 'product', 'operator']

    movie1 = 'Toy Story (1995)'
    movie2 = 'Fight Club (1999)'
    movie3 = 'Saving Private Ryan (1998)'
    
    n = 10

    for sim_name, sim_mat_fn in zip(sim_names, sim_mat_fns):
        t = time.time()
        sim_mat = sim_mat_fn(tfidf_matrix)
        rec1 = content_recommender(df, sim_mat, movie1, n)
        rec2 = content_recommender(df, sim_mat, movie2, n)
        rec3 = content_recommender(df, sim_mat, movie3, n)
        print('Time for {} is {:.6f}'.format(sim_name, time.time() - t))

