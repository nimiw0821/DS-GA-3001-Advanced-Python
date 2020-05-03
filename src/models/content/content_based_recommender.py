from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np


def preprocess(file_path, feature):
    '''
    Load the data file and preprocess the pipe-separated feature.
    '''
    df = pd.read_csv(file_path, header = 0)
    df[feature] = df[feature].str.split('|').values
    df[feature] = df[feature].apply(lambda val: ' '.join(val))
    return df


def create_tfidf_matrix(df, feature):
    '''
    Generate a sparse matrix of Tfidf based on the feature of df.
    '''
    tfidf = TfidfVectorizer(analyzer = 'word')
    tfidf_matrix = tfidf.fit_transform(df[feature])
    return tfidf_matrix  # sparse (9742, 24)


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


def content_recommender(df, sim_matrix, title, n):
    '''
    Recommends a list of similar items for the user given a specific item
    and the number of recommendations.
    '''
    idx_title = df['title'] # pd Series
    # find the index of the given item
    idx = idx_title[idx_title == title].index[0]
    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
    # top n+1 recommendations
    target_sim_scores = sim_scores[:n+1]
    # exclude the given item since it will always have a sim_score close to 1
    target_sim_scores = [score for score in target_sim_scores if score[0] != idx]
    target_indices = [score[0] for score in target_sim_scores]
    return list(idx_title.iloc[target_indices].values)


if __name__ == '__main__':
    df = preprocess('ml-latest-small/movies.csv', 'genres')
    tfidf_matrix = create_tfidf_matrix(df, 'genres')
    sim_matrix_kernel = create_sim_matrix_kernel(tfidf_matrix)
    #print(type(tfidf_matrix))
    #print(tfidf_matrix.shape)
    movie1 = 'Toy Story (1995)'
    movie2 = 'Fight Club (1999)'
    movie3 = 'Saving Private Ryan (1998)'
    n = 10
    rec1 = content_recommender(df, sim_matrix_kernel, movie1, n)
    rec2 = content_recommender(df, sim_matrix_kernel, movie2, n)
    rec3 = content_recommender(df, sim_matrix_kernel, movie3, n)
    print('Top {} recommendations for {}:\n{}\n'.format(n, movie1, rec1))
    print('Top {} recommendations for {}:\n{}\n'.format(n, movie2, rec2))
    print('Top {} recommendations for {}:\n{}\n'.format(n, movie3, rec3))

    print('-' * 10)
    sim_matrix_kernel = create_sim_matrix_kernel(tfidf_matrix)
    sim_matrix_cosine = create_sim_matrix_cosine(tfidf_matrix)
    sim_matrix_distance = create_sim_matrix_distance(tfidf_matrix)
    #sim_matrix_product = create_sim_matrix_product(tfidf_matrix)
    sim_matrix_numpy = create_sim_matrix_numpy(tfidf_matrix)
    print(sim_matrix_kernel[:5])
    print(sim_matrix_cosine[:5])
    print(sim_matrix_distance[:5])
    #print(sim_matrix_product[:5])
    print(sim_matrix_numpy[:5])
    print('Finished the previous parts.\nCreating similarity matrix from scratch takes some time.')
    sim_matrix_scratch_np = create_sim_matrix_scratch_np(tfidf_matrix)
    print(sim_matrix_scratch_np[:5])
