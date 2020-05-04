from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

def preprocess(file_path, feature):
    '''
    Load the data file and preprocess the pipe-separated feature.
    '''
    df = pd.read_csv(file_path, header = 0)
    df[feature] = df[feature].str.replace('|', ' ')
    return df


def create_tfidf_matrix(df, feature):
    '''
    Generate a sparse matrix of Tfidf based on the feature of df.
    '''
    tfidf = TfidfVectorizer(analyzer = 'word')
    tfidf_matrix = tfidf.fit_transform(df[feature])
    return tfidf_matrix  # sparse (9742, 24)


def get_target_indices(sim_matrix, idx, n):
    '''
    Utility function to find a list of indices of the items to recommend
    given the similarity matrix and the index of the given item.
    '''
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
