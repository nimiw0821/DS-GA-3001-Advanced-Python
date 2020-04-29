from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
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


def create_sim_matrix(df, feature):
    '''
    Create the pairwise similarity matrix for all the movies in the dataset.
    '''
    tfidf = TfidfVectorizer(analyzer = 'word')
    tfidf_matrix = tfidf.fit_transform(df[feature])
    sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
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
    sim_matrix = create_sim_matrix(df, 'genres')
    movie1 = 'Toy Story (1995)'
    movie2 = 'Fight Club (1999)'
    movie3 = 'Saving Private Ryan (1998)'
    n = 10
    rec1 = content_recommender(df, sim_matrix, movie1, n)
    rec2 = content_recommender(df, sim_matrix, movie2, n)
    rec3 = content_recommender(df, sim_matrix, movie3, n)
    print('Top {} recommendations for {}:\n{}\n'.format(n, movie1, rec1))
    print('Top {} recommendations for {}:\n{}\n'.format(n, movie2, rec2))
    print('Top {} recommendations for {}:\n{}\n'.format(n, movie3, rec3))
