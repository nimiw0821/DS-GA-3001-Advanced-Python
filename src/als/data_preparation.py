import numpy as np
import pandas as pd
import scipy.sparse as sparse

def load_file(file_path, col_select_list = None):
    '''
    load a file from directory
    '''
    file = pd.read_csv(file_path, header=0, usecols = col_select_list)
    return file

def data_preprocess(file_path_ratings, file_path_movies, col_select_list = None):
    '''
    return full data, train data, test data in sparse format, and a lookup table for movieId in dictionary
    '''
    # load in data
    print('Loading and preprocessing ratings data...')
    df = pd.read_csv(file_path_ratings, header=0, usecols = col_select_list)

    # Create user by item matrix
    data = pd.pivot_table(df, values='rating', index='userId', columns='movieId').fillna(0)

    # maximum and minimum number of movies watched by a user
    l =[]
    for i in range(len(data)):
        l.append(data.values[i, :].nonzero()[0].shape[0])
    print('Maximum number of movies watched by a user: {}\nMinimum number of movies watched by a user: {}'.format(max(l), min(l)))
    
    
    # train test split based on minimum #movies
    print('Select 10 movies for each user in test set')
    print('Splitting data into train and test...')
    train_data = data.values.copy()
    test_data = np.zeros(data.shape)

    for i in range(data.shape[0]):
        test_rating = np.random.choice(data.values[i, :].nonzero()[0], size=10, replace=False)
        train_data[i, test_rating] = 0
        test_data[i, test_rating] = data.values[i, test_rating]

    # Contruct a sparse matrix for train and test
    full_sparse = sparse.csr_matrix(data)
    train_sparse = sparse.csr_matrix(train_data)
    test_sparse = sparse.csr_matrix(test_data)
    
    print('Loading and preprocessing movies data...')
    movies = pd.read_csv(file_path_movies, header=0)
    
    # Since there are 9724 movies but the movieId ranges from 1 to 193609, we create a lookup table
    print('Generate lookup table for movie index...')
    k = np.unique(df.movieId, return_inverse=True)[1]
    movieId_lookup = dict(zip(k, df.movieId))
    
    print('Done!')
    
    return full_sparse, train_sparse, test_sparse, movies, movieId_lookup