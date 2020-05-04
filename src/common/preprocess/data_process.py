import numpy as np
import pandas as pd
import scipy.sparse as sparse

def load_file(file_path, col_select_list = None):
    '''
    load a file from directory
    '''
    file = pd.read_csv(file_path, header=0, usecols = col_select_list)
    return file

def generate_userByItem(file_path_ratings, col_select_list = None, p = False, dense = False, info = False):
    '''
    input:
        file_path_ratings: The file path to ratings.csv
        col_select_list: a list of columns to select
        p: return pandas dataframe
        dense: return sparse matrix
        info: max and min number of movies watched by some user in the data
    output:
        numpy array of userByItem table by default
    '''
    # load in data
    print('Loading and pivoting ratings data...')
    df = pd.read_csv(file_path_ratings, header=0, usecols = col_select_list)

    # Create user by item matrix
    data = pd.pivot_table(df, values='rating', index='userId', columns='movieId').fillna(0)
    
    # maximum and minimum number of movies watched by a user
    if info:
        l =[]
        for i in range(len(data)):
            l.append(data.values[i, :].nonzero()[0].shape[0])
        print('Maximum number of movies watched by a user: {}\nMinimum number of movies watched by a user: {}'.format(max(l), min(l)))
        
    print('Done')
    if p:
        return data
    elif dense:
        return sparse.csr_matrix(data)
    else:
        return data.values
    
    
def split(data, p = False, dense = False):
    '''
    split data into train test.
    Support np.array, pd.DataFrame, and sparse.csr_matrix for both input and output
    '''
    # check input type and transform it to np.array for splitting later
    if isinstance(data, pd.DataFrame):
        data = data.values.copy()
    elif isinstance(data, sparse.csr_matrix):
        data = data.toarray().copy()
    
    # train test split based on minimum #movies
    print('Assign 10 movies for each user to test and the rest to train')
    print('Splitting data into train and test...')
    train_data = data.copy()
    test_data = np.zeros(data.shape)

    for i in range(data.shape[0]):
        test_rating = np.random.choice(data[i, :].nonzero()[0], size=10, replace=False)
        train_data[i, test_rating] = 0
        test_data[i, test_rating] = data[i, test_rating]
        
    print('Done')
    if p:
        return pd.DataFrame(train_data), pd.DataFrame(test_data)
    elif dense:
        return sparse.csr_matrix(train_data), sparse.csr_matrix(test_data)
    else:
        return train_data, test_data
        
    
def load_movies(file_path_movies):
    '''
    Load the movies.csv and create a lookup table for movieId in dictionary
    '''
    print('Loading and preprocessing movies data...')
    movies = pd.read_csv(file_path_movies, header=0)
    
    # Since there are 9724 movies but the movieId ranges from 1 to 193609, we create a lookup table
    print('Generating lookup table for movie index...')
    k = np.unique(movies.movieId, return_inverse=True)[1]
    movieId_lookup = dict(zip(k, movies.movieId))
    
    print('Done')
    return movies, movieId_lookup
