from content_recommender import *
import sim_matrix
import time
import os

if __name__ == '__main__':
    file_path = os.path.abspath('ml-latest-small/movies.csv')
    df = preprocess(file_path, 'genres')
    tfidf_matrix = create_tfidf_matrix(df, 'genres')

    movie1 = 'Toy Story (1995)'
    movie2 = 'Fight Club (1999)'
    movie3 = 'Saving Private Ryan (1998)'
    n = 10
    sim_name = 'cython'

    t = time.time()
    sim_mat = sim_matrix.create_sim_matrix_cython(tfidf_matrix)
    rec1 = content_recommender(df, sim_mat, movie1, n)
    rec2 = content_recommender(df, sim_mat, movie2, n)
    rec3 = content_recommender(df, sim_mat, movie3, n)
    print('Top {} recommendations for {}:\n{}\n'.format(n, movie1, rec1))
    print('Top {} recommendations for {}:\n{}\n'.format(n, movie2, rec2))
    print('Top {} recommendations for {}:\n{}\n'.format(n, movie3, rec3))
    print('Time for {} is {:.6f}'.format(sim_name, time.time() - t))
    
