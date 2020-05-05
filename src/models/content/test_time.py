from content_recommender import *
from cosine_similarity import *
import time
import os

if __name__ == '__main__':
    data_path = os.path.dirname(os.path.abspath('../..'))
    file_path = os.path.join(data_path, 'data', 'movies.csv')

    t0 = time.time()
    df = preprocess(file_path, 'genres')
    tfidf_matrix = create_tfidf_matrix(df, 'genres')
    print('Time for Tfidf matrix preparation: {:.6f}'.format(time.time() - t0))
    print('-' * 20)
    
    sim_mat_fns = [create_sim_matrix_kernel, create_sim_matrix_cosine,
                   create_sim_matrix_distance, create_sim_matrix_dot,
                   create_sim_matrix_product, create_sim_matrix_operator,
                   create_sim_matrix_scratch]

    sim_names = ['linear_kernal', 'cosine_similarity', 'pairwise_distance', 'numpy.dot',
                 'product', 'operator', 'nested for loops']

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
        # print('Top {} recommendations for {}:\n{}\n'.format(n, movie1, rec1))
        # print('Top {} recommendations for {}:\n{}\n'.format(n, movie2, rec2))
        # print('Top {} recommendations for {}:\n{}\n'.format(n, movie3, rec3))
        print('Time for {} is {:.6f}'.format(sim_name, time.time() - t))
        print('-' * 20)
