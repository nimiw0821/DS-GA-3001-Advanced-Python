from content_recommender import *
from cosine_similarity import *
import time
import os

if __name__ == '__main__':
    data_path = os.path.dirname(os.path.abspath('../..'))
    file_path = os.path.join(data_path, 'data', 'movies.csv')

    t0 = time.time()
    df = pd.read_csv(file_path, header = 0)
    df['genres'] = df['genres'].str.replace('|', ' ')
    tfidf = TfidfVectorizer(analyzer = 'word')
    tfidf_matrix = tfidf.fit_transform(df['genres'])
    print('Time for Tfidf matrix preparation: {:.6f}'.format(time.time() - t0))
    print('-' * 20)
    
    sim_mat_fns = [create_sim_matrix_kernel, create_sim_matrix_cosine,
                   create_sim_matrix_distance, create_sim_matrix_dot,
                   create_sim_matrix_product, create_sim_matrix_operator,
                   create_sim_matrix_scratch]

    sim_names = ['linear_kernal', 'cosine_similarity', 'pairwise_distance', 'numpy.dot',
                 'product', 'operator', 'nested for loops']

    movies = ['Toy Story (1995)', 'Fight Club (1999)', 'Saving Private Ryan (1998)']

    n = 10
    idx_title = df['title']

    for sim_name, sim_mat_fn in zip(sim_names, sim_mat_fns):
        t = time.time()
        sim_mat = sim_mat_fn(tfidf_matrix)
        for movie in movies:
            idx = idx_title[idx_title == movie].index[0]
            target_indices = get_target_indices(sim_mat, idx, n)
            rec = list(idx_title.iloc[target_indices].values)
            # print('Top {} recommendations for {}:\n{}\n'.format(n, movie, rec))
        print('Time for {} is {:.6f}'.format(sim_name, time.time() - t))
        print('-' * 20)
