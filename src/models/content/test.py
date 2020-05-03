from content_based_recommender import preprocess, create_sim_matrix, content_recommender
import time

if __name__ == '__main__':
    ts1 = time.time()
    df = preprocess('ml-latest-small/movies.csv', 'genres')
    print('Preprocessing takes {:.4f} seconds.'.format(time.time() - ts1))

    ts2 = time.time()
    sim_matrix = create_sim_matrix(df, 'genres')
    print('Creating similarity matrix takes {:.4f} seconds.'.format(time.time() - ts2))

    ts3 = time.time()
    rec = content_recommender(df, sim_matrix, 'Toy Story (1995)', 10)
    print('Recommending 10 movies takes {:.4f} seconds.'.format(time.time() - ts3))
    print('Recommendations:\n', rec)
