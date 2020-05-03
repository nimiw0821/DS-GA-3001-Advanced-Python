import pandas as pd

def load_file(file_path, col_select_list = None):
    file = pd.read_csv(file_path, header=0, usecols = col_select_list)
    return file


### data loading

#use util/load.py/ load_file(file_path)

movies = pd.read_csv('data/movies.csv', header = 0)
ratings = pd.read_csv('data/ratings.csv', header = 0, 
                      usecols = ['userId', 'movieId', 'rating'])

#user-movie matrix
um = pd.pivot_table(ratings, values='rating', index='userId', columns='movieId')
um.fillna(0, inplace=True)

sparsity = ratings.shape[0]/um.size
#print('Sparsity: {:.2f}%'.format(sparsity*100)) 
## 1.7% of the user-movie ratings have a value.

### Train-Test Split
train_um = um.values.copy()
test_um = np.zeros(um.shape)

for user in range(um.shape[0]):
    test_rating = np.random.choice(um.values[user, :].nonzero()[0], size=10, replace=False)
    train_um[user, test_rating] = 0
    test_um[user, test_rating] = um.values[user, test_rating]
    
print(test_um.shape)
print(train_um.shape)

assert(np.all((train_um * test_um) == 0)) ## test train and test are disjoint
