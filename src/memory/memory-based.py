### memory-based model
''' Ruoyu Zhu '''

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances
#import matplotlib.pyplot as plt


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


### Cosine Similarity
def similarity(user_item, kind, epsilon=1e-9):
	#user_item = user or item ???
	# kind = 'user': user-based prediction
	# kind = 'item': item-based prediction
    # epsilon -> small number for handling dived-by-zero errors
    if kind == 'user':
        sim = user_item.dot(user_item.T) + epsilon
    elif kind == 'item':
        sim = user_item.T.dot(user_item) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)


user_sim = similarity(train_um, kind='user')
item_sim = similarity(train_um, kind='item')
#print(user_sim.shape) #（610，610）
#print(item_sim.shape) #（9724，9724）


user_sim = 1-pairwise_distances(train_um, metric='cosine')
item_sim = 1-pairwise_distances(train_um.T, metric='cosine')
#print(user_sim.shape) #（610，610）
#print(item_sim.shape) #（9724，9724）


def predict(user_item, similarity, kind):  
	# kind = 'user': user-based prediction
	# kind = 'item': item-based prediction
    if kind == 'user':
        return similarity.dot(user_item) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif kind == 'item':
        return user_item.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])

user_pred = predict(train_um, user_sim, kind='user')
item_pred = predict(train_um, item_sim, kind='item')


### Recommendations
def memory_based_recommender(prediction, userid, rating, n_recom):
    #rename: recommend_movies
    
    # Get and sort the user's predictions
    pred = pd.DataFrame(prediction, index=um.index)
    pred_sort = pred.loc[userid].sort_values(ascending=False)
    pred_s ort.rename('prediction', inplace=True)
    
    user_data = rating[rating.userId == userid]
    user_data = user_data.merge(movies, how = 'left', on = 'movieId').sort_values(['rating'], ascending=False)
    
    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    notsee = movies[~movies['movieId'].isin(user_data['movieId'])]
    notsee = notsee.merge(pd.DataFrame(pred_sort), how='left', left_on='movieId', right_index=True)
    notsee = notsee.sort_values('prediction', ascending=False)
    recommendation = notsee.iloc[:n_recom, :-1]
    
    return recommendation

user_sim = similarity(train_um, kind='user')
user_pred = predict(train_um, user_sim, kind='user')
memory_based_recommender(user_pred, 610, ratings, 5)

user_sim = similarity(um.values, kind='user')
user_pred = predict(um.values, user_sim, kind='user')
memory_based_recommender(user_pred, 610, ratings, 5)

memory_based_recommender(item_pred, 610, ratings, 5)


### Evaluation: RMSE
def get_rmse(pred, act):
    # Ignore nonzero terms.
    pred = pred[act.nonzero()].flatten()
    act = act[act.nonzero()].flatten()
    rmse = mean_squared_error(pred, act)**0.5
    return rmse

print('User-based RMSE: ', get_rmse(user_pred, test_um))
print('Item-based RMSE: ', get_rmse(item_pred, test_um))

