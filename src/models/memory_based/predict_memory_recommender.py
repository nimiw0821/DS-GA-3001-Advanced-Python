### memory-based model
''' Ruoyu Zhu '''

import pandas as pd
import numpy as np
#from sklearn.metrics import mean_squared_error
#from sklearn.metrics.pairwise import pairwise_distances

# ### Cosine Similarity
# def similarity(user_item, kind, epsilon=1e-9):
# 	#user_item = user or item ???
# 	# kind = 'user': user-based prediction
# 	# kind = 'item': item-based prediction
#     # epsilon -> small number for handling dived-by-zero errors
#     if kind == 'user':
#         sim = user_item.dot(user_item.T) + epsilon
#     elif kind == 'item':
#         sim = user_item.T.dot(user_item) + epsilon
#     norms = np.array([np.sqrt(np.diagonal(sim))])
#     return (sim / norms / norms.T)


# user_sim = similarity(train_um, kind='user')
# item_sim = similarity(train_um, kind='item')
# #print(user_sim.shape) #（610，610）
# #print(item_sim.shape) #（9724，9724）


# user_sim = 1-pairwise_distances(train_um, metric='cosine')
# item_sim = 1-pairwise_distances(train_um.T, metric='cosine')
# #print(user_sim.shape) #（610，610）
# #print(item_sim.shape) #（9724，9724）


def predict(user_item, similarity, kind):  
	# kind = 'user': user-based prediction
	# kind = 'item': item-based prediction
    if kind == 'user':
        return similarity.dot(user_item) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif kind == 'item':
        return user_item.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])

#user_pred = predict(train_um, user_sim, kind='user')
#item_pred = predict(train_um, item_sim, kind='item')


### Recommendations
def memory_based_recommender(prediction, userid, rating, n_recom, user_matrix, movies_df):
"""basic recommender function"""
    # Get and sort the user's predictions
    pred = pd.DataFrame(prediction, index=user_matrix.index)  # user_matrix = um
    pred_sort = pred.loc[userid].sort_values(ascending=False)
    pred_sort.rename('prediction', inplace=True)
    
    user_data = rating[rating.userId == userid]
    user_data = user_data.merge(movies_df, how = 'left', on = 'movieId').sort_values(['rating'], ascending=False)
    
    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    notsee = movies_df[~movies_df['movieId'].isin(user_data['movieId'])]
    notsee = notsee.merge(pd.DataFrame(pred_sort), how='left', left_on='movieId', right_index=True)
    notsee = notsee.sort_values('prediction', ascending=False)
    recommendation = notsee.iloc[:n_recom, :-1]
    
    return recommendation


def memory_based_recommender_2():
"""add ctype"""
    return 0




