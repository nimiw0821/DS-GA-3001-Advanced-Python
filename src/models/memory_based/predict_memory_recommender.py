### memory-based model
''' Ruoyu Zhu '''

import pandas as pd
import numpy as np

def predict(user_item, similarity, kind):  
	# kind = 'user': user-based prediction
	# kind = 'item': item-based prediction
    if kind == 'user':
        return similarity.dot(user_item) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif kind == 'item':
        return user_item.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])

#user_pred = predict(train_um, user_sim, kind='user')
#item_pred = predict(train_um, item_sim, kind='item')


# Recommender function
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


from numba import jit
@jit
def memory_based_recommender_2(prediction, userid, rating, n_recom, user_matrix, movies_df):
    """numba jit"""
    
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


# def memory_based_recommender_3():
# """add ctype"""
#     return 0


