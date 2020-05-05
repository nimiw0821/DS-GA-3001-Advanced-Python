### best memory-based model
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


def memory_based_recommender_best(prediction, userid, rating, n_recom, movies_df):
    """best recommender function - with lowest runtime (model4)"""
  
    pred = pd.DataFrame(prediction, index=np.arange(1,prediction.shape[0] + 1))  # eliminate pass-in variable
    pred_sort = pred.loc[userid].sort_values(ascending=False)
    pred_sort.rename('prediction', inplace=True)
    
    watched_id = rating[rating.userId == userid].movieId #use list instead of pandas
 
    notsee = movies_df[~movies_df['movieId'].isin(watched_id)]
    notsee = notsee.merge(pd.DataFrame(pred_sort), how='left', left_on='movieId', right_index=True)
    notsee = notsee.sort_values('prediction', ascending=False)
    recommendation = notsee.iloc[:n_recom, :-1]
    
    return recommendation