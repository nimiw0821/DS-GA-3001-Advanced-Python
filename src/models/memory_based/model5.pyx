import numpy as np
import pandas as pd



def memory_based_recommender_combined(prediction, int userid, rating,int n_recom, movies_df):
    """cython and reduce dataframe merge"""
    # Get and sort the user's predictions
    cdef list watched_id = []
    pred = pd.DataFrame(prediction, index=np.arange(1,prediction.shape[0] + 1))  # eliminate pass-in variable
    pred_sort = pred.loc[userid].sort_values(ascending=False)
    pred_sort.rename('prediction', inplace=True)
    
    watched_id = rating[rating.userId == userid].movieId.tolist()  #use list instead of pandas
    
    notsee = movies_df[~movies_df['movieId'].isin(watched_id)]
    notsee = notsee.merge(pd.DataFrame(pred_sort), how='left', left_on='movieId', right_index=True)
    notsee = notsee.sort_values('prediction', ascending=False)
    recommendation = notsee.iloc[:n_recom, :-1]
    
    return recommendation
