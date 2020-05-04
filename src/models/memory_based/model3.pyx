import pandas as pd
import numpy as np

def memory_based_recommender_cython(prediction,int userid, rating,int n_recom, user_matrix, movies_df):
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
