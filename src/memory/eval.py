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

