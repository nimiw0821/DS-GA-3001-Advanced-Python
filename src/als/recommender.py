import numpy as np
from sklearn.preprocessing import MinMaxScaler
#------------------------------
# CREATE USER RECOMMENDATIONS
#------------------------------

def recommend(user_id, data_sparse, user_vecs, item_vecs, num_items=10):
    """Recommend items for a given user given a trained model
    
    Args:
        user_id (int): The id of the user we want to create recommendations for.
        data_sparse (csr_matrix): Our train or test sparse data
        user_vecs (csr_matrix): The trained user x features vectors
        item_vecs (csr_matrix): The trained item x features vectors
        num_items (int): How many recommendations we want to return
    Returns:
        recommendations: a list of movie index
    
    """
  
    # Get all interactions by the user
    user_interactions = data_sparse[user_id,:].toarray()
    
    # We don't want to recommend items the user has consumed. So let's
    # set them all to 0 and the unknowns to 1.
    user_interactions = user_interactions.reshape(-1) + 1 #Reshape to turn into 1D array
    user_interactions[user_interactions > 1] = 0
        
    # This is where we calculate the recommendation by taking the 
    # dot-product of the user vectors with the item vectors.
    rec_vector = user_vecs[user_id,:].dot(item_vecs.T).toarray()

    # Let's scale our scores between 0 and 1 to make it all easier to interpret.
    min_max = MinMaxScaler()
    rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1,1))[:,0]
    recommend_vector = user_interactions*rec_vector_scaled
   
    # Get all the artist indices in order of recommendations (descending) and
    # select only the top "num_items" items. 
    item_idx = np.argsort(recommend_vector)[::-1][:num_items]
    return item_idx

def avg_precision(train_sparse, test_sparse, user_vecs, item_vecs):
    p = []
    t = test_sparse.toarray()
    for i in range(test_sparse.shape[0]):
        rec_list = recommend(i, train_sparse, user_vecs, item_vecs)
        true_list = t[i].nonzero()[0]
        p.append(np.intersect1d(true_list, rec_list).shape[0]/10)
    avg_p = np.mean(p)
    print('The average prevision: %.3f' % avg_p)
    return avg_p