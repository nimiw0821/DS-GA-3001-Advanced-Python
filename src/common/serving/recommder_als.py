import numpy as np

#------------------------------
# CREATE USER RECOMMENDATIONS
#------------------------------

def recommend(data_sparse, user_vecs, item_vecs, num_items=10):
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
    # Get all interactions by all users
    users_interactions = data_sparse.toarray()
    
    # We don't want to recommend items the user has consumed. So let's
    # set them all to 0 and the unknowns to 1.
    users_interactions = users_interactions + 1
    users_interactions[users_interactions > 1] = 0
        
    # This is where we calculate the recommendation by taking the 
    # dot-product of the users vectors with the items vectors.
    rec_vector = user_vecs.dot(item_vecs.T).toarray()
    
    # get the final recommend vector with masked interations
    recommend_vector = users_interactions*rec_vector
   
    # Get all the artist indices in order of recommendations (descending) and
    # select only the top "num_items" items. 
    item_idx = np.flip(np.argsort(recommend_vector), 1)[:,:num_items]
    return item_idx

def real_watched_movie(data_sparse):
    '''
    get the movies that users really watch in test set
    '''
    
    data_sparse = data_sparse.toarray()
    item_idx = data_sparse.nonzero()[1].reshape(data_sparse.shape[0],-1)
    return item_idx