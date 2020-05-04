import numpy as np

### Implementations on http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html

def precision(true_arr, rec_arr, k = None, disp = True):
    '''
    input:
        true_arr(shape: num_users * num_movies): movies truely watched for all users in a 2D numpy array
        rec_arr(shape: num_users * num_movies): movies recommended for all users in a 2D numpy array
        k: the cutoff (positive integer)
        disp: print the result
    output:
        precision at cutoff = k 
    Note: By design, true_arr and rec_arr should have the same shape. However, this func can handle different num_movies.
        The values in 2 input arrays can be numeric and string
    '''
    if k:
        assert k <= rec_arr.shape[1]
        assert k > 0
        rec_arr = rec_arr[:,:k]
    else:
        if k == 0: raise Exception()
        else: k = rec_arr.shape[1]
        
    p = []
    for i in range(rec_arr.shape[0]):
        p.append(np.intersect1d(true_arr[i], rec_arr[i]).shape[0]/k)
    avg_p = np.mean(p)
    
    if disp:
        print('The average precision: %.3f' % avg_p)
    return avg_p


def MAP(true_arr, rec_arr, k = None, disp = True):
    '''
    input:
        true_arr(shape: num_users * num_movies): movies truely watched for all users in a 2D numpy array
        rec_arr(shape: num_users * num_movies): movies recommended for all users in a 2D numpy array
        k: the cutoff (positive integer)
        disp: print the result
    output:
        MAP at cutoff = k 
    Note: true_arr and rec_arr must have same shape for convinience.
        The values in these 2 arrays can be numeric and string
    '''
    if k:
        assert k <= rec_arr.shape[1]
        assert k > 0
        rec_arr = rec_arr[:,:k]
    else:
        if k == 0: raise Exception()
        else: k = rec_arr.shape[1]
    
    # create a user by k matrix for storing AP@k 
    # by finding the number of common elements in rec and real for each user i at k = j
    p = np.zeros((rec_arr.shape[0], k))
    for i in range(rec_arr.shape[0]):
        for j in range(k):
            p[i,j] = np.intersect1d(true_arr[i,:j+1], rec_arr[i,:j+1]).shape[0]/(j+1)
    
    mAP = np.mean(np.mean(p,1))
    
    if disp:
        print('The mean average prevision(MAP): %.3f' % mAP)
    
    return mAP

