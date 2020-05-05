import sys
sys.path.append("/Users/zihaoguo/NYU/ADPY/DS-GA-3001-Advanced-Python/src/")

import numpy as np
import pandas as pd
import scipy.sparse as sparse
import time
from common.preprocess.data_process import generate_userByItem, split, load_movies
from models.als.model import implicit_als, implicit_als_cg
from models.als.model_jit import implicit_als_cg_jit
import os
import shutil
# in case there is no .so file
try:
    from models.als.model_cython import implicit_als_cg_cython
except:
    source_path = os.path.join(os.path.abspath(''), 'model_cython.cpython-37m-darwin.so')
    dest_path = os.path.join(os.path.abspath('../../..'), 'models','als', 'model_cython.cpython-37m-darwin.so')
    shutil.move(source_path, dest_path)
from models.als.model_cython import implicit_als_cg_cython
from common.serving.recommder_als import recommend, real_watched_movie
from evaluation.evaluate import MAP, precision


if __name__ == '__main__':
    data_path = os.path.dirname(os.path.abspath('../../..'))
    file_path = os.path.join(data_path, 'data', 'ratings.csv')
    
    full_sparse = generate_userByItem(file_path, ['userId', 'movieId', 'rating'], dense =True, info =True)
    train_sparse, test_sparse = split(full_sparse, dense = True)
    
    types = ['Original ALS training time', 'ALS training time with conjugate gradient', \
             'ALS training time using conjugate gradient with Numba JIT optimization', \
            'ALS training time using conjugate gradient with Cython optimization']

    for i, t in enumerate(types):  
        start = time.time()
        if i == 0:
            user_vecs, item_vecs = implicit_als(train_sparse, alpha_val=15, iterations=20, lambda_val=0.1, features=20)
        elif i == 1:
            user_vecs, item_vecs = implicit_als_cg(train_sparse, alpha_val=15, iterations=20, lambda_val=0.1, features=20)
        elif i == 2:
            user_vecs, item_vecs = implicit_als_cg_jit(train_sparse.toarray(), alpha_val=15, \
                                                       iterations=20, lambda_val=0.1, features=20)
        else:
            user_vecs, item_vecs = implicit_als_cg_cython(train_sparse.toarray(), alpha_val=15, \
                                                       iterations=20, lambda_val=0.1, features=20)
        end = time.time()
        print('{}: {}s'.format(t, end-start))
        