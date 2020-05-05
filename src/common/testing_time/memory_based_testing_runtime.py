# run the functions and test the time takes
import sys
sys.path.append("/Users/ruoyuzhu/Documents/3001-AdvancedPython/Final_Project/DS-GA-3001-Advanced-Python/src/")
from common.preprocess.data_process import generate_userByItem, split, load_movies
from common.preprocess.data_process import load_file, generate_userByItem, split, load_movies
from evaluation.evaluate import precision, MAP
from models.memory_based.predict_memory_recommender import *
from models.content.cosine_similarity import *
from models.memory_based.model3 import *
from models.memory_based.model_combined import *

import time
import pandas as pd
import numpy as np


# load data from ratings.csv
path_rating = f"/Users/ruoyuzhu/Documents/3001-AdvancedPython/Final_Project/DS-GA-3001-Advanced-Python/data/ratings.csv"
path_movies = f"/Users/ruoyuzhu/Documents/3001-AdvancedPython/Final_Project/DS-GA-3001-Advanced-Python/data/movies.csv"

um = generate_userByItem(path_rating , p = True)
ratings = load_file(path_rating , col_select_list = ['userId', 'movieId', 'rating'])
movies = load_file(path_movies)

# train test split
train_um, test_um = split(um)

user_sim = create_sim_matrix_cosine(train_um)
item_sim= create_sim_matrix_cosine(train_um.T)

user_pred = predict(train_um, user_sim, kind='user')
item_pred = predict(train_um, item_sim, kind='item')

# test memory_based recommender run time 

print('___________memory-based model runtime_____________')

ts_model_1 = time.time()
print(memory_based_recommender(user_pred, 10, ratings, 5, um, movies))
print(memory_based_recommender(item_pred, 10, ratings, 5, um, movies))
ts_model_2 = time.time()



ts_model_3 = time.time()
memory_based_recommender_2(user_pred, 10, ratings, 5, um, movies)
memory_based_recommender_2(item_pred, 10, ratings, 5, um, movies)
ts_model_4 = time.time()


ts_model_5 = time.time()
model3.memory_based_recommender_cython(user_pred, 10, ratings, 5, um, movies)
model3.memory_based_recommender_cython(item_pred, 10, ratings, 5, um, movies)
ts_model_6 = time.time()


ts_model_7 = time.time()
print(memory_based_recommender_4(user_pred, 10, ratings, 5, movies))
print(memory_based_recommender_4(item_pred, 10, ratings, 5, movies))
ts_model_8 = time.time()

ts_model_9 = time.time()
print(memory_based_recommender_combined(user_pred, 10, ratings, 5, movies))
print(memory_based_recommender_combined(item_pred, 10, ratings, 5, movies))
ts_model_10 = time.time()


print('model 1 takes', ts_model_2-ts_model_1, 'secs')
print('model 2 takes', ts_model_4-ts_model_3, 'secs')
print('model 3 cython takes', ts_model_6-ts_model_5, 'secs')
print('model 4 - reduce variables takes', ts_model_8-ts_model_7, 'secs')
print('model 5 - combined function takes', ts_model_10-ts_model_9, 'secs')


print('___________memory-based model runtime end_____________')


# user_sim = similarity(um.values, kind='user')
# user_pred = predict(um.values, user_sim, kind='user')
# memory_based_recommender(user_pred, 610, ratings, 5)


#print('User-based RMSE: ', get_rmse(user_pred, test_um, k = 5))
#print('Item-based RMSE: ', get_rmse(item_pred, test_um, k = 5))

