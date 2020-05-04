# run the functions and test the time takes
import time
import pandas as pd
import numpy as np
import sys
import model3
sys.path.append("/Users/ruoyuzhu/Documents/3001-AdvancedPython/Final_Project/DS-GA-3001-Advanced-Python/src/")
from common.preprocess.data_process import generate_userByItem, split, load_movies

sys.path.append("/Users/ruoyuzhu/Documents/3001-AdvancedPython/Final_Project/DS-GA-3001-Advanced-Python/src/")
from common.preprocess.data_process import load_file, generate_userByItem, split, load_movies
from common.cosine_similarity import *
from evaluation.evaluate import precision, MAP
from models.memory_based.predict_memory_recommender import *



# load data from ratings.csv
path_rating = f"/Users/ruoyuzhu/Documents/3001-AdvancedPython/Final_Project/DS-GA-3001-Advanced-Python/data/ratings.csv"
path_movies = f"/Users/ruoyuzhu/Documents/3001-AdvancedPython/Final_Project/DS-GA-3001-Advanced-Python/data/movies.csv"

um = generate_userByItem(path_rating , p = True)
ratings = load_file(path_rating , col_select_list = ['userId', 'movieId', 'rating'])
movies = load_file(path_movies)



# train test split
train_um, test_um = split(um)

# test cosine similarity run time 
print('___________testing similarity runtime_____________')
ts1 = time.time()
user_sim_1 = cos_similarity_1(train_um, kind='user')
item_sim_1 = cos_similarity_1(train_um, kind='item')
ts2 = time.time()
print('cosine simiarity 1 takes', ts2-ts1, 'secs')

ts3 = time.time()
user_sim_2 = create_sim_matrix_distance(train_um)
item_sim_2 = create_sim_matrix_distance(train_um.T) 
ts4 = time.time()
print('cosine simiarity 2 takes', ts4-ts3, 'secs')




print('________end of testing similarity runtime___________')


#Evaluation



user_sim = user_sim_1
item_sim = item_sim_1

user_pred = predict(train_um, user_sim, kind='user')
item_pred = predict(train_um, item_sim, kind='item')




print('___________memory-based model runtime_____________')

ts_model_1 = time.time()
print(memory_based_recommender(user_pred, 10, ratings, 5, um, movies))
print(memory_based_recommender(item_pred, 10, ratings, 5, um, movies))
ts_model_2 = time.time()
print('model 1 takes', ts_model_2-ts_model_1, 'secs')


ts_model_3 = time.time()
print(memory_based_recommender_2(user_pred, 10, ratings, 5, um, movies))
print(memory_based_recommender_2(item_pred, 10, ratings, 5, um, movies))
ts_model_4 = time.time()
print('model 2 takes', ts_model_4-ts_model_3, 'secs')

ts_model_5 = time.time()
print(model3.memory_based_recommender_cython(user_pred, 10, ratings, 5, um, movies))
print(model3.memory_based_recommender_cython(item_pred, 10, ratings, 5, um, movies))
ts_model_6 = time.time()
print('model 3 cython takes', ts_model_6-ts_model_5, 'secs')



# user_sim = similarity(um.values, kind='user')
# user_pred = predict(um.values, user_sim, kind='user')
# memory_based_recommender(user_pred, 610, ratings, 5)

print('___________memory-based model runtime end_____________')

#print('User-based RMSE: ', get_rmse(user_pred, test_um, k = 5))
#print('Item-based RMSE: ', get_rmse(item_pred, test_um, k = 5))

