import os
#retval = os.getcwd()
#print("Current working directory %s" % retval)

import pandas as pd
import numpy as np

# #** Put here the directory where you have the file with your function**
# os.chdir("/Users/ruoyuzhu/Documents/3001-AdvancedPython/Final_Project/DS-GA-3001-Advanced-Python/src/common/preprocess")
# retval = os.getcwd()
# print("Current working directory %s" % retval)
import sys
sys.path.append("/Users/ruoyuzhu/Documents/3001-AdvancedPython/Final_Project/DS-GA-3001-Advanced-Python/src/")
from common.preprocess.data_process import generate_userByItem, split, load_movies



path_movie = f"/Users/ruoyuzhu/Documents/3001-AdvancedPython/Final_Project/DS-GA-3001-Advanced-Python/data/ratings.csv"
#os.chdir('/data/movies.csv'
um = generate_userByItem(path_movie, p = True)

print(um.values)