#test function on github

import pandas as pd
import numpy as np

from load_copy import load_file

rating = load_file(f'/data/ratings.csv', ['userId', 'movieId', 'rating'])
rating.shape