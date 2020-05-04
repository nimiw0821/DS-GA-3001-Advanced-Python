# DS-GA-3001-Advanced-Python

Dataset: https://grouplens.org/datasets/movielens/latest/ 

*Small one*: 100,000 ratings and 3,600 tag applications applied to 9,000 movies by 600 users.

## Content-based Recommender

Time for creating a similarity matrix and generate 10 recommendations for each of the 3 movies:

- `linear_kernel`: 0.895984
- `cosine_similarity`: 0.937319
- `pairwise_distance`: 1.472686
- `numpy.dot`: 0.553272
- product: 0.539438
- operator: 0.539446
- nested for loops: 3196.772852
- parallel computing: TODO


## ALS Recommender

Time for training models:

- `Basic ALS`: 470s
- `ALS with Conjugate Gradient`: 74s
- `ALS with Conjugate Gradient using Numba JIT (C compiler)`: 10s