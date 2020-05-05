# DS-GA-3001-Advanced-Python

Dataset: https://grouplens.org/datasets/movielens/latest/ 

*Small one*: 100,000 ratings and 3,600 tag applications applied to 9,000 movies by 600 users.

## Content-based Recommender

Time for creating a similarity matrix and generate 10 recommendations for each of the 3 movies:

With Function Calls:

- `linear_kernel`: 0.895984s
- `cosine_similarity`: 0.937319s
- `pairwise_distance`: 1.472686s
- `numpy.dot`: 0.553272s
- product: 0.539438s
- operator: 0.539446s
- nested for loops: 3196.772852s
- using `Cython` to optimize nested for loops: 1478.108414s

Remove Function Calls:

- `linear_kernel`: 0.858322s
- `cosine_similarity`: 0.905019s
- `pairwise_distance`: 1.603787s
- `numpy.dot`: 0.505195s
- product: 0.506122s
- operator: 0.495365s

## ALS Recommender

Time for training models:

- `Basic ALS`: 470s
- `ALS with Conjugate Gradient`: 74s
- `ALS with Conjugate Gradient using Numba JIT (C compiler)`: 10s
