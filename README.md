# DS-GA-3001-Advanced-Python
Github repo: https://github.com/nimiw0821/DS-GA-3001-Advanced-Python

Dataset: https://grouplens.org/datasets/movielens/latest/ 

*Small one*: 100,000 ratings and 3,600 tag applications applied to 9,000 movies by 600 users.

## Content-based Recommender

Time for creating a similarity matrix and generating 10 recommendations for each of the 3 movies:

|  Method  | w/ Function Calls (s)  | w/o Function Calls (s)  |
|---|---|---|
| linear_kernel  | 0.896 | 0.858  |
| cosine_similarity  |  0.937 | 0.905   |
| pairwise_distances | 1.473  | 1.406  |
| matrix multiplication (np.dot)  | 0.553  | 0.505 |
| matrix multiplication (operator) | 0.539 | 0.495 |
| nested for loops  | 3196.773 | 3098.554  |
| nested for loops (Cython) | 1478.108 | 1276.212 |

## Memory-based Recommender
|  Method  |  Time in Seconds  |
|---|---|
|  Base   |  0.031  |
|  Simplified code  | 0.021  |
|  Numba.jit()  |  0.294 |
|  Cython  | 0.019 |
|  Combined Cython and simplified code | 0.021  |

## ALS Recommender

Time for training ALS models:
|  Method  |  Time in Seconds  |
|---|---|
|  Base ALS  |  470  |
|  ALS with Conjugate Gradient (CG)  | 74  |
|  ALS with CG using Cython  |  84  |
|  ALS with CG using Numba JIT  | 10  |
