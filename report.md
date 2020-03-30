## N vs P

### Parameters:

1. probability of y = 1 is 0.5

2. beta ~ uniform(-1, 1) (this describes the sepratability of different dimensions)
3. U ~ MVN(mu, sigma)
4. mu = 0
5. sigma = identity matrix
6. total number of datasets, N = 100
7. separate 70% of a dataset as the training data, 30% as the validation data

6. number of example in one dataset, n = [20, 40, 50, 80, 100, 200, 300, 400, 500]

8. number of predictors/dimension of the data, p = 200

### Outputs:

1. the average training & testing accuracy over 100 datasets for each setting for each model
2. a plot that describes how training and testing accruacy changes along with n for each of model

3. Coefficients



## Covariance-correlation involved

### Parameters

1. probability of y = 1 is 0.5

2. beta ~ uniform(-1, 1) (this describes the sepratability of different dimensions)
3. U ~ MVN(mu, sigma)
4. mu = 0
5. sigma = make_sparse_spd_matrix(p, *alpha*=0.5, *smallest_coef*=0, *largest_coef*=1), this generate a symmetric, spd matrix with certain sparsity
6. total number of datasets, N = 100
7. separate 70% of a dataset as the training data, 30% as the validation data
8. number of examples in one dataset, n = 100 ( < 200 )
9. number of predictors/dimension of the data, p = 200

10. change sparsity first, alpha is one of 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9

### Outputs:

with normalizing the diagonals to 1's

1. the average training & testing accuracy over 100 datasets for each setting for each model
2. a plot that describes how training and testing accruacy changes along with alpha for each of model
3. Coefficients
4. Since for each alpha, we generate 100 sigma, so we record the sigma once every 20 of them



## Beta-sparsity

### Parameters:

1. probability of y = 1 is 0.5

2. beta: `scipy.sparse.random`**(**1, p, density=0.01,  format='coo', dtype=None,random_state=None, data_rvs=None)
3. U ~ MVN(mu, sigma)
4. mu = 0
5. sigma = identity matrix
6. total number of datasets, N = 100
7. separate 70% of a dataset as the training data, 30% as the validation data
8. number of examples in one dataset, n = 100 ( < 200 )
9. number of predictors/dimension of the data, p = 200

10. 4 possible types of beta:
    - sparse + small values
    - sparse + large entries
    - dense + small entries
    - dense + large entries
    - **density = [0.01, 0.5, 0.1, 0.2, 0.5, 0.7, 0.9]**
    - small : uniform btw [-1, 1]
    - large: uniform btw [1, 2], [2, 3], [3, 4] .. 
    - **HERE WE CHOOSE TO FIX entries being uniformly distributed within [-1, 1] and control the density**

### Outputs:

1. the average training & testing accuracy over 100 datasets for each setting for each model (accuracy_comparison.csv)
2. a plot that describes how training and testing accruacy changes along with density for each of model

3. Coefficients
4. Since for each density, we generate 100 beta vectors, so we record the sigma once every 20 of them



## Beta-weight

### Parameters

1. probability of y = 1 is 0.5

2. beta: `scipy.sparse.random`**(**1, p, density=0.01,  format='coo', dtype=None,random_state=None, data_rvs=None)
3. U ~ MVN(mu, sigma)
4. mu = 0
5. sigma = identity matrix
6. total number of datasets, N = 100
7. separate 70% of a dataset as the training data, 30% as the validation data
8. number of examples in one dataset, n = 100 ( < 200 )
9. number of predictors/dimension of the data, p = 200

10. 4 possible types of beta:
    - sparse + small values
    - sparse + large entries
    - dense + small entries
    - dense + large entries
    - density = [0.01, 0.5, 0.1, **0.2**, 0.5, 0.7, 0.9]
    - small : uniform btw [-1, 1]
    - **large: uniform btw [0, 2], [0, 3], [0, 4] .. [0, 8]**
    - HERE WE CHOOSE TO FIX density = 0.2 and control entries being uniformly distributed within  **[0, 2], [0, 3], [0, 4] .. [0, 8]**

### Outputs:

1. the average training & testing accuracy over 100 datasets for each setting for each model for previously described each setting (accuracy_comparison.csv)
2. a plot that describes how training and testing accruacy changes along with weights of entries for each of model

3. Coefficients
4. Since for each setting, we generate 100 beta vectors, so we record the sigma once every 20 of them

