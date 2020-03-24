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

======n = 20 p = 200 ======

lasso: train accuracy= 1.0 and test accuracy= 0.9466666666666665

======n = 40 p = 200 ======

lasso: train accuracy= 1.0 and test accuracy= 0.9933333333333334

======n = 50 p = 200 ======

lasso: train accuracy= 1.0 and test accuracy= 0.9993333333333334

======n = 80 p = 200 ======

lasso: train accuracy= 1.0 and test accuracy= 0.9995833333333334

======n = 100 p = 200 ======

lasso: train accuracy= 1.0 and test accuracy= 1.0

======n = 200 p = 200 ======

lasso: train accuracy= 1.0 and test accuracy= 1.0

======n = 300 p = 200 ======

lasso: train accuracy= 1.0 and test accuracy= 1.0

======n = 400 p = 200 ======

lasso: train accuracy= 1.0 and test accuracy= 1.0

======n = 500 p = 200 ======

lasso: train accuracy= 1.0 and test accuracy= 1.0

======n = 600 p = 200 ======

lasso: train accuracy= 1.0 and test accuracy= 1.0

======n = 20 p = 200 ======

dlda: train accuracy= 1.0 and test accuracy= 1.0

======n = 40 p = 200 ======

dlda: train accuracy= 1.0 and test accuracy= 1.0

======n = 50 p = 200 ======

dlda: train accuracy= 1.0 and test accuracy= 1.0

======n = 80 p = 200 ======

dlda: train accuracy= 1.0 and test accuracy= 1.0

======n = 100 p = 200 ======

dlda: train accuracy= 1.0 and test accuracy= 1.0

======n = 200 p = 200 ======

dlda: train accuracy= 1.0 and test accuracy= 1.0

======n = 300 p = 200 ======

dlda: train accuracy= 1.0 and test accuracy= 1.0

======n = 400 p = 200 ======

dlda: train accuracy= 1.0 and test accuracy= 1.0

======n = 500 p = 200 ======

dlda: train accuracy= 1.0 and test accuracy= 1.0

======n = 600 p = 200 ======

dlda: train accuracy= 1.0 and test accuracy= 1.0

======n = 20 p = 200 ======

svm: train accuracy= 1.0 and test accuracy= 1.0

======n = 40 p = 200 ======

svm: train accuracy= 1.0 and test accuracy= 1.0

======n = 50 p = 200 ======

svm: train accuracy= 1.0 and test accuracy= 1.0

======n = 80 p = 200 ======

svm: train accuracy= 1.0 and test accuracy= 1.0

======n = 100 p = 200 ======

svm: train accuracy= 1.0 and test accuracy= 1.0

======n = 200 p = 200 ======

svm: train accuracy= 1.0 and test accuracy= 1.0

======n = 300 p = 200 ======

svm: train accuracy= 1.0 and test accuracy= 1.0

======n = 400 p = 200 ======

svm: train accuracy= 1.0 and test accuracy= 1.0

======n = 500 p = 200 ======

svm: train accuracy= 1.0 and test accuracy= 1.0

======n = 600 p = 200 ======

svm: train accuracy= 1.0 and test accuracy= 1.0

python3 np_ratio.py 51.59s user 5.04s system 206% cpu 27.393 total



## Covariance

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

10. change sparsity first, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9

### Outputs:

without normalizing the diagonals to 1

======alpha = 0.1 ======

lasso: train accuracy= 1.0 and test accuracy= 0.9126666666666665

/Users/linqian/Desktop/wn2020/stats489/env/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:476: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 0.0077840444798695785, tolerance: 0.006908571428571429

 positive)

======alpha = 0.2 ======

lasso: train accuracy= 1.0 and test accuracy= 0.8846666666666664

======alpha = 0.3 ======

lasso: train accuracy= 0.9997142857142858 and test accuracy= 0.8463333333333333

======alpha = 0.4 ======

lasso: train accuracy= 0.9974285714285713 and test accuracy= 0.8129999999999994

======alpha = 0.5 ======

lasso: train accuracy= 0.9901428571428567 and test accuracy= 0.7973333333333332

======alpha = 0.6 ======

lasso: train accuracy= 0.9742857142857138 and test accuracy= 0.7630000000000002

======alpha = 0.7 ======

lasso: train accuracy= 0.9574285714285705 and test accuracy= 0.7779999999999992

======alpha = 0.8 ======

lasso: train accuracy= 0.9387142857142847 and test accuracy= 0.7643333333333331

======alpha = 0.9 ======

lasso: train accuracy= 0.9007142857142852 and test accuracy= 0.7613333333333331

======alpha = 0.1 ======

dlda: train accuracy= 0.8452857142857139 and test accuracy= 0.763333333333333

======alpha = 0.2 ======

dlda: train accuracy= 0.871571428571428 and test accuracy= 0.7950000000000003

======alpha = 0.3 ======

dlda: train accuracy= 0.9194285714285709 and test accuracy= 0.8523333333333336

======alpha = 0.4 ======

dlda: train accuracy= 0.9308571428571425 and test accuracy= 0.8616666666666666

======alpha = 0.5 ======

dlda: train accuracy= 0.9595714285714287 and test accuracy= 0.9009999999999997

======alpha = 0.6 ======

dlda: train accuracy= 0.9825714285714281 and test accuracy= 0.944666666666667

======alpha = 0.7 ======

dlda: train accuracy= 0.9975714285714283 and test accuracy= 0.980666666666667

======alpha = 0.8 ======

dlda: train accuracy= 0.9997142857142858 and test accuracy= 0.9986666666666667

======alpha = 0.9 ======

dlda: train accuracy= 1.0 and test accuracy= 1.0

======alpha = 0.1 ======

svm: train accuracy= 1.0 and test accuracy= 0.9780000000000004

======alpha = 0.2 ======

svm: train accuracy= 1.0 and test accuracy= 0.9746666666666671

======alpha = 0.3 ======

svm: train accuracy= 1.0 and test accuracy= 0.9723333333333336

======alpha = 0.4 ======

svm: train accuracy= 1.0 and test accuracy= 0.9756666666666672

======alpha = 0.5 ======

svm: train accuracy= 1.0 and test accuracy= 0.9826666666666671

======alpha = 0.6 ======

svm: train accuracy= 1.0 and test accuracy= 0.9840000000000002

======alpha = 0.7 ======

svm: train accuracy= 1.0 and test accuracy= 0.9923333333333335

======alpha = 0.8 ======

svm: train accuracy= 1.0 and test accuracy= 0.9986666666666667

======alpha = 0.9 ======

svm: train accuracy= 1.0 and test accuracy= 1.0



with normalizing the diagonals to 1's

======alpha = 0.1 ======

lasso: train accuracy= 1.0 and test accuracy= 1.0

======alpha = 0.2 ======

lasso: train accuracy= 1.0 and test accuracy= 1.0

======alpha = 0.3 ======

lasso: train accuracy= 1.0 and test accuracy= 0.9996666666666667

======alpha = 0.4 ======

lasso: train accuracy= 1.0 and test accuracy= 0.9953333333333334

======alpha = 0.5 ======

lasso: train accuracy= 0.9985714285714287 and test accuracy= 0.9863333333333337

======alpha = 0.6 ======

lasso: train accuracy= 0.992285714285714 and test accuracy= 0.9636666666666669

======alpha = 0.7 ======

lasso: train accuracy= 0.9859999999999995 and test accuracy= 0.9366666666666671

======alpha = 0.8 ======

lasso: train accuracy= 0.9598571428571429 and test accuracy= 0.8919999999999999

======alpha = 0.9 ======

lasso: train accuracy= 0.9007142857142852 and test accuracy= 0.8216666666666664

======alpha = 0.1 ======

dlda: train accuracy= 1.0 and test accuracy= 1.0

======alpha = 0.2 ======

dlda: train accuracy= 1.0 and test accuracy= 1.0

======alpha = 0.3 ======

dlda: train accuracy= 1.0 and test accuracy= 1.0

======alpha = 0.4 ======

dlda: train accuracy= 1.0 and test accuracy= 1.0

======alpha = 0.5 ======

dlda: train accuracy= 1.0 and test accuracy= 1.0

======alpha = 0.6 ======

dlda: train accuracy= 1.0 and test accuracy= 1.0

======alpha = 0.7 ======

dlda: train accuracy= 1.0 and test accuracy= 1.0

======alpha = 0.8 ======

dlda: train accuracy= 1.0 and test accuracy= 1.0

======alpha = 0.9 ======

dlda: train accuracy= 1.0 and test accuracy= 1.0

======alpha = 0.1 ======

svm: train accuracy= 1.0 and test accuracy= 1.0

======alpha = 0.2 ======

svm: train accuracy= 1.0 and test accuracy= 1.0

======alpha = 0.3 ======

svm: train accuracy= 1.0 and test accuracy= 1.0

======alpha = 0.4 ======

svm: train accuracy= 1.0 and test accuracy= 1.0

======alpha = 0.5 ======

svm: train accuracy= 1.0 and test accuracy= 1.0

======alpha = 0.6 ======

svm: train accuracy= 1.0 and test accuracy= 1.0

======alpha = 0.7 ======

svm: train accuracy= 1.0 and test accuracy= 1.0

======alpha = 0.8 ======

svm: train accuracy= 1.0 and test accuracy= 1.0

======alpha = 0.9 ======

svm: train accuracy= 1.0 and test accuracy= 1.0



## Beta

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
    - density = [0.01, 0.5, 0.1, 0.2, 0.5, 0.7, 0.9]
    - small : uniform btw [0, 1]
    - large: uniform btw [1, 2], [2, 3], [3, 4] .. 

### Outputs:

./outputs/coef_impact_sparsity_{}

output_beta_impact_sparsity_{}.png

small values : [-1, 1]

======density = 0.01 ======

lasso: train accuracy= 0.9702857142857135 and test accuracy= 0.7126666666666667

======density = 0.5 ======

lasso: train accuracy= 1.0 and test accuracy= 0.9993333333333334

======density = 0.1 ======

lasso: train accuracy= 0.9972857142857141 and test accuracy= 0.9803333333333336

======density = 0.2 ======

lasso: train accuracy= 0.999857142857143 and test accuracy= 0.9966666666666668

======density = 0.5 ======

lasso: train accuracy= 1.0 and test accuracy= 0.9993333333333334

======density = 0.7 ======

lasso: train accuracy= 1.0 and test accuracy= 1.0

======density = 0.9 ======

lasso: train accuracy= 1.0 and test accuracy= 0.9996666666666667

======density = 0.01 ======

dlda: train accuracy= 0.8734285714285711 and test accuracy= 0.7133333333333336

======density = 0.5 ======

dlda: train accuracy= 1.0 and test accuracy= 1.0

======density = 0.1 ======

dlda: train accuracy= 0.9961428571428569 and test accuracy= 0.9890000000000003

======density = 0.2 ======

dlda: train accuracy= 1.0 and test accuracy= 0.9993333333333334

======density = 0.5 ======

dlda: train accuracy= 1.0 and test accuracy= 1.0

======density = 0.7 ======

dlda: train accuracy= 1.0 and test accuracy= 1.0

======density = 0.9 ======

dlda: train accuracy= 1.0 and test accuracy= 1.0

======density = 0.01 ======

svm: train accuracy= 1.0 and test accuracy= 0.6113333333333337

======density = 0.5 ======

svm: train accuracy= 1.0 and test accuracy= 1.0

======density = 0.1 ======

svm: train accuracy= 1.0 and test accuracy= 0.9646666666666668

======density = 0.2 ======

svm: train accuracy= 1.0 and test accuracy= 0.9976666666666668

======density = 0.5 ======

svm: train accuracy= 1.0 and test accuracy= 1.0

======density = 0.7 ======

svm: train accuracy= 1.0 and test accuracy= 1.0

======density = 0.9 ======

svm: train accuracy= 1.0 and test accuracy= 1.0

