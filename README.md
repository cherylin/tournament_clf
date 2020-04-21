# About
This is a data analysis project built for testing Tournament Classifier, a linear classifier for high dimensional data.

## To run the python files:
The project is run in Python3 environment with the following packages:

|Package|Version|
|---|---|
|matplotlib|3.1.3|
|mlpy|3.5.0|
|numpy|1.18.1|
|scikit-learn|0.22.1|
|scipy|1.4.1|

## Basic data(X) generation process and parameters used:
### Main method:

X = Y*beta + U, where:

X has n examples (n rows) and p dimensions (p columns)

dim(Y) = (n,1); Entries of Y are either 1 or -1

beta: a vector in dimension=(1, p), following uniform(-1, 1) (this describes the sepratability of different dimensions). In implementation, I used numpy function: np.random.uniform(-1, 1, size=(1, p)).

U ~ Multivariate normal(mu, sigma), where mu is a vector in dimension=(1,p), sigma is the covariance matrix
with dimension=(p,p)

### Basic setting of parameters:

**This is the part to change for later experiments**

n = 50 = number of examples in training set

p = 100 = number of predictors/dimension of data

prob = 0.5 = P(Y = 1)

mu = zero vector (i.e. np.zeros(p))

sigma = identity matrix (i.e. assume independence between predictors and standard deviation of each dimension)

beta = vector with p entries, each entry following uniform distribution in range [-1, 1]

### Other Implementation settings:

N = 100 = number of datasets generated fro each parameter setting

train_percent = 0.8 = proportion of training data that is used to train the model (i.e. 80% for training, 20% for validation set)

## Comparing models based on given data

Procedure1: Generate 100 training dataset under a given parameter setting, then for each of the 4 models (svm, lasso, dlda, tournament_clf), fit them over these 100 datasets and take the average of their performance in terms of accuracy over 100 datasets. Take a mean of 100 coefficient vectors for each model and save them into files.

Procedure2: Repeat procedure1 with different parameter settings.

## Setting 1: n vs p

Fix all other parameters as the basic setting (refer to previous section), but try n = [20, 40, 50, 80, 100, 200, 300, 400, 500]

outputs:
- in outputs/setting_1/ folder
- A csv that compares accuracies of four models in each parameter settings
- 4 csv that record the coefficients of 4 models in each parameter setting
- 4 plots that describes the trends of accuracy data with respect to change of n
- sigma_data.csv, X_data.csv, y_label.csv, beta_data.csv that record the data generated at n = 80, these are used for deduce optimal classifier

## Setting 2: Beta-sparsity

Fix all parameters but beta. We tune the sparsity of beta (i.e. number of 0 entries in beta vector), but still assuming that the non-zeros entries follow the uniform distribution between [-1, 1] as mentioned in basic setting.

To tune the sparsity, use scipy function: scipy.sparse.random(1, p, **density=d**, dtype=float, random_state=None, data_rvs=scipy.stats.uniform(loc=-1, scale=2).rvs), where higher density means less sparsity.

We try density = [0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 0.9]

outputs: in outputs/setting_2/ folder

## Setting 3: Beta-weights

Fix all parameters but beta. This time we fix sparsity, so just use function as in the basic setting: np.random.uniform(a, b, size=(1, p)), but substitute tuple [a, b] to each of [(-1, 1), (-3, 3), (-10, 10), (-0.1, 0.1), (0, 0.01), (-5, 0.001)]

outputs: in outputs/setting_3/ folder

## Setting 4: Combine setting 2 and setting 3

We try combinations of density in [0.1, 0.9] and (a, b) in [(-0.1, 0.1), (-10, 10), (-5, 0.001)]. So basically 6 combinations

outputs: in outputs/setting_4/ folder

## Setting 5: number of pairs of predictors that have covariance

Here we consider changing sigma. We allow the variance of each dimension not to be between 0 and 1 and allow some correlations between different predictors. We restrict the magnitudes of correlations to be in range [0, 1]. So sigma is not identity matrix in this setting. We use make_sparse_spd_matrix(p, alpha=a, smallest_coef=-1, largest_coef=1, norm_diag=False) to generate sigma. Here we change parameter alpha, which represents the probability that an entry of sigma is 0. So the higher alpha, the sparser the sigma matrix is, meaning the less number of pairs of predictors are having correlation (i.e. more independent predictors). Vice versa.

We try alpha = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

outputs: in outputs/setting_5/ folder

## Setting 6: magnitude of covariance

Similar to setting 4, but we change the range of smalles_coef and largest_coef in the make_sparse_spd_matrix()
function. So this change the range of possible covariance. We fix alpha = 0.5. And try different (smalles_coef, largest_coef) tuple. [(-1, 1), (-5, 1), (-1, 5), (-10, 10), (-1, 10), (-1, 20)]

outputs: in outputs/setting_6/ folder

## Setting 7: Combine setting 5 and setting 6

We try exhaustive combinations between alpha and (smalles_coef, largest_coef) tuple. We try alpha in [0.1, 0.9] and (smalles_coef, largest_coef) in [(-1, 1), (-10, 10), (-1, 20), (-5, 1)]

outputs: in outputs/setting_7/ folder

## Setting 8: Imbalanced class

Fix all other settings, but change prob = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

outputs: in outputs/setting_8/ folder

## Implementation:

8 python files to testing four models: setting_*.py

helpers: generate_data.py -> used to generate X, y data

         models.py -> interface for training and validating four models

         utils.py -> provide a way to preprocess matrix in files

         derive_coefs -> find recorded X,y,sigma,beta data for deriving optimal(Bayes) Classifier, learn an optimal linear model

         compare_coefs -> used to compare optimal linear model with other 4 models to find the one with the best consistency/performance

# Analysis and results:

Tournament Classifier:

Seems to have higher test accuracy than SVM and DLDA in setting 7, with parameters alpha = 0.9, a = -5, b = 1 / alpha = 0.9, a = -1, b = 20

Works well as SVM and DLDA in setting 8, where class labels are unbalanced.

Lasso:

Didn't converge in all parameter settings in setting 7
