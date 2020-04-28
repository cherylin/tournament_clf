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

Outputs:
- in outputs/setting_1/ folder
- accuracy_comparison.csv compares accuracies of four models in each n,p parameter setting
- 4 csv that record the coefficients of 4 models in each parameter setting
- 4 plots that describes the trends of accuracy data with respect to change of n
- sigma_data.csv, X_data.csv, y_label.csv, beta_data.csv that record the data generated at n = 80, these are used for deduce optimal classifier

Observation:
- Comparing accuracy data, we can see almost all the models works well on separating two classes.
- At situation when n << p, tournament classifier(tc), dlda and svm works slightly better than the lasso model.

## Setting 2: Beta-sparsity

Fix all parameters but beta. We tune the sparsity of beta (i.e. number of 0 entries in beta vector), but still assuming that the non-zeros entries follow the uniform distribution between [-1, 1] as mentioned in basic setting.

To tune the sparsity, use scipy function: scipy.sparse.random(1, p, **density=d**, dtype=float, random_state=None, data_rvs=scipy.stats.uniform(loc=-1, scale=2).rvs), where higher density means less sparsity.

We try density = [0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 0.9]

Outputs:
- in outputs/setting_2/ folder
- accuracy_comparison.csv that compares accuracies of four models in each parameter setting
- 4 csv that record the model coefficients of 4 models in each parameter setting
- 4 plots that describes the trends of training and testing accuracy data with respect to each density
- generated_beta_density=m_nth-example.csv record the beta vector generated in every 20th dataset with density m.

Observation:
- At low density value, the beta vector has higher sparsity, thus the data has less signals. All the models' test accuracy are around 60%
- As density increases, accuracy generally increases for all models. While, tc, svm and dlda works better than lasso because they reach higher accuracy at the highest density value. But SVM and DLDA has a faster rate of increasing accuracy than TC.

## Setting 3: Beta-weights

Fix all parameters but beta. This time we fix sparsity, so just use function as in the basic setting: np.random.uniform(a, b, size=(1, p)), but substitute tuple [a, b] to each of [(-1, 1), (-3, 3), (-10, 10), (-0.1, 0.1), (0, 0.01), (-5, 0.001)]

outputs:
- in outputs/setting_3/ folder
- accuracy_comparison.csv that compares accuracies of four models in each parameter setting
- 4 files that record the model coefficients of 4 models in each parameter setting
- generated_beta_range=(a,b)_ith-example.csv record the beta vector generated in every 20th dataset with range=(a,b).

Observation:
- Comparing accuracy data, it is found that tc surprisingly doesn't work well when the range is (-5, 0.001). It has test accuracy 0.32, comparing to others with test accuracy almost 1. And all models have much lower test accuracy when range=(-0.1, 0.1) and (0, 0.01).

## Setting 4: Combine setting 2 and setting 3

We try combinations of density in [0.1, 0.9] and (a, b) in [(-0.1, 0.1), (-10, 10), (-5, 0.001)]. So basically 6 combinations

outputs:
- in outputs/setting_4/ folder
- accuracy_comparison.csv that compares accuracies of four models in each parameter setting
- 4 txt files that record the model coefficients of 4 models in each parameter setting
- 4 plots that describes the trends of training and testing accuracy data with respect to each density
- generated_beta_combination=j_ith-example.csv record the beta vector generated in every 20th dataset with jth combination of parameter setting as specified above.

Observation:
- With different density value, tc still doesn't work well at range=(-5, 0.001) comparing to others.
- At parameter setting: density = 0.9, range = (-0.1, 0.1), tc has the highest test accuracy than all other 3 models.

## Setting 5: number of pairs of predictors that have covariance

Here we consider changing sigma. We allow the variance of each dimension not to be between 0 and 1 and allow some correlations between different predictors. We restrict the magnitudes of correlations to be in range [0, 1]. So sigma is not identity matrix in this setting. We use make_sparse_spd_matrix(p, alpha=a, smallest_coef=-1, largest_coef=1, norm_diag=False) to generate sigma. Here we change parameter alpha, which represents the probability that an entry of sigma is 0. So the higher alpha, the sparser the sigma matrix is, meaning the less number of pairs of predictors are having correlation (i.e. more independent predictors). Vice versa.

We try alpha = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

outputs:
- in outputs/setting_5/ folder
- accuracy_comparison.csv that compares accuracies of four models in each parameter setting
- 4 csv files that record the model coefficients of 4 models in each parameter setting
- 4 plots that describes the trends of training and testing accuracy data with respect to each alpha value
- sigma_alpha=a_ith-example.csv record the sigma matrix generated in every 20th dataset with alpha = a.

Observation:
- At low sparsity setting, where there are more pairs of dimension that has correlations with each other, tc has higher test accuracy than lasso, but lower than svm and dlda.
- As sparsity increases, dimensions are more independent, all 4 models' test accuracy show increasing trends.

## Setting 6: magnitude of covariance

Similar to setting 4, but we change the range of smalles_coef and largest_coef in the make_sparse_spd_matrix()
function. So this change the range of possible covariance. We fix alpha = 0.5. And try different (smalles_coef, largest_coef) tuple. [(-1, 1), (-5, 1), (-1, 5), (-10, 10), (-1, 10), (-1, 20)]

outputs:
- in outputs/setting_6/ folder
- accuracy_comparison.csv that compares accuracies of four models in each parameter setting
- 4 csv files that record the model coefficients of 4 models in each parameter setting
- range=(a,b)_ith-example.csv record the sigma matrix generated in every 20th dataset with range = (a,b) as described above.

Observation:
- **The test accuracy of tc is the highest among 4 models for all parameter tuple except (-1, 1).**

## Setting 7: Combine setting 5 and setting 6

We try exhaustive combinations between alpha and (smalles_coef, largest_coef) tuple. We try alpha in [0.1, 0.9] and (smalles_coef, largest_coef) in [(-1, 1), (-10, 10), (-1, 20), (-5, 1)]

outputs:
- in outputs/setting_7/ folder
- accuracy_comparison.csv that compares accuracies of four models in each parameter setting
- 4 csv files that record the model coefficients of 4 models in each parameter setting
- generated_sigma_combination=j_ith-example.csv record the sigma matrix generated in every 20th dataset with jth combination of parameters as described above.

Observation:
- **At tuple (-10, 10) and (-1, 20), tc has the highest test accuracy among the 4 models.**

## Implementation:

7 python files to testing four models: setting_*.py

helper modules: 

generate_data.py -> used to generate X, y data

models.py -> interface for training and validating four models

utils.py -> provide a way to preprocess matrix in files

derive_coefs -> find recorded X,y,sigma,beta data for deriving optimal(Bayes) Classifier, learn an optimal linear model

compare_coefs -> used to compare optimal linear model with other 4 models to find the one with the best consistency/performance

# To find the optimal classifier(Bayes Classifier):

In derive_coefs.py, we first compute P(x|y=1) and P(x|y=-1) for each example x, assuming the prior probability P(y=1)=prob, where prob is the probability we used to generate dataset. (In basic setting P(y=1)=0.5). Then we apply bayes rule to compute P(y=1|x). After this, we now obtain an n*1 vector Y' where each ith entry represent the probability of ith example to be class 1. Then we fitted a linear model, where a coefficient vector v is the least-square solution of Xv = Y. This v is the optimal coefficient we obtained.

Then we used this coefficient to compare with the coefficients of other models.

Note: 
- X|Y=1 ~ Multivariate Normal(beta, sigma)
- X|Y=1 ~ Multivariate Normal(-beta, sigma)
- So P(x|y=1) is just the pdf in corresponding generative model.

- Bayesian rule: 
- $$P(y=1|x) = \frac{P(x|y=1)P(y=1)}{P(x|y=1)*P(y=1)+P(x|y=-1)*P(y=-1)}$$

To be continued.

