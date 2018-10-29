# Machine Learning Project: Finding the Higgs Boson

The Higgs boson is an elementary particle in the Standard Model of physics which explains why other particles have mass. Since it decays rapidly, scientists do not observe it directly, but rather measure its decay signature. By applying binary classification techniques, using original data from CERN, we can predict whether the decay signature of a collision event was a signal from a Higgs boson, or something else.  

This is the aim of Project 1 in the Machine Learning course CS-433 at EPFL. The [project description](./project1/project1_description.pdf) can be found in PDF format. 

The project includes a [Kaggle competition](https://www.kaggle.com/c/epfml18-higgs), similar to the [Higgs Boson Machine Learning Challenge](https://www.kaggle.com/c/Higgs-boson) (2014).   

## Getting Started

First, you should place the `train.csv` and `test.csv` in a `data` folder at the root of the project. 

## Scripts

#### `implementations.py` 

Contains the six regression methods needed for the project,

* **`least_squares_GD`**: Linear regression using gradient descent
* **`least_squares_SGD`**: Linear regression using stochastic gradient descent
* **`least_squares`**: Least squares regression using normal equations
* **`ridge_regression`**: Ridge regression using normal equations
* **`logistic_regression`**: Logistic regression using stochastic gradient descent
* **`reg_logistic_regression`**: Regularized logistic regression using stochastic gradient descent

along with necessary helper functions,

* **`compute_loss`**
* **`compute_gradient`**
* etc.

#### `proj1_helpers.py` 

Contains helper functions,

* **`load_csv_data`**: Loads data
* **`predict_labels`**: Generates class predictions
* **`create_csv_submission`**: Creates an output file in CSV format for submission to Kaggle
* **`build_poly`**: Polynomial basis functions
* **`split_data`**: Split the dataset based on a split ratio

#### `data_cleaning.py`

Contains functions to clean the dataset,

* **`create_poly_features`**: Creates an array that contains the original data with polynomials
* **`replace_undefined_with_nan`**: Turnes undefined value to NaN
* **`replace_undefined_with_mean`**: Replace undefined value with the mean
* **`replace_undefined`**: replace undefined values with a parameter
* **`mean_std_normalization`**: Normalize a data matrix
* **`mean_std_unnormalize`**: Returns a data matrix unnormalized

#### `run.py`

Script that generates the exact CSV file submitted on Kaggle. 

