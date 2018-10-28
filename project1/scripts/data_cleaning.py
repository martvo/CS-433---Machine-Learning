import numpy as np
from proj1_helpers import *

UNDEFINED_VALUE = np.float64(-999.0)
COLUMN_TO_DROP = 22


def create_poly_features(data, degrees):
    """
    Creates a np.ndarray that contains the original data with polynomials from 1 to degrees
    :param data:
    :param degrees:
    :return np.ndarray:
    """
    
    """
    ************************************************************************************
    DO NOT NEED TO ADD THE 1'S AS WE ARE NORMALIZING WITH MEAN AND STD WHICH WILL MAKE TE VALUES EQUAL TO ZERO
    AND THE WEIGHT WILL NEVER PLAY A PART OF OUT MODEL AS ITS W*0
    ************************************************************************************
    """
    new_data = []
    for deg in range(1, degrees + 1):
        new_data.append(np.power(data, deg))
    return np.concatenate(new_data, axis=1)


def replace_undefoned_with_nan(data, undefined):
    """
    Turnes undefined value to NaN
    :param data:
    :param undefined:
    :return numpy.ndarray:
    """
    data[data == undefined] = np.nan
    return data


def replace_undefined_with_most_frequent(data, undefined):
    """ HOW COULD THIS BE DONE? np.bincount only works on ints
    Retrun data with undefined values replaced with most frequent value of that column
    :param data:
    :param undefined:
    :return numpy.ndarray:
    """
    for i in range(data.shape[1]):
        i_col_most_frequent = np.bincount(data[:, i]).argmax()
        data[data == undefined] = np.float64(i_col_most_frequent)
    return data


def replace_undefined(data, limit, replace_with):
    """
    Covert values equal to limit param to replace_with param
    :param data:
    :param limit:
    :param replace_with:
    :return numpy.ndarray:
    """
    data[data == limit] = np.float64(replace_with)
    # return np.clip(data, lower_limit, upper_limit)
    return data


def precentage_is_undefined(data):
    data_nan = replace_undefoned_with_nan(data, UNDEFINED_VALUE)
    return np.count_nonzero(~np.isnan(data_nan))


def mean_std_normalization(data, data_mean=[], data_std=[]):
    """
    Normalize the data matrix with (data - mean(data)) / std(data)
    :param data_std:
    :param data_mean:
    :param data:
    :return np.ndmatrix, np.ndarray, np.ndarray:
    """
    if len(data_mean) == 0 and len(data_std) == 0:
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
    return np.divide(np.subtract(data, data_mean), data_std), data_mean, data_std



def mean_std_unnormalize(data, data_mean, data_std):
    """
    Returns the data matrix unnormalized. PS: data_mean and data_std MUST be the same as the mean and std that normalized
    the data matrix
    :param data:
    :param data_mean:
    :param data_std:
    :return: np.ndmatrix
    """
    return np.add(np.multiply(data, data_std), data_mean)


if __name__ == "__main__":
    y_train, x_train, ids_train = load_csv_data("../data/train.csv")
    # y_test, x_test, ids_test = load_csv_data("../data/test.csv")

    PRI_jet_num = x_train[:, COLUMN_TO_DROP]
    x_train = np.delete(x_train, COLUMN_TO_DROP, axis=1)
    print(x_train.shape)

    x_train = replace_undefined(x_train, UNDEFINED_VALUE, 0.0)
    x_train = create_poly_features(x_train, 3)
    print(x_train.shape)

    norm_x_train, data_mean, data_std = mean_std_normalization(x_train)
    print(norm_x_train.shape)

    mean_std_normalization(np.random.rand(2, 3))

    unnorm_x_train = mean_std_unnormalize(norm_x_train, data_mean, data_std)
    print(unnorm_x_train[0][0])
    print(unnorm_x_train.shape)


# TODO:
# remove row and column with too many -999 values
# replace -999.0 with most frequent value in that column???
# IKKE GJØR: -999 om til gjennomsnitt av kolonnen -> blir 0 etter vi har standarisert det

# Found out:
# 1. all features is numpy.float64
# PRI_jet_num is a categorical value, {0, 1, 2,.....}