#!/usr/bin/env python3

from __future__ import print_function, division

from itertools import product
import numpy as np

from cross_validation import cross_validate
from cross_validation import data_splitters as ds


def mle(training_data, lamdas, params, state):
    # This is probably more efficiently solved using two separate
    # cross_validate calls, but it's a simple way of checking whether using
    # multiple lamda values works
    (a1, a2) = params
    (lamda1, lamda2) = lamdas
    n = training_data.shape[1]
    avg_data = training_data.mean(axis=1)
    x_hat_1 = a1 * avg_data[0] / (a1**2 + lamda1**2 / n)
    x_hat_2 = a2 * avg_data[1] / (a2**2 + lamda2**2 / n)
    return np.array([x_hat_1, x_hat_2]), state


def l2_error(test_data, x_hat, lamdas_list, params):
    (a1, a2) = params
    na = np.newaxis
    a_x_hat = np.array([a1, a2])[:, na] * x_hat  # (dim, lamdas)
    # test_data is (dim, num_test)
    return np.mean((test_data[:, :, na] - a_x_hat[:, na, :])**2, axis=(0, 1))


def test_cv():
    n = 100
    x = np.array([1, 1])
    a1 = 10
    a2 = 25

    y = np.empty((2, n))
    np.random.seed(42)
    y[0] = a1 * x[0] + 3 * np.random.randn(n)
    y[1] = a2 * x[0] + 5 * np.random.randn(n)

    k = 5
    splitter = ds.KFold(n, k, randomize=True)

    params = (a1, a2)
    lamda1 = np.array([0.3, 3, 30])
    lamda2 = np.array([0.5, 5, 50])
    lamdas_list = [lamda1, lamda2]

    for verbose, save_x_hats in product([True, False], [True, False]):
        ret = cross_validate(y, splitter, mle, params, lamdas_list,
                             l2_error, params, verbose, save_x_hats)

        if save_x_hats:
            (lamda_stars, lamda_star_indices, error, mean_error, x_hats) = ret
        else:
            (lamda_stars, lamda_star_indices, error, mean_error) = ret

        print(mean_error)
        assert lamda_star_indices[0] == 1
        assert lamda_star_indices[1] == 1
