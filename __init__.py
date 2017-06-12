#!/usr/bin/env python3

from __future__ import print_function, division

import itertools
import numpy as np


def cross_validate(cv_data, data_splitter, estimator, estimator_params,
                   lamdas_list, error_fn, error_fn_params, verbose=True,
                   save_x_hats=False):
    """
    Compute e(lamda) for the given set of lamdas against the given estimator
    for the given set of cross-validation data using the given partition
    function (to generate the training and testing dataset)

    lamdas_list should be a tuple of lamda values for the different dimensions
    over which grid search needs to be performed.
    For e.g., for plain lasso, lamdas_list can simply be ([0.1, 1, 10], ).
    For elastic net, lamdas_list might be ([0.1, 1, 10], [0.1, 1, 10]) (for the
        two different lamdas corresponding to the l1 and l2 costs).

    Returns
    -------
    lamda_stars : tuple of double
        Values of the lamdas which minimize test error. lamda_star[i] is the
        best value out of lamdas_list[i].
    lamda_star_indices : tuple of int
        Indices corresponding to the best values of lamdas.
        lamda_star_indices[i] is the index of lamda_stars[i] in lamdas_list[i].
    error : numpy array of shape (num_folds, num_lamdas)
        For each fold, gives the error between the test data of that
        fold and the training estimate of that fold, for every value of lamda
    mean_error : numpy array of shape (num_lamdas,)
        Mean of `error` over folds.
    x_hats : numpy array of shape (num_sources, num_folds, num_lamdas)
        Returned only if save_x_hats is True.
        Value of the training estimate from each fold, for every value of
        lamda.

    Notes
    -----
    For the last three return values, note that num_lamdas is the product of
    the lengths of each list in lamdas_list. lamdas_list is ravelled using
    itertools.product, which means that the last lamda changes fastest. For
    e.g., if 2 values of lamda_1, lamda_2 and lamda_3 are provided, then they
    are ordered as 000, 001, 010, 011, 100, 101, 110, 111. A desired index can
    be computed using np.unravel_index.
    """

    lamdas_sizes = tuple(lamdas.size for lamdas in lamdas_list)
    num_lamdas = np.prod(lamdas_sizes)

    num_folds = data_splitter.num_folds
    error = np.empty((num_folds, num_lamdas))

    # Iterate over "folds"
    for k, (training_indices, test_indices) in enumerate(data_splitter):
        training_data = cv_data[..., training_indices]
        test_data = cv_data[..., test_indices]

        if verbose:
            print(k, end=' ', flush=True)

        # Ininitialize the state variable, which the estimator uses to avoid
        # re-computing certain expressions. This may change for different
        # training data, so we re-initialize it in every iteration.
        state = None

        # Find the best x for different regression parameters
        for i, lamdas in enumerate(itertools.product(*lamdas_list)):
            if i == 0:
                (x_hat_temp, state) = estimator(training_data, lamdas,
                                                estimator_params, state)
                x_hat = np.empty((x_hat_temp.size, num_lamdas),
                                 dtype=x_hat_temp.dtype)
                x_hat[:, i] = x_hat_temp
            else:
                x_hat[:, i], state = estimator(training_data, lamdas,
                                               estimator_params, state)

        error[k, :] = error_fn(test_data, x_hat, lamdas_list, error_fn_params)

        if save_x_hats:
            if k == 0:
                x_hats = np.empty((x_hat.shape[0], num_folds, num_lamdas),
                                  dtype=x_hat.dtype)
            x_hats[:, k, :] = x_hat

    # Compute mean error
    mean_error = error.mean(axis=0)

    # Hence compute best lamda values
    # XXX Check this for multi-dimensional lamdas_list!!
    lamda_star_ravelled_index = np.argmin(mean_error)
    lamda_star_indices = np.unravel_index(lamda_star_ravelled_index,
                                          lamdas_sizes)
    lamda_stars = (lamdas_list[i][lamda_star_index]
                   for (i, lamda_star_index) in enumerate(lamda_star_indices))

    if save_x_hats:
        return (lamda_stars, lamda_star_indices, error, mean_error, x_hats)
    else:
        return (lamda_stars, lamda_star_indices, error, mean_error)
