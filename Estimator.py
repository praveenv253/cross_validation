#!/usr/bin/env python3

"""
Provides the class Estimator, which describes an interface that all estimators
called by the cross validation function should provide.

Some derived estimators are included for convenience, and as a reference.
"""

from __future__ import print_function, division

from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt


class Estimator(ABC):
    """
    Defines an estimator: an object that takes some "observations" (training
    data), some system properties and some regularization parameter(s) as input
    and produces an "estimate" of the underlying state of the system as output.

    The above description is one of many ways of seeing the following problem:
              +---+
        x --> | H | --> y
              +---+
    We wish to estimate x, given certain properties of H and the observations
    y. Presumably, some noise has also been added in as the quantity of
    interest, x, passed through the system.

    An object of this class should satisfy one property:
    1. It should be callable (i.e. it should define a function __call__), which
       takes training_data and lamdas as parameters and produces an estimate
       x_hat.

    The object is usually initialized with system properties.

    It might be useful for the object to maintain some internal pre-computed
    variables upon initialization for speed.a
    """

    @abstractmethod
    def __call__(self, training_data, lamdas):
        pass


class LeastSquaresL2(Estimator):
    """
    Defines the L2-regularized least squares estimator.

    The system H is assumed to be a linear function, so that
        y = H*x + n
    where n is iid gaussian noise.
    """

    def __init__(self, H):
        self.H = H
        # XXX HOW DO WE DO CACHES IN OBJECTS? IS IT EFFICIENT?
        self.cache = {'H_Hh': np.dot(H, H.conj().T)}

    def __call__(self, training_data, lamdas):
        (lamda,) = lamdas # Needs only a 1-D lamda value
        if 
        b = training_data.mean(1)   # Average training data
        state = (b, H_Hh)
        # XXX It is 3:20AM, so I'm giving up.
