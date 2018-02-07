#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np

from cross_validation import cross_validate
from cross_validation import data_splitters as ds


def test_single_fold():
    data = list(range(10))

    for i in range(10):
        splitter = ds.SingleFold(10, i)
        assert splitter.num_folds == 1
        for train, test in splitter:
            training_data = data[train]
            test_data = data[test]
            assert training_data == list(range(i))
            assert test_data == list(range(i, 10))


def test_k_fold():
    data = np.arange(10)

    for k in range(1, 11):
        splitter = ds.KFold(10, k)
        assert splitter.num_folds == k
        for train, test in splitter:
            agg_data = set(data[train]) | set(data[test])
            assert agg_data == set(data)
            assert len(agg_data) == 10
