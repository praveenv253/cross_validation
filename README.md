[![Build Status](https://travis-ci.org/praveenv253/cross_validation.svg?branch=master)](https://travis-ci.org/praveenv253/cross_validation)

Cross Validation
================

This is a simplistic but general-purpose cross-validation module.

L2-regularized linear regression: an example
--------------------------------------------

Consider solving the noisy linear system `y = H x + n`. You receive multiple
observations `{y_i}, i=1..M` and you must reconstruct `x`.

Write a simple minimum-norm estimator as follows:

```python
def mne(training_data, lamdas, params, state=None):
    (H, ) = params
    (lamda, ) = lamdas   # This estimator takes only a 1-D lamda
    # State is used to avoid re-computing data that is common across runs
    if state is None:
        H_Hh = np.dot(H, H.conj().T)
        b = training_data.mean(1)   # Average training data
        state = (b, H_Hh)
    else:
        (b, H_Hh) = state
    A = (H_Hh + lamda * np.identity(H.shape[0]))
    x_hat = np.dot(H.conj().T, la.solve(A, b))
    return (x_hat, state)
```

Say you want to use single-fold cross validation (i.e. plain old validation).

Ready your data splitter and your parameter space:

```python
lamdas_mne = np.array([exponent * mantissa
                       for exponent in np.logspace(-10, -3, 8)
                       for mantissa in np.arange(1, 10)])
data_splitter = SingleFold(num_data, num_train)
```

Define the loss-function that describes how to evaluate the output of the
estimator against the testing data

```python
def error_l2(test_data, x_hat, lamdas_list, error_fn_params):
    """
    Computes mean L2 norm error against test data
    """
    (H, ) = error_fn_params
    y_hat = np.dot(H, x_hat)[:, np.newaxis, :]
    return np.mean(npla.norm(test_data[:, :, np.newaxis] - y_hat, axis=0)**2,
                   axis=0)
```

Finally, call `cross_validate` to get your optimal lambda value and testing
error.

```python
ret = cross_validate(all_data, data_splitter, mne, (H,), (lamdas_mne,),
                     error_l2, (H,))
# Unpack return values
((lamda_star,), (lamda_star_index,), error, mean_error) = ret
```
