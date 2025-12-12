# Calculate approximate leave-one-out cross-validation (LOO-CV) for a model.

Calculate approximate leave-one-out cross-validation (LOO-CV) for a
model.

## Usage

``` r
# S3 method for class 'StanBase'
loo(x, pointwise_ll_fun, additional_args = list(), moment_match = FALSE, ...)
```

## Arguments

- x:

  A `StanBase` object.

- pointwise_ll_fun:

  Function that calculates the pointwise log-likelihood given a vector
  of parameter values.

- additional_args:

  List of additional arguments to be passed to `pointwise_ll_fun`.

- moment_match:

  (logical) Whether to use a
  [moment-matching](https://mc-stan.org/loo/reference/loo_moment_match.html)
  correction for problematic observations.

- ...:

  Additional arguments to be passed to
  [`loo::loo()`](https://mc-stan.org/loo/reference/loo.html).
