# Calculate the log probability of the model given a vector of unconstrained variables.

Calculate the log probability of the model given a vector of
unconstrained variables.

## Usage

``` r
log_prob(stan_object, unconstrained_variables, jacobian = TRUE)

# S4 method for class 'StanBase'
log_prob(stan_object, unconstrained_variables, jacobian = TRUE)
```

## Arguments

- stan_object:

  A `StanBase` object.

- unconstrained_variables:

  Vector of unconstrained variables.

- jacobian:

  Whether to include the Jacobian adjustment.
