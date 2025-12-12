# Calculate the log probability, its gradient, and its hessian matrix of the model given a vector of unconstrained variables.

Calculate the log probability, its gradient, and its hessian matrix of
the model given a vector of unconstrained variables.

## Usage

``` r
hessian(stan_object, unconstrained_variables, jacobian = TRUE)

# S4 method for class 'StanBase'
hessian(stan_object, unconstrained_variables, jacobian = TRUE)
```

## Arguments

- stan_object:

  A `StanBase` object.

- unconstrained_variables:

  Vector of unconstrained variables.

- jacobian:

  Whether to include the Jacobian adjustment.
