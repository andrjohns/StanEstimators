# Unconstrain all parameter draws.

Unconstrain all parameter draws.

## Usage

``` r
unconstrain_draws(stan_object, draws = NULL)

# S4 method for class 'StanBase'
unconstrain_draws(stan_object, draws = NULL)

# S4 method for class 'StanOptimize'
unconstrain_draws(stan_object, draws = NULL)
```

## Arguments

- stan_object:

  A `StanBase` object.

- draws:

  (optional) A `posterior::draws_*` object to be unconstrained (instead
  of the draws in the `StanBase` object).
