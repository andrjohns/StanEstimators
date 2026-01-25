# Getting Started with StanEstimators

``` r
library(StanEstimators)
```

## Introduction

This vignette provides a basic introduction to the features and
capabilities of the `StanEstimators` package. `StanEstimators` provides
an interface for estimating R functions using the various algorithms and
methods implemented by [Stan](https://mc-stan.org/).

The `StanEstimators` package supports all algorithms implemented by
Stan. The available methods, and their corresponding functions, are:

| Stan Method                   | `StanEstimators` Function |
|-------------------------------|---------------------------|
| MCMC Sampling                 | `stan_sample`             |
| Maximum Likelihood Estimation | `stan_optimize`           |
| Variational Inference         | `stan_variational`        |
| Pathfinder                    | `stan_pathfinder`         |
| Laplace Approximation         | `stan_laplace`            |

### Motivations

While Stan is powerful, it can have a high barrier to entry for new
users - as they need to translate their existing models/functions into
the Stan language. `StanEstimators` provides a simple interface for
users to estimate their R functions using Stan’s algorithms without
needing to learn the Stan language. Additionally, it also allows for
users to ‘sanity-check’ their Stan code by comparing the results of
their Stan code to the results of their original R function. Finally, it
can be difficult to interface Stan with existing R packages and
functions - as this requires bespoke Stan models for the problem at
hand, something that may be too great a time investment for many users.

`StanEstimators` aims to address these issues.

### Estimating a Model

As an example, we will use the popular ‘Eight Schools’ example from the
Stan documentation. This example is used to demonstrate the use of
hierarchical models in Stan. The model is defined as:

\$\$ y_j \sim N(\theta_j, \sigma_j), \quad j=1,\ldots,8 \\ \theta_j \sim
N(\mu, \tau), \quad j=1,\ldots,8 \$\$

With data:

| School | Estimate ($y_{j}$) | Standard Error ($\sigma_{j}$) |
|--------|--------------------|-------------------------------|
| A      | 28                 | 15                            |
| B      | 8                  | 10                            |
| C      | -3                 | 16                            |
| D      | 7                  | 11                            |
| E      | -1                 | 9                             |
| F      | 1                  | 11                            |
| G      | 18                 | 10                            |
| H      | 12                 | 18                            |

``` r
y <- c(28,  8, -3,  7, -1,  1, 18, 12)
sigma <- c(15, 10, 16, 11,  9, 11, 10, 18)
```

#### Specifying the Function

To specify this as a function compatible with `StanEstimators`, we need
to define a function that takes in a vector of parameters as the first
argument and returns a single value (generally the unnormalized target
log density):

``` r
eight_schools_lpdf <- function(v, y, sigma) {
  mu <- v[1]
  tau <- v[2]
  eta <- v[3:10]

  # Use the non-centered parameterisation for eta
  # https://mc-stan.org/docs/stan-users-guide/reparameterization.html
  theta <- mu + tau * eta

  sum(dnorm(eta, mean = 0, sd = 1, log = TRUE)) +
    sum(dnorm(y, mean = theta, sd = sigma, log = TRUE))
}
```

Note that any additional data required by the function are passed as
additional arguments. In this case, we need to pass the data for $y$ and
$\sigma$. Alternatively, the function can assume that these data will be
available in the global environment, rather than passed as arguments.

#### Estimating the Function

To estimate our model, we simply pass the function to the relevant
`StanEstimators` function. For example, to estimate the function using
MCMC sampling, we use the `stan_sample` function (which uses the
No-U-Turn Sampler by default).

##### Parameter Bounds

Because we are estimating a standard deviation in our model ($\tau$), we
need to ensure that it is positive. We can do this by specifying a lower
bound of 0 for $\tau$. This is done by passing a vector of lower bounds
to the `lower` argument, with the corresponding elements of the vector
matching the order of the parameters in the function. Noting that $\tau$
is the second parameter in the function, and we do not want to specify a
lower bound for any other parameters, we can specify the lower bounds
as:

``` r
lower <- c(-Inf, 0, rep(-Inf, 8))
```

##### Running the Model

We can now pass these arguments to the `stan_sample` function to
estimate our model. We will use the default number of warmup iterations
(1000) and sampling iterations (1000).

Note that we need to specify the number of parameters in the model (10)
using the `n_pars` argument. This is because the function does not know
how many parameters are in the model, and so cannot automatically
determine this.

``` r
fit <- stan_sample(eight_schools_lpdf,
                   n_pars = 10,
                   additional_args = list(y = y, sigma = sigma),
                   lower = lower,
                   num_chains = 1)
```

#### Inspecting the Results

The estimates are stored in the `fit` object using the
[`posterior::draws`
format](https://CRAN.R-project.org/package=posterior/vignettes/posterior.html).
We can inspect the estimates using the `summary` function (which calls
[`posterior::summarise_draws`](https://mc-stan.org/posterior/reference/draws_summary.html)):

``` r
summary(fit)
#> # A tibble: 11 × 10
#>    variable     mean   median    sd   mad      q5    q95  rhat ess_bulk ess_tail
#>    <chr>       <dbl>    <dbl> <dbl> <dbl>   <dbl>  <dbl> <dbl>    <dbl>    <dbl>
#>  1 lp__     -3.96e+1 -3.94e+1 2.68  2.60  -44.4   -35.7  1.00      345.     665.
#>  2 pars[1]   7.40e+0  7.65e+0 5.10  4.75   -1.37   15.3  1.00      546.     331.
#>  3 pars[2]   6.80e+0  5.23e+0 6.29  4.61    0.546  17.7  1.00      476.     542.
#>  4 pars[3]   4.31e-1  4.77e-1 0.986 0.960  -1.14    2.01 1.000     808.     459.
#>  5 pars[4]   7.45e-2  1.15e-1 0.828 0.776  -1.34    1.45 1.00     1114.     814.
#>  6 pars[5]  -1.76e-1 -1.55e-1 0.922 0.871  -1.73    1.35 1.000    1308.     694.
#>  7 pars[6]  -2.60e-3 -1.11e-3 0.884 0.857  -1.43    1.47 0.999     878.     716.
#>  8 pars[7]  -3.16e-1 -3.34e-1 0.889 0.850  -1.76    1.14 0.999    1197.     739.
#>  9 pars[8]  -1.82e-1 -2.12e-1 0.894 0.837  -1.55    1.30 0.999    1023.     673.
#> 10 pars[9]   3.40e-1  3.49e-1 0.861 0.845  -1.01    1.78 1.00      967.     611.
#> 11 pars[10]  6.25e-2  6.76e-2 0.940 0.870  -1.50    1.68 1.01      873.     507.
```

### Model Checking and Comparison - Leave-One-Out Cross-Validation (LOO-CV)

`StanEstimators` also supports the use of the
[loo](https://mc-stan.org/loo/articles/loo2-example.html) package for
model checking and comparison. To use this, we need to specify a
function which returns the pointwise unnormalized target log density for
each observation in the data - as our original function returns the sum
over all observations.

For our model, we can define this function as:

``` r
eight_schools_pointwise <- function(v, y, sigma) {
  mu <- v[1]
  tau <- v[2]
  eta <- v[3:10]

  # Use the non-centered parameterisation for eta
  # https://mc-stan.org/docs/stan-users-guide/reparameterization.html
  theta <- mu + tau * eta

  # Only the density for the outcome variable
  dnorm(y, mean = theta, sd = sigma, log = TRUE)
}
```

This can then be used with the `loo` function to calculate the LOO-CV
estimate:

``` r
loo(fit, pointwise_ll_fun = eight_schools_pointwise,
    additional_args = list(y = y, sigma = sigma))
#> Warning: Some Pareto k diagnostic values are too high. See help('pareto-k-diagnostic') for details.
#> 
#> Computed from 1000 by 8 log-likelihood matrix.
#> 
#>          Estimate  SE
#> elpd_loo    -31.2 1.0
#> p_loo         1.6 0.4
#> looic        62.5 1.9
#> ------
#> MCSE of elpd_loo is NA.
#> MCSE and ESS estimates assume MCMC draws (r_eff in [0.6, 1.2]).
#> 
#> Pareto k diagnostic values:
#>                           Count Pct.    Min. ESS
#> (-Inf, 0.67]   (good)     6     75.0%   471     
#>    (0.67, 1]   (bad)      2     25.0%   <NA>    
#>     (1, Inf)   (very bad) 0      0.0%   <NA>    
#> See help('pareto-k-diagnostic') for details.
```
