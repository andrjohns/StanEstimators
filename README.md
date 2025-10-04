
<!-- README.md is generated from README.Rmd. Please edit that file -->

# StanEstimators

<!-- badges: start -->

[![CRAN
status](https://www.r-pkg.org/badges/version/StanEstimators)](https://CRAN.R-project.org/package=StanEstimators)
[![R-CMD-check](https://github.com/andrjohns/StanEstimators/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/andrjohns/StanEstimators/actions/workflows/R-CMD-check.yaml)
[![StanEstimators status
badge](https://andrjohns.r-universe.dev/badges/StanEstimators)](https://andrjohns.r-universe.dev/StanEstimators)
<!-- badges: end -->

The `StanEstimators` package provides an estimation back-end for R
functions, similar to those provided by the `optim` package, using the
algorithms provided by the Stan probabilistic programming language.

As Stan’s algorithms are gradient-based, function gradients can be
automatically calculated using finite-differencing or the user can
provide a function for analytical calculation.

## Installation

You can install pre-built binaries using:

``` r
# we recommend running this is a fresh R session or restarting your current session
install.packages('StanEstimators', repos = c('https://andrjohns.r-universe.dev', 'https://cloud.r-project.org'))
```

Or you can build from source using:

``` r
# install.packages("remotes")
remotes::install_github("andrjohns/StanEstimators")
```

## Usage

Consider the goal of estimating the mean and standard deviation of a
normal distribution, with uniform uninformative priors on both
parameters:

$$
y \sim \textbf{N}(\mu, \sigma)
$$

$$
\mu \sim \textbf{U}[-\infty, \infty]
$$

$$
\sigma \sim \textbf{U}[0, \infty]
$$

With known true values for verification:

``` r
y <- rnorm(500, 10, 2)
```

As with other estimation routines provided in R, we need to specify this
as a function which takes a vector of parameters as its first argument
and returns a single scalar value (the unnormalized target log density),
as well as initial values for the parameters:

``` r
loglik_fun <- function(v, x) {
  sum(dnorm(x, v[1], v[2], log = TRUE))
}

inits <- c(0, 5)
```

Estimation time can also be significantly reduced by providing a
gradient function, rather than relying on finite-differencing:

``` r
grad <- function(v, x) {
  inv_sigma <- 1 / v[2]
  y_scaled = (x - v[1]) * inv_sigma
  scaled_diff = inv_sigma * y_scaled
  c(sum(scaled_diff),
    sum(inv_sigma * (y_scaled*y_scaled) - inv_sigma)
  )
}
```

### MCMC Estimation

Full MCMC estimation is provided by the `stan_sample()` function, which
uses Stan’s default No U-Turn Sampler (NUTS) unless otherwise specified:

``` r
library(StanEstimators)

fit <- stan_sample(loglik_fun, inits, additional_args = list(y),
                   lower = c(-Inf, 0), # Enforce a positivity constraint for SD
                   num_chains = 1, seed = 1234)
```

We can see that the parameters were recovered accurately and that the
estimation was relatively fast: ~1 sec for 1000 warmup and 1000
iterations

``` r
unlist(fit@timing)
#>   warmup sampling 
#>    0.720    0.707
summary(fit)
#> # A tibble: 3 × 10
#>   variable    mean  median     sd    mad      q5     q95  rhat ess_bulk ess_tail
#>   <chr>      <dbl>   <dbl>  <dbl>  <dbl>   <dbl>   <dbl> <dbl>    <dbl>    <dbl>
#> 1 lp__     -1.05e3 -1.05e3 0.973  0.788  -1.05e3 -1.05e3  1.01     521.     720.
#> 2 pars[1]   9.96e0  9.97e0 0.0912 0.0911  9.81e0  1.01e1  1.00     943.     712.
#> 3 pars[2]   1.97e0  1.96e0 0.0637 0.0674  1.87e0  2.08e0  1.00     878.     615.
```

Estimation time can be improved further by providing a gradient
function:

``` r
fit_grad <- stan_sample(loglik_fun, inits, additional_args = list(y),
                        grad_fun = grad,
                        lower = c(-Inf, 0),
                        num_chains = 1,
                        seed = 1234)
```

Which shows that the estimation time was dramatically improved, now
~0.15 seconds for 1000 warmup and 1000 iterations.

``` r
unlist(fit_grad@timing)
#>   warmup sampling 
#>    0.103    0.093
summary(fit_grad)
#> # A tibble: 3 × 10
#>   variable    mean  median     sd    mad      q5     q95  rhat ess_bulk ess_tail
#>   <chr>      <dbl>   <dbl>  <dbl>  <dbl>   <dbl>   <dbl> <dbl>    <dbl>    <dbl>
#> 1 lp__     -1.05e3 -1.05e3 0.952  0.763  -1.05e3 -1.05e3 1.01      500.     675.
#> 2 pars[1]   9.97e0  9.97e0 0.0905 0.0954  9.82e0  1.01e1 1.000     830.     531.
#> 3 pars[2]   1.96e0  1.96e0 0.0619 0.0616  1.87e0  2.07e0 1.00     1047.     640.
```

### Optimization

``` r
opt_fd <- stan_optimize(loglik_fun, inits, additional_args = list(y),
                          lower = c(-Inf, 0),
                          seed = 1234)
opt_grad <- stan_optimize(loglik_fun, inits, additional_args = list(y),
                          grad_fun = grad,
                          lower = c(-Inf, 0),
                          seed = 1234)
```

``` r
summary(opt_fd)
#>        lp__ pars[1] pars[2]
#> 1 -1046.049  9.9691 1.96036
summary(opt_grad)
#>        lp__ pars[1] pars[2]
#> 1 -1046.049  9.9691 1.96036
```

### Laplace Approximation

``` r
# Can provide the mode as a numeric vector:
lapl_num <- stan_laplace(loglik_fun, inits, additional_args = list(y),
                          mode = c(10, 2),
                          lower = c(-Inf, 0),
                          seed = 1234)

# Can provide the mode as a StanOptimize object:
lapl_opt <- stan_laplace(loglik_fun, inits, additional_args = list(y),
                          mode = opt_fd,
                          lower = c(-Inf, 0),
                          seed = 1234)

# Can estimate the mode before sampling:
lapl_est <- stan_laplace(loglik_fun, inits, additional_args = list(y),
                          lower = c(-Inf, 0),
                          seed = 1234)
```

``` r
summary(lapl_num)
#> # A tibble: 4 × 10
#>   variable     mean    median     sd    mad       q5        q95  rhat ess_bulk
#>   <chr>       <dbl>     <dbl>  <dbl>  <dbl>    <dbl>      <dbl> <dbl>    <dbl>
#> 1 log_p__  -1477.   -1475.    55.3   56.0   -1572.   -1389.     1.00      986.
#> 2 log_q__     -1.01    -0.695  1.01   0.743    -3.03    -0.0443 1.00      913.
#> 3 pars[1]     10.0     10.00   0.335  0.343     9.47    10.5    0.999     831.
#> 4 pars[2]      7.45     7.39   0.897  0.893     6.10     9.08   1.00      987.
#> # ℹ 1 more variable: ess_tail <dbl>
summary(lapl_opt)
#> # A tibble: 4 × 10
#>   variable     mean    median     sd    mad       q5        q95  rhat ess_bulk
#>   <chr>       <dbl>     <dbl>  <dbl>  <dbl>    <dbl>      <dbl> <dbl>    <dbl>
#> 1 log_p__  -1458.   -1457.    52.8   53.5   -1549.   -1374.     1.00      986.
#> 2 log_q__     -1.01    -0.695  1.01   0.743    -3.03    -0.0443 1.00      913.
#> 3 pars[1]      9.97     9.97   0.321  0.329     9.46    10.5    0.999     830.
#> 4 pars[2]      7.16     7.10   0.827  0.824     5.91     8.66   1.00      987.
#> # ℹ 1 more variable: ess_tail <dbl>
summary(lapl_est)
#> # A tibble: 4 × 10
#>   variable     mean    median     sd    mad       q5        q95  rhat ess_bulk
#>   <chr>       <dbl>     <dbl>  <dbl>  <dbl>    <dbl>      <dbl> <dbl>    <dbl>
#> 1 log_p__  -1458.   -1457.    52.8   53.5   -1549.   -1374.     1.00      986.
#> 2 log_q__     -1.01    -0.695  1.01   0.743    -3.03    -0.0443 1.00      913.
#> 3 pars[1]      9.97     9.97   0.321  0.329     9.46    10.5    0.999     830.
#> 4 pars[2]      7.16     7.10   0.827  0.824     5.91     8.66   1.00      987.
#> # ℹ 1 more variable: ess_tail <dbl>
```

### Variational Inference

``` r
var_fd <- stan_variational(loglik_fun, inits, additional_args = list(y),
                              lower = c(-Inf, 0),
                              seed = 1234)
var_grad <- stan_variational(loglik_fun, inits, additional_args = list(y),
                              grad_fun = grad,
                              lower = c(-Inf, 0),
                              seed = 1234)
```

``` r
summary(var_fd)
#> # A tibble: 5 × 10
#>   variable      mean    median     sd    mad       q5        q95   rhat ess_bulk
#>   <chr>        <dbl>     <dbl>  <dbl>  <dbl>    <dbl>      <dbl>  <dbl>    <dbl>
#> 1 lp__         0         0     0      0          0        0      NA          NA 
#> 2 log_p__  -1047.    -1046.    1.25   0.975  -1049.   -1045.      1.00     1017.
#> 3 log_g__     -0.978    -0.660 0.966  0.678     -2.84    -0.0566  1.00     1054.
#> 4 pars[1]     10.0      10.0   0.0847 0.0877     9.88    10.2     0.999    1025.
#> 5 pars[2]      1.92      1.92  0.0528 0.0523     1.83     2.01    1.00     1047.
#> # ℹ 1 more variable: ess_tail <dbl>
summary(var_grad)
#> # A tibble: 5 × 10
#>   variable      mean    median     sd    mad       q5        q95   rhat ess_bulk
#>   <chr>        <dbl>     <dbl>  <dbl>  <dbl>    <dbl>      <dbl>  <dbl>    <dbl>
#> 1 lp__         0         0     0      0          0        0      NA          NA 
#> 2 log_p__  -1047.    -1046.    1.25   0.975  -1049.   -1045.      1.00     1017.
#> 3 log_g__     -0.978    -0.660 0.966  0.678     -2.84    -0.0566  1.00     1054.
#> 4 pars[1]     10.0      10.0   0.0847 0.0877     9.88    10.2     0.999    1025.
#> 5 pars[2]      1.92      1.92  0.0528 0.0523     1.83     2.01    1.00     1047.
#> # ℹ 1 more variable: ess_tail <dbl>
```

### Pathfinder

``` r
path_fd <- stan_pathfinder(loglik_fun, inits, additional_args = list(y),
                              lower = c(-Inf, 0),
                              seed = 1234)
path_grad <- stan_pathfinder(loglik_fun, inits, additional_args = list(y),
                              grad_fun = grad,
                              lower = c(-Inf, 0),
                              seed = 1234)
```

``` r
summary(path_fd)
#> # A tibble: 5 × 10
#>   variable        mean   median     sd    mad        q5      q95  rhat ess_bulk
#>   <chr>          <dbl>    <dbl>  <dbl>  <dbl>     <dbl>    <dbl> <dbl>    <dbl>
#> 1 lp_approx__     3.04     3.45 1.19   0.704      0.609     4.07 1.00    652.  
#> 2 lp__        -1046.   -1046.   1.09   0.661  -1049.    -1045.   1.00    653.  
#> 3 path__          2.51     3    1.10   1.48       1         4    2.65      1.20
#> 4 pars[1]         9.97     9.96 0.0872 0.0835     9.82     10.1  1.000   803.  
#> 5 pars[2]         1.96     1.96 0.0633 0.0606     1.86      2.07 1.00    734.  
#> # ℹ 1 more variable: ess_tail <dbl>
summary(path_grad)
#> # A tibble: 5 × 10
#>   variable        mean   median     sd    mad        q5      q95  rhat ess_bulk
#>   <chr>          <dbl>    <dbl>  <dbl>  <dbl>     <dbl>    <dbl> <dbl>    <dbl>
#> 1 lp_approx__     3.04     3.45 1.19   0.704      0.609     4.07 1.00    652.  
#> 2 lp__        -1046.   -1046.   1.09   0.661  -1049.    -1045.   1.00    653.  
#> 3 path__          2.51     3    1.10   1.48       1         4    2.65      1.20
#> 4 pars[1]         9.97     9.96 0.0872 0.0835     9.82     10.1  1.000   803.  
#> 5 pars[2]         1.96     1.96 0.0633 0.0606     1.86      2.07 1.00    734.  
#> # ℹ 1 more variable: ess_tail <dbl>
```
