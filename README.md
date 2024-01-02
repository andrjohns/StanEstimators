
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
and returns a single scalar value (the log-likelihood), as well as
initial values for the parameters:

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
#>    0.527    0.490
summary(fit)
#> # A tibble: 3 × 10
#>   variable    mean  median     sd    mad      q5     q95  rhat ess_bulk ess_tail
#>   <chr>      <dbl>   <dbl>  <dbl>  <dbl>   <dbl>   <dbl> <dbl>    <dbl>    <dbl>
#> 1 lp__     -1.08e3 -1.08e3 1.03   0.749  -1.08e3 -1.08e3  1.00     507.     672.
#> 2 pars[1]   1.01e1  1.01e1 0.0940 0.0948  9.97e0  1.03e1  1.00     895.     671.
#> 3 pars[2]   2.11e0  2.10e0 0.0686 0.0670  2.00e0  2.22e0  1.00     860.     696.
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
#>    0.111    0.087
summary(fit_grad)
#> # A tibble: 3 × 10
#>   variable    mean  median     sd    mad      q5     q95  rhat ess_bulk ess_tail
#>   <chr>      <dbl>   <dbl>  <dbl>  <dbl>   <dbl>   <dbl> <dbl>    <dbl>    <dbl>
#> 1 lp__     -1.08e3 -1.08e3 1.02   0.741  -1.08e3 -1.08e3  1.00     572.     712.
#> 2 pars[1]   1.01e1  1.01e1 0.0928 0.0943  9.97e0  1.03e1  1.00     950.     623.
#> 3 pars[2]   2.10e0  2.10e0 0.0691 0.0696  1.99e0  2.22e0  1.00     725.     613.
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
#>       lp__ pars[1] pars[2]
#> 1 -1079.84 10.1221 2.09743
summary(opt_grad)
#>       lp__ pars[1] pars[2]
#> 1 -1079.84 10.1221 2.09743
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
#> 1 log_p__  -1082.   -1082.    2.39   2.18   -1087.   -1080.     0.999     989.
#> 2 log_q__     -1.04    -0.692 1.04   0.716     -3.21    -0.0582 0.999    1047.
#> 3 pars[1]     10.0     10.0   0.0899 0.0867     9.85    10.1    1.00      933.
#> 4 pars[2]      2.00     2.00  0.0626 0.0635     1.90     2.11   1.00     1051.
#> # ℹ 1 more variable: ess_tail <dbl>
summary(lapl_opt)
#> # A tibble: 4 × 10
#>   variable     mean    median     sd    mad       q5        q95  rhat ess_bulk
#>   <chr>       <dbl>     <dbl>  <dbl>  <dbl>    <dbl>      <dbl> <dbl>    <dbl>
#> 1 log_p__  -1080.   -1080.    1.06   0.712  -1082.   -1079.     0.999    1044.
#> 2 log_q__     -1.04    -0.692 1.04   0.716     -3.21    -0.0582 0.999    1047.
#> 3 pars[1]     10.1     10.1   0.0940 0.0897     9.96    10.3    1.00      932.
#> 4 pars[2]      2.10     2.10  0.0688 0.0697     1.99     2.21   1.00     1051.
#> # ℹ 1 more variable: ess_tail <dbl>
summary(lapl_est)
#> # A tibble: 4 × 10
#>   variable     mean    median     sd    mad       q5        q95  rhat ess_bulk
#>   <chr>       <dbl>     <dbl>  <dbl>  <dbl>    <dbl>      <dbl> <dbl>    <dbl>
#> 1 log_p__  -1080.   -1080.    1.06   0.712  -1082.   -1079.     0.999    1044.
#> 2 log_q__     -1.04    -0.692 1.04   0.716     -3.21    -0.0582 0.999    1047.
#> 3 pars[1]     10.1     10.1   0.0940 0.0897     9.96    10.3    1.00      932.
#> 4 pars[2]      2.10     2.10  0.0688 0.0697     1.99     2.21   1.00     1051.
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
#>   variable     mean    median     sd    mad       q5        q95   rhat ess_bulk
#>   <chr>       <dbl>     <dbl>  <dbl>  <dbl>    <dbl>      <dbl>  <dbl>    <dbl>
#> 1 lp__         0        0     0      0          0        0      NA          NA 
#> 2 log_p__  -1081.   -1080.    1.33   0.986  -1083.   -1079.      0.999     997.
#> 3 log_g__     -1.03    -0.714 1.03   0.731     -3.29    -0.0486  1.00      959.
#> 4 pars[1]     10.2     10.2   0.0869 0.0898    10.1     10.4     1.00     1012.
#> 5 pars[2]      2.09     2.09  0.0650 0.0639     1.99     2.20    1.00      850.
#> # ℹ 1 more variable: ess_tail <dbl>
summary(var_grad)
#> # A tibble: 5 × 10
#>   variable     mean    median     sd    mad       q5        q95   rhat ess_bulk
#>   <chr>       <dbl>     <dbl>  <dbl>  <dbl>    <dbl>      <dbl>  <dbl>    <dbl>
#> 1 lp__         0        0     0      0          0        0      NA          NA 
#> 2 log_p__  -1081.   -1080.    1.33   0.986  -1083.   -1079.      0.999     997.
#> 3 log_g__     -1.03    -0.714 1.03   0.731     -3.29    -0.0486  1.00      959.
#> 4 pars[1]     10.2     10.2   0.0869 0.0898    10.1     10.4     1.00     1012.
#> 5 pars[2]      2.09     2.09  0.0650 0.0639     1.99     2.20    1.00      850.
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
#> # A tibble: 4 × 10
#>   variable    mean  median     sd    mad      q5     q95  rhat ess_bulk ess_tail
#>   <chr>      <dbl>   <dbl>  <dbl>  <dbl>   <dbl>   <dbl> <dbl>    <dbl>    <dbl>
#> 1 lp_appr…  2.96e0  3.27e0 0.998  0.716   1.03e0  3.90e0 1.00      949.     909.
#> 2 lp__     -1.08e3 -1.08e3 1.04   0.726  -1.08e3 -1.08e3 1.00      946.     820.
#> 3 pars[1]   1.01e1  1.01e1 0.0955 0.0920  9.96e0  1.03e1 0.999    1004.     800.
#> 4 pars[2]   2.10e0  2.11e0 0.0668 0.0695  1.99e0  2.21e0 1.00      998.     907.
summary(path_grad)
#> # A tibble: 4 × 10
#>   variable    mean  median     sd    mad      q5     q95  rhat ess_bulk ess_tail
#>   <chr>      <dbl>   <dbl>  <dbl>  <dbl>   <dbl>   <dbl> <dbl>    <dbl>    <dbl>
#> 1 lp_appr…  2.96e0  3.27e0 0.998  0.716   1.03e0  3.90e0 1.00      949.     909.
#> 2 lp__     -1.08e3 -1.08e3 1.04   0.726  -1.08e3 -1.08e3 1.00      946.     820.
#> 3 pars[1]   1.01e1  1.01e1 0.0955 0.0920  9.96e0  1.03e1 0.999    1004.     800.
#> 4 pars[2]   2.10e0  2.11e0 0.0668 0.0695  1.99e0  2.21e0 1.00      998.     907.
```
