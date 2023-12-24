
<!-- README.md is generated from README.Rmd. Please edit that file -->

# StanEstimators

<!-- badges: start -->

[![R-CMD-check](https://github.com/andrjohns/StanEstimators/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/andrjohns/StanEstimators/actions/workflows/R-CMD-check.yaml)
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
install.packages("StanEstimators", 
                 repos = c("https://andrjohns.github.io/StanEstimators/", getOption("repos")))
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
#>    0.855    0.725
summary(fit)
#> # A tibble: 3 × 10
#>   variable    mean  median     sd    mad      q5     q95  rhat ess_bulk ess_tail
#>   <chr>      <dbl>   <dbl>  <dbl>  <dbl>   <dbl>   <dbl> <dbl>    <dbl>    <dbl>
#> 1 lp__     -1.06e3 -1.06e3 1.08   0.801  -1.06e3 -1.06e3 1.00      442.     420.
#> 2 pars[1]   9.97e0  9.97e0 0.0956 0.101   9.82e0  1.01e1 0.999    1034.     765.
#> 3 pars[2]   2.02e0  2.02e0 0.0670 0.0661  1.92e0  2.13e0 1.00     1026.     521.
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
#>    0.130    0.117
summary(fit_grad)
#> # A tibble: 3 × 10
#>   variable    mean  median     sd    mad      q5     q95  rhat ess_bulk ess_tail
#>   <chr>      <dbl>   <dbl>  <dbl>  <dbl>   <dbl>   <dbl> <dbl>    <dbl>    <dbl>
#> 1 lp__     -1.06e3 -1.06e3 1.06   0.756  -1.06e3 -1.06e3 0.999     486.     667.
#> 2 pars[1]   9.97e0  9.97e0 0.0917 0.0828  9.81e0  1.01e1 1.00     1149.     758.
#> 3 pars[2]   2.02e0  2.02e0 0.0662 0.0682  1.92e0  2.14e0 1.00     1070.     663.
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
#> 1 -1059.86 9.96874 2.01536
summary(opt_grad)
#>       lp__ pars[1] pars[2]
#> 1 -1059.86 9.96874 2.01536
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
#> 1 log_p__  -1060.   -1060.    1.16   0.801  -1063.   -1059.     0.999    1050.
#> 2 log_q__     -1.04    -0.692 1.04   0.716     -3.21    -0.0582 0.999    1047.
#> 3 pars[1]     10.0     10.0   0.0897 0.0850     9.85    10.1    1.00      931.
#> 4 pars[2]      2.00     2.00  0.0651 0.0660     1.90     2.11   1.00     1051.
#> # ℹ 1 more variable: ess_tail <dbl>
summary(lapl_opt)
#> # A tibble: 4 × 10
#>   variable     mean    median     sd    mad       q5        q95  rhat ess_bulk
#>   <chr>       <dbl>     <dbl>  <dbl>  <dbl>    <dbl>      <dbl> <dbl>    <dbl>
#> 1 log_p__  -1060.   -1060.    1.06   0.712  -1062.   -1059.     0.999    1048.
#> 2 log_q__     -1.04    -0.692 1.04   0.716     -3.21    -0.0582 0.999    1047.
#> 3 pars[1]      9.97     9.97  0.0903 0.0862     9.82    10.1    1.00      932.
#> 4 pars[2]      2.02     2.02  0.0661 0.0670     1.91     2.13   1.00     1051.
#> # ℹ 1 more variable: ess_tail <dbl>
summary(lapl_est)
#> # A tibble: 4 × 10
#>   variable     mean    median     sd    mad       q5        q95  rhat ess_bulk
#>   <chr>       <dbl>     <dbl>  <dbl>  <dbl>    <dbl>      <dbl> <dbl>    <dbl>
#> 1 log_p__  -1060.   -1060.    1.06   0.712  -1062.   -1059.     0.999    1048.
#> 2 log_q__     -1.04    -0.692 1.04   0.716     -3.21    -0.0582 0.999    1047.
#> 3 pars[1]      9.97     9.97  0.0903 0.0862     9.82    10.1    1.00      932.
#> 4 pars[2]      2.02     2.02  0.0661 0.0670     1.91     2.13   1.00     1051.
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
#> 2 log_p__  -1061.    -1061.    1.67   1.33   -1064.   -1059.      1.00      996.
#> 3 log_g__     -0.966    -0.697 0.963  0.729     -3.03    -0.0399  1.00     1094.
#> 4 pars[1]      9.94      9.94  0.0813 0.0830     9.80    10.1     0.999    1104.
#> 5 pars[2]      2.11      2.11  0.0710 0.0692     1.99     2.22    1.00      944.
#> # ℹ 1 more variable: ess_tail <dbl>
summary(var_grad)
#> # A tibble: 5 × 10
#>   variable     mean    median     sd    mad       q5        q95   rhat ess_bulk
#>   <chr>       <dbl>     <dbl>  <dbl>  <dbl>    <dbl>      <dbl>  <dbl>    <dbl>
#> 1 lp__         0        0     0      0          0        0      NA          NA 
#> 2 log_p__  -1061.   -1060.    1.35   1.01   -1063.   -1059.      0.999    1003.
#> 3 log_g__     -1.03    -0.714 1.03   0.731     -3.29    -0.0486  1.00      959.
#> 4 pars[1]     10.1     10.1   0.0834 0.0862     9.93    10.2     1.00     1012.
#> 5 pars[2]      2.01     2.01  0.0625 0.0614     1.91     2.11    1.00      850.
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
#> 1 lp_appr…  3.03e0  3.33e0 0.965  0.702   1.15e0  3.96e0 1.00      977.    1018.
#> 2 lp__     -1.06e3 -1.06e3 0.970  0.697  -1.06e3 -1.06e3 0.999     991.    1018.
#> 3 pars[1]   9.97e0  9.97e0 0.0886 0.0850  9.82e0  1.01e1 1.00     1047.     824.
#> 4 pars[2]   2.02e0  2.02e0 0.0648 0.0688  1.91e0  2.13e0 0.999     795.     793.
summary(path_grad)
#> # A tibble: 4 × 10
#>   variable    mean  median     sd    mad      q5     q95  rhat ess_bulk ess_tail
#>   <chr>      <dbl>   <dbl>  <dbl>  <dbl>   <dbl>   <dbl> <dbl>    <dbl>    <dbl>
#> 1 lp_appr…  3.03e0  3.33e0 0.965  0.702   1.15e0  3.96e0 1.00      977.    1018.
#> 2 lp__     -1.06e3 -1.06e3 0.970  0.697  -1.06e3 -1.06e3 0.999     991.    1018.
#> 3 pars[1]   9.97e0  9.97e0 0.0886 0.0850  9.82e0  1.01e1 1.00     1047.     824.
#> 4 pars[2]   2.02e0  2.02e0 0.0648 0.0688  1.91e0  2.13e0 0.999     795.     793.
```
