
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

fit <- stan_sample(loglik_fun, inits, y,
                   lower = c(-Inf, 0), # Enforce a positivity constraint for SD
                   num_chains = 1,
                        seed = 1234)
```

We can see that the parameters were recovered accurately and that the
estimation was relatively fast: ~1 sec for 1000 warmup and 1000
iterations

``` r
unlist(fit@timing)
#>   warmup sampling 
#>    0.537    0.504
summary(fit)
#> # A tibble: 3 × 10
#>   variable    mean  median     sd    mad      q5     q95  rhat ess_bulk ess_tail
#>   <chr>      <dbl>   <dbl>  <dbl>  <dbl>   <dbl>   <dbl> <dbl>    <dbl>    <dbl>
#> 1 lp__     -1.05e3 -1.05e3 1.05   0.771  -1.05e3 -1.05e3 1.01      493.     595.
#> 2 pars[1]   1.00e1  1.00e1 0.0924 0.0901  9.87e0  1.02e1 1.01     1321.     693.
#> 3 pars[2]   1.99e0  1.99e0 0.0641 0.0679  1.89e0  2.10e0 0.999     841.     740.
```

Estimation time can be improved further by providing a gradient
function:

``` r
fit_grad <- stan_sample(loglik_fun, inits, y,
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
#>    0.078    0.082
summary(fit_grad)
#> # A tibble: 3 × 10
#>   variable    mean  median     sd    mad      q5     q95  rhat ess_bulk ess_tail
#>   <chr>      <dbl>   <dbl>  <dbl>  <dbl>   <dbl>   <dbl> <dbl>    <dbl>    <dbl>
#> 1 lp__     -1.05e3 -1.05e3 1.03   0.756  -1.05e3 -1.05e3 0.999     467.     487.
#> 2 pars[1]   1.00e1  1.00e1 0.0903 0.0959  9.88e0  1.02e1 1.00      935.     644.
#> 3 pars[2]   1.99e0  1.99e0 0.0649 0.0692  1.89e0  2.10e0 1.00      690.     538.
```

### Optimization

``` r
opt_fd <- stan_optimize(loglik_fun, inits, y,
                          lower = c(-Inf, 0),
                          seed = 1234)
opt_grad <- stan_optimize(loglik_fun, inits, y,
                          grad_fun = grad,
                          lower = c(-Inf, 0),
                          seed = 1234)
```

``` r
summary(opt_fd)
#>       lp__ pars[1] pars[2]
#> 1 -1051.19  10.028 1.98072
summary(opt_grad)
#>       lp__ pars[1] pars[2]
#> 1 -1051.19  10.028 1.98072
```

### Variational Inference

``` r
var_fd <- stan_variational(loglik_fun, inits, y,
                              lower = c(-Inf, 0),
                              seed = 1234)
var_grad <- stan_variational(loglik_fun, inits, y,
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
#> 2 log_p__  -1052.   -1052.    1.36   1.02   -1055.   -1051.      0.999    1002.
#> 3 log_g__     -1.03    -0.714 1.03   0.731     -3.29    -0.0486  1.00      959.
#> 4 pars[1]     10.1     10.1   0.0819 0.0847     9.99    10.3     1.00     1012.
#> 5 pars[2]      1.97     1.97  0.0614 0.0604     1.88     2.07    1.00      850.
#> # ℹ 1 more variable: ess_tail <dbl>
summary(var_grad)
#> # A tibble: 5 × 10
#>   variable     mean    median     sd    mad       q5        q95   rhat ess_bulk
#>   <chr>       <dbl>     <dbl>  <dbl>  <dbl>    <dbl>      <dbl>  <dbl>    <dbl>
#> 1 lp__         0        0     0      0          0        0      NA          NA 
#> 2 log_p__  -1052.   -1052.    1.36   1.02   -1055.   -1051.      0.999    1003.
#> 3 log_g__     -1.03    -0.714 1.03   0.731     -3.29    -0.0486  1.00      959.
#> 4 pars[1]     10.1     10.1   0.0819 0.0847     9.99    10.3     1.00     1012.
#> 5 pars[2]      1.97     1.97  0.0614 0.0604     1.88     2.07    1.00      850.
#> # ℹ 1 more variable: ess_tail <dbl>
```

### Pathfinder

``` r
path_fd <- stan_pathfinder(loglik_fun, inits, y,
                              lower = c(-Inf, 0),
                              seed = 1234)
path_grad <- stan_pathfinder(loglik_fun, inits, y,
                              grad_fun = grad,
                              lower = c(-Inf, 0),
                              seed = 1234)
```

``` r
summary(path_fd)
#> # A tibble: 4 × 10
#>   variable        mean   median     sd    mad        q5      q95  rhat ess_bulk
#>   <chr>          <dbl>    <dbl>  <dbl>  <dbl>     <dbl>    <dbl> <dbl>    <dbl>
#> 1 lp_approx__     3.07     3.44 1.06   0.704      0.838     4.04  1.00    1015.
#> 2 lp__        -1051.   -1051.   1.01   0.667  -1054.    -1051.    1.00    1022.
#> 3 pars[1]        10.0     10.0  0.0899 0.0864     9.87     10.2   1.00     899.
#> 4 pars[2]         1.98     1.98 0.0597 0.0625     1.89      2.08  1.00    1143.
#> # ℹ 1 more variable: ess_tail <dbl>
summary(path_grad)
#> # A tibble: 4 × 10
#>   variable        mean   median     sd    mad        q5      q95  rhat ess_bulk
#>   <chr>          <dbl>    <dbl>  <dbl>  <dbl>     <dbl>    <dbl> <dbl>    <dbl>
#> 1 lp_approx__     3.07     3.44 1.06   0.704      0.838     4.04  1.00    1016.
#> 2 lp__        -1051.   -1051.   1.01   0.667  -1054.    -1051.    1.00    1022.
#> 3 pars[1]        10.0     10.0  0.0899 0.0864     9.87     10.2   1.00     899.
#> 4 pars[2]         1.98     1.98 0.0597 0.0625     1.89      2.08  1.00    1143.
#> # ℹ 1 more variable: ess_tail <dbl>
```
