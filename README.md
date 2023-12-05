
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
#>    0.552    0.499
summary(fit)
#> # A tibble: 3 × 10
#>   variable    mean  median     sd    mad      q5     q95  rhat ess_bulk ess_tail
#>   <chr>      <dbl>   <dbl>  <dbl>  <dbl>   <dbl>   <dbl> <dbl>    <dbl>    <dbl>
#> 1 lp__     -1.03e3 -1.03e3 0.986  0.712  -1.04e3 -1.03e3  1.00     476.     562.
#> 2 pars[1]   1.01e1  1.01e1 0.0833 0.0850  9.92e0  1.02e1  1.00    1107.     700.
#> 3 pars[2]   1.92e0  1.92e0 0.0618 0.0593  1.82e0  2.02e0  1.00     885.     626.
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
#>    0.082    0.088
summary(fit_grad)
#> # A tibble: 3 × 10
#>   variable    mean  median     sd    mad      q5     q95  rhat ess_bulk ess_tail
#>   <chr>      <dbl>   <dbl>  <dbl>  <dbl>   <dbl>   <dbl> <dbl>    <dbl>    <dbl>
#> 1 lp__     -1.03e3 -1.03e3 0.988  0.712  -1.04e3 -1.03e3  1.00     438.     594.
#> 2 pars[1]   1.01e1  1.01e1 0.0837 0.0884  9.93e0  1.02e1  1.00     816.     768.
#> 3 pars[2]   1.92e0  1.92e0 0.0624 0.0618  1.82e0  2.03e0  1.00     875.     644.
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
#> 1 -1034.57 10.0684 1.91591
summary(opt_grad)
#>       lp__ pars[1] pars[2]
#> 1 -1034.57 10.0684 1.91591
```

### Laplace Approximation

``` r
# Can provide the mode as a numeric vector: 
lapl_num <- stan_laplace(loglik_fun, inits, y,
                          mode = c(10, 2),
                          lower = c(-Inf, 0),
                          seed = 1234)

# Can provide the mode as a StanOptimize object: 
lapl_opt <- stan_laplace(loglik_fun, inits, y,
                          mode = opt_fd,
                          lower = c(-Inf, 0),
                          seed = 1234)

# Can estimate the mode before sampling: 
lapl_est <- stan_laplace(loglik_fun, inits, y,
                          lower = c(-Inf, 0),
                          seed = 1234)
```

``` r
summary(lapl_num)
#> # A tibble: 4 × 10
#>   variable     mean    median     sd    mad       q5        q95  rhat ess_bulk
#>   <chr>       <dbl>     <dbl>  <dbl>  <dbl>    <dbl>      <dbl> <dbl>    <dbl>
#> 1 log_p__  -1036.   -1036.    1.84   1.56   -1040.   -1034.     1.00      985.
#> 2 log_q__     -1.04    -0.692 1.04   0.716     -3.21    -0.0582 0.999    1047.
#> 3 pars[1]     10.0     10.0   0.0897 0.0863     9.85    10.1    1.00      931.
#> 4 pars[2]      2.00     2.00  0.0685 0.0694     1.89     2.12   1.00     1051.
#> # ℹ 1 more variable: ess_tail <dbl>
summary(lapl_opt)
#> # A tibble: 4 × 10
#>   variable     mean    median     sd    mad       q5        q95  rhat ess_bulk
#>   <chr>       <dbl>     <dbl>  <dbl>  <dbl>    <dbl>      <dbl> <dbl>    <dbl>
#> 1 log_p__  -1035.   -1035.    1.06   0.712  -1037.   -1034.     0.999    1043.
#> 2 log_q__     -1.04    -0.692 1.04   0.716     -3.21    -0.0582 0.999    1047.
#> 3 pars[1]     10.1     10.1   0.0859 0.0818     9.92    10.2    1.00      932.
#> 4 pars[2]      1.92     1.92  0.0628 0.0636     1.82     2.02   1.00     1051.
#> # ℹ 1 more variable: ess_tail <dbl>
summary(lapl_est)
#> # A tibble: 4 × 10
#>   variable     mean    median     sd    mad       q5        q95  rhat ess_bulk
#>   <chr>       <dbl>     <dbl>  <dbl>  <dbl>    <dbl>      <dbl> <dbl>    <dbl>
#> 1 log_p__  -1035.   -1035.    1.06   0.712  -1037.   -1034.     0.999    1043.
#> 2 log_q__     -1.04    -0.692 1.04   0.716     -3.21    -0.0582 0.999    1047.
#> 3 pars[1]     10.1     10.1   0.0859 0.0818     9.92    10.2    1.00      932.
#> 4 pars[2]      1.92     1.92  0.0628 0.0636     1.82     2.02   1.00     1051.
#> # ℹ 1 more variable: ess_tail <dbl>
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
#> 2 log_p__  -1035.   -1035.    1.37   1.04   -1038.   -1034.      0.999    1002.
#> 3 log_g__     -1.03    -0.714 1.03   0.731     -3.29    -0.0486  1.00      959.
#> 4 pars[1]     10.2     10.2   0.0792 0.0818    10.0     10.3     1.00     1012.
#> 5 pars[2]      1.91     1.91  0.0594 0.0584     1.81     2.01    1.00      850.
#> # ℹ 1 more variable: ess_tail <dbl>
summary(var_grad)
#> # A tibble: 5 × 10
#>   variable     mean    median     sd    mad       q5        q95   rhat ess_bulk
#>   <chr>       <dbl>     <dbl>  <dbl>  <dbl>    <dbl>      <dbl>  <dbl>    <dbl>
#> 1 lp__         0        0     0      0          0        0      NA          NA 
#> 2 log_p__  -1035.   -1035.    1.37   1.04   -1038.   -1034.      0.999    1002.
#> 3 log_g__     -1.03    -0.714 1.03   0.731     -3.29    -0.0486  1.00      959.
#> 4 pars[1]     10.2     10.2   0.0792 0.0818    10.0     10.3     1.00     1012.
#> 5 pars[2]      1.91     1.91  0.0594 0.0584     1.81     2.01    1.00      850.
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
#> 1 lp_approx__     3.12     3.49 1.06   0.689      0.901     4.10  1.00    1014.
#> 2 lp__        -1035.   -1035.   0.996  0.638  -1037.    -1034.    1.00    1028.
#> 3 pars[1]        10.1     10.1  0.0840 0.0815     9.93     10.2   1.00     895.
#> 4 pars[2]         1.92     1.92 0.0590 0.0605     1.82      2.01  1.00    1145.
#> # ℹ 1 more variable: ess_tail <dbl>
summary(path_grad)
#> # A tibble: 4 × 10
#>   variable        mean   median     sd    mad        q5      q95  rhat ess_bulk
#>   <chr>          <dbl>    <dbl>  <dbl>  <dbl>     <dbl>    <dbl> <dbl>    <dbl>
#> 1 lp_approx__     3.12     3.49 1.06   0.689      0.901     4.10  1.00    1014.
#> 2 lp__        -1035.   -1035.   0.996  0.638  -1037.    -1034.    1.00    1028.
#> 3 pars[1]        10.1     10.1  0.0840 0.0815     9.93     10.2   1.00     895.
#> 4 pars[2]         1.92     1.92 0.0590 0.0605     1.82      2.01  1.00    1145.
#> # ℹ 1 more variable: ess_tail <dbl>
```
