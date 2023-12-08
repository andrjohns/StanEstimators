
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

fit <- stan_sample(loglik_fun, inits, additional_args = list(y),
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
#>    0.567    0.534
summary(fit)
#> # A tibble: 3 × 10
#>   variable    mean  median     sd    mad      q5     q95  rhat ess_bulk ess_tail
#>   <chr>      <dbl>   <dbl>  <dbl>  <dbl>   <dbl>   <dbl> <dbl>    <dbl>    <dbl>
#> 1 lp__     -1.06e3 -1.06e3 0.993  0.712  -1.07e3 -1.06e3 1.01      511.     587.
#> 2 pars[1]   9.91e0  9.91e0 0.0860 0.0829  9.76e0  1.01e1 0.999    1041.     717.
#> 3 pars[2]   2.04e0  2.04e0 0.0678 0.0690  1.93e0  2.15e0 1.01      940.     558.
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
#>    0.082    0.089
summary(fit_grad)
#> # A tibble: 3 × 10
#>   variable    mean  median     sd    mad      q5     q95  rhat ess_bulk ess_tail
#>   <chr>      <dbl>   <dbl>  <dbl>  <dbl>   <dbl>   <dbl> <dbl>    <dbl>    <dbl>
#> 1 lp__     -1.06e3 -1.06e3 0.992  0.704  -1.07e3 -1.06e3  1.00     510.     699.
#> 2 pars[1]   9.90e0  9.90e0 0.0913 0.0930  9.75e0  1.00e1  1.00     821.     743.
#> 3 pars[2]   2.04e0  2.04e0 0.0652 0.0668  1.94e0  2.15e0  1.00     757.     691.
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
#> 1 -1063.79 9.90594 2.03121
summary(opt_grad)
#>       lp__ pars[1] pars[2]
#> 1 -1063.79 9.90594 2.03121
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
#> 1 log_p__  -1065.   -1064.    1.58   1.26   -1068.   -1063.     1.00      996.
#> 2 log_q__     -1.04    -0.692 1.04   0.716     -3.21    -0.0582 0.999    1047.
#> 3 pars[1]     10.0     10.0   0.0899 0.0848     9.85    10.1    1.00      930.
#> 4 pars[2]      2.00     2.00  0.0646 0.0655     1.90     2.11   1.00     1051.
#> # ℹ 1 more variable: ess_tail <dbl>
summary(lapl_opt)
#> # A tibble: 4 × 10
#>   variable     mean    median     sd    mad       q5        q95  rhat ess_bulk
#>   <chr>       <dbl>     <dbl>  <dbl>  <dbl>    <dbl>      <dbl> <dbl>    <dbl>
#> 1 log_p__  -1064.   -1064.    1.06   0.712  -1066.   -1063.     0.999    1044.
#> 2 log_q__     -1.04    -0.692 1.04   0.716     -3.21    -0.0582 0.999    1047.
#> 3 pars[1]      9.91     9.91  0.0910 0.0868     9.75    10.1    1.00      932.
#> 4 pars[2]      2.03     2.03  0.0666 0.0675     1.93     2.14   1.00     1051.
#> # ℹ 1 more variable: ess_tail <dbl>
summary(lapl_est)
#> # A tibble: 4 × 10
#>   variable     mean    median     sd    mad       q5        q95  rhat ess_bulk
#>   <chr>       <dbl>     <dbl>  <dbl>  <dbl>    <dbl>      <dbl> <dbl>    <dbl>
#> 1 log_p__  -1064.   -1064.    1.06   0.712  -1066.   -1063.     0.999    1044.
#> 2 log_q__     -1.04    -0.692 1.04   0.716     -3.21    -0.0582 0.999    1047.
#> 3 pars[1]      9.91     9.91  0.0910 0.0868     9.75    10.1    1.00      932.
#> 4 pars[2]      2.03     2.03  0.0666 0.0675     1.93     2.14   1.00     1051.
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
#> 2 log_p__  -1065.   -1064.    1.34   1.01   -1067.   -1063.      0.999    1000.
#> 3 log_g__     -1.03    -0.714 1.03   0.731     -3.29    -0.0486  1.00      959.
#> 4 pars[1]     10.0      9.99  0.0841 0.0869     9.86    10.1     1.00     1012.
#> 5 pars[2]      2.02     2.02  0.0630 0.0619     1.92     2.13    1.00      850.
#> # ℹ 1 more variable: ess_tail <dbl>
summary(var_grad)
#> # A tibble: 5 × 10
#>   variable     mean    median     sd    mad       q5        q95   rhat ess_bulk
#>   <chr>       <dbl>     <dbl>  <dbl>  <dbl>    <dbl>      <dbl>  <dbl>    <dbl>
#> 1 lp__         0        0     0      0          0        0      NA          NA 
#> 2 log_p__  -1065.   -1064.    1.34   1.01   -1067.   -1063.      0.999    1000.
#> 3 log_g__     -1.03    -0.714 1.03   0.731     -3.29    -0.0486  1.00      959.
#> 4 pars[1]     10.0      9.99  0.0841 0.0869     9.86    10.1     1.00     1012.
#> 5 pars[2]      2.02     2.02  0.0630 0.0619     1.92     2.13    1.00      850.
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
#>   variable        mean   median     sd    mad        q5      q95  rhat ess_bulk
#>   <chr>          <dbl>    <dbl>  <dbl>  <dbl>     <dbl>    <dbl> <dbl>    <dbl>
#> 1 lp_approx__     2.95     3.26 1.02   0.746      0.965     3.91 0.999    1007.
#> 2 lp__        -1064.   -1064.   1.10   0.771  -1066.    -1063.   0.999     993.
#> 3 pars[1]         9.91     9.92 0.0912 0.0862     9.76     10.1  0.999    1022.
#> 4 pars[2]         2.04     2.04 0.0693 0.0694     1.93      2.15 1.00      953.
#> # ℹ 1 more variable: ess_tail <dbl>
summary(path_grad)
#> # A tibble: 4 × 10
#>   variable        mean   median     sd    mad        q5      q95  rhat ess_bulk
#>   <chr>          <dbl>    <dbl>  <dbl>  <dbl>     <dbl>    <dbl> <dbl>    <dbl>
#> 1 lp_approx__     2.95     3.26 1.02   0.746      0.965     3.91 0.999    1007.
#> 2 lp__        -1064.   -1064.   1.10   0.771  -1066.    -1063.   0.999     992.
#> 3 pars[1]         9.91     9.92 0.0912 0.0862     9.76     10.1  0.999    1022.
#> 4 pars[2]         2.04     2.04 0.0693 0.0694     1.93      2.15 1.00      953.
#> # ℹ 1 more variable: ess_tail <dbl>
```
