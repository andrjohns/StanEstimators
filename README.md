
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
                   num_chains = 1, seed = 1234)
```

We can see that the parameters were recovered accurately and that the
estimation was relatively fast: ~1 sec for 1000 warmup and 1000
iterations

``` r
unlist(fit@timing)
#>   warmup sampling 
#>    0.522    0.562
summary(fit)
#> # A tibble: 3 × 10
#>   variable    mean  median     sd    mad      q5     q95  rhat ess_bulk ess_tail
#>   <chr>      <dbl>   <dbl>  <dbl>  <dbl>   <dbl>   <dbl> <dbl>    <dbl>    <dbl>
#> 1 lp__     -1.05e3 -1.05e3 0.955  0.741  -1.05e3 -1.05e3 1.00      524.     737.
#> 2 pars[1]   9.96e0  9.96e0 0.0890 0.0899  9.81e0  1.01e1 0.999     776.     756.
#> 3 pars[2]   1.98e0  1.98e0 0.0633 0.0632  1.88e0  2.09e0 1.00     1006.     721.
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
#>    0.078    0.075
summary(fit_grad)
#> # A tibble: 3 × 10
#>   variable    mean  median     sd    mad      q5     q95  rhat ess_bulk ess_tail
#>   <chr>      <dbl>   <dbl>  <dbl>  <dbl>   <dbl>   <dbl> <dbl>    <dbl>    <dbl>
#> 1 lp__     -1.05e3 -1.05e3 0.956  0.689  -1.05e3 -1.05e3 1.00      473.     598.
#> 2 pars[1]   9.95e0  9.95e0 0.0853 0.0905  9.81e0  1.01e1 0.999     974.     743.
#> 3 pars[2]   1.98e0  1.98e0 0.0636 0.0598  1.88e0  2.09e0 1.00     1034.     592.
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
#> 1 -1049.46  9.9546  1.9739
summary(opt_grad)
#>       lp__ pars[1] pars[2]
#> 1 -1049.46  9.9546  1.9739
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
#> 1 log_p__  -1050.   -1050.    1.20   0.890  -1052.   -1049.     1.00     1027.
#> 2 log_q__     -1.04    -0.692 1.04   0.716     -3.21    -0.0582 0.999    1047.
#> 3 pars[1]     10.0     10.0   0.0897 0.0854     9.85    10.1    1.00      930.
#> 4 pars[2]      2.00     2.00  0.0665 0.0673     1.90     2.11   1.00     1051.
#> # ℹ 1 more variable: ess_tail <dbl>
summary(lapl_opt)
#> # A tibble: 4 × 10
#>   variable     mean    median     sd    mad       q5        q95  rhat ess_bulk
#>   <chr>       <dbl>     <dbl>  <dbl>  <dbl>    <dbl>      <dbl> <dbl>    <dbl>
#> 1 log_p__  -1050.   -1049.    1.06   0.712  -1052.   -1049.     0.999    1045.
#> 2 log_q__     -1.04    -0.692 1.04   0.716     -3.21    -0.0582 0.999    1047.
#> 3 pars[1]      9.95     9.96  0.0885 0.0844     9.81    10.1    1.00      932.
#> 4 pars[2]      1.97     1.97  0.0647 0.0656     1.87     2.08   1.00     1051.
#> # ℹ 1 more variable: ess_tail <dbl>
summary(lapl_est)
#> # A tibble: 4 × 10
#>   variable     mean    median     sd    mad       q5        q95  rhat ess_bulk
#>   <chr>       <dbl>     <dbl>  <dbl>  <dbl>    <dbl>      <dbl> <dbl>    <dbl>
#> 1 log_p__  -1050.   -1049.    1.06   0.712  -1052.   -1049.     0.999    1045.
#> 2 log_q__     -1.04    -0.692 1.04   0.716     -3.21    -0.0582 0.999    1047.
#> 3 pars[1]      9.95     9.96  0.0885 0.0844     9.81    10.1    1.00      932.
#> 4 pars[2]      1.97     1.97  0.0647 0.0656     1.87     2.08   1.00     1051.
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
#> 2 log_p__  -1051.    -1050.    1.67   1.33   -1054.   -1049.      1.00      996.
#> 3 log_g__     -0.966    -0.697 0.963  0.729     -3.03    -0.0399  1.00     1094.
#> 4 pars[1]      9.92      9.93  0.0796 0.0813     9.79    10.0     0.999    1104.
#> 5 pars[2]      2.06      2.06  0.0696 0.0678     1.95     2.17    1.00      944.
#> # ℹ 1 more variable: ess_tail <dbl>
summary(var_grad)
#> # A tibble: 5 × 10
#>   variable     mean    median     sd    mad       q5        q95   rhat ess_bulk
#>   <chr>       <dbl>     <dbl>  <dbl>  <dbl>    <dbl>      <dbl>  <dbl>    <dbl>
#> 1 lp__         0        0     0      0          0        0      NA          NA 
#> 2 log_p__  -1050.   -1050.    1.36   1.02   -1053.   -1049.      0.999    1003.
#> 3 log_g__     -1.03    -0.714 1.03   0.731     -3.29    -0.0486  1.00      959.
#> 4 pars[1]     10.0     10.0   0.0817 0.0844     9.91    10.2     1.00     1012.
#> 5 pars[2]      1.97     1.96  0.0612 0.0601     1.87     2.07    1.00      850.
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
#> 1 lp_appr…  3.11e0  3.40e0 0.975  0.716   1.20e0  4.04e0 1.00     1053.     981.
#> 2 lp__     -1.05e3 -1.05e3 0.947  0.682  -1.05e3 -1.05e3 0.999    1044.     956.
#> 3 pars[1]   9.95e0  9.96e0 0.0830 0.0771  9.81e0  1.01e1 1.00     1030.     972.
#> 4 pars[2]   1.98e0  1.98e0 0.0633 0.0671  1.87e0  2.08e0 0.999     760.     809.
summary(path_grad)
#> # A tibble: 4 × 10
#>   variable    mean  median     sd    mad      q5     q95  rhat ess_bulk ess_tail
#>   <chr>      <dbl>   <dbl>  <dbl>  <dbl>   <dbl>   <dbl> <dbl>    <dbl>    <dbl>
#> 1 lp_appr…  3.11e0  3.40e0 0.975  0.716   1.20e0  4.04e0 1.00     1053.     981.
#> 2 lp__     -1.05e3 -1.05e3 0.947  0.682  -1.05e3 -1.05e3 0.999    1044.     956.
#> 3 pars[1]   9.95e0  9.96e0 0.0830 0.0771  9.81e0  1.01e1 1.00     1030.     972.
#> 4 pars[2]   1.98e0  1.98e0 0.0633 0.0671  1.87e0  2.08e0 0.999     760.     809.
```
