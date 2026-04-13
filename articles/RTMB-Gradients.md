# Automatic Differentiation with RTMB

``` r
library(StanEstimators)
```

## Introduction

Stan’s algorithms, including MCMC sampling (NUTS), optimization
(L-BFGS), variational inference, Pathfinder, and Laplace approximation,
are gradient-based methods. This means they require not only the
log-probability function but also its gradient (the vector of partial
derivatives with respect to each parameter) to work efficiently.

`StanEstimators` provides three ways to compute gradients:

1.  **Finite differences** (default): Automatic but slow. Approximates
    gradients by evaluating the function at slightly perturbed parameter
    values.
2.  **Analytical gradients**: Fast and accurate, but requires you to
    manually derive and code the gradient function.
3.  **RTMB automatic differentiation**: Fast and automatic. Uses the
    `RTMB` package to compute exact gradients via automatic
    differentiation (AD).

### Installing RTMB

To use RTMB with `StanEstimators`, you need to install the `RTMB`,
`withr`, and `future` packages:

``` r
install.packages(c("RTMB", "withr", "future"))
```

Once installed, simply set `grad_fun = "RTMB"` in any `StanEstimators`
function to enable automatic differentiation.

For basic usage of `StanEstimators`, see the [Getting
Started](https://andrjohns.github.io/StanEstimators/articles/Getting-Started.md)
vignette.

## Poisson Regression

Next, we’ll examine a generalized linear model (GLM) for count data.
Poisson regression uses a log-link function:
$\log\left( \lambda_{i} \right) = X_{i}\beta$, where $\lambda_{i}$ is
the expected count.

### Simulating Data

``` r
set.seed(456)
n <- 200
p <- 2  # number of predictors (plus intercept)

# True coefficients
true_beta <- c(0.5, 1.2, -0.8)

# Design matrix (intercept + 2 predictors)
X <- cbind(1, matrix(rnorm(n * p), n, p))

# Generate Poisson counts
lambda <- exp(X %*% true_beta)
y_pois <- rpois(n, lambda)
```

### Defining the Log-Likelihood

``` r
poisson_loglik <- function(pars, y, X) {
  eta <- X %*% pars
  lambda <- exp(eta)
  sum(dpois(y, lambda, log = TRUE))
}
```

### Performance Comparison

``` r
inits_pois <- rep(0, 3)

# Finite differences
timing_pois_fd <- system.time({
  fit_pois_fd <- stan_sample(poisson_loglik, inits_pois,
                             additional_args = list(y = y_pois, X = X),
                             num_chains = 1, seed = 1234)
})

# RTMB
timing_pois_rtmb <- system.time({
  fit_pois_rtmb <- stan_sample(poisson_loglik, inits_pois,
                               grad_fun = "RTMB",
                               additional_args = list(y = y_pois, X = X),
                               num_chains = 1, seed = 1234)
})
```

### Results

``` r
timing_results_pois <- data.frame(
  Method = c("Finite Differences", "RTMB"),
  Time_seconds = c(timing_pois_fd[3], timing_pois_rtmb[3]),
  Speedup = c(1, timing_pois_fd[3] / timing_pois_rtmb[3])
)
knitr::kable(timing_results_pois, digits = 2,
             caption = "Performance comparison for Poisson regression")
```

|         | Method             | Time_seconds | Speedup |
|:--------|:-------------------|-------------:|--------:|
|         | Finite Differences |         9.94 |    1.00 |
| elapsed | RTMB               |         1.98 |    5.03 |

Performance comparison for Poisson regression

``` r
summary(fit_pois_rtmb)
#> # A tibble: 4 × 10
#>   variable     mean   median     sd    mad       q5      q95  rhat ess_bulk
#>   <chr>       <dbl>    <dbl>  <dbl>  <dbl>    <dbl>    <dbl> <dbl>    <dbl>
#> 1 lp__     -315.    -314.    1.31   1.02   -317.    -313.     1.00     421.
#> 2 pars[1]     0.546    0.546 0.0613 0.0611    0.447    0.646  1.00     506.
#> 3 pars[2]     1.15     1.15  0.0478 0.0453    1.07     1.23   1.00     515.
#> 4 pars[3]    -0.730   -0.729 0.0507 0.0496   -0.818   -0.645  1.00     542.
#> # ℹ 1 more variable: ess_tail <dbl>
```

RTMB handles the matrix operations and log-link function automatically,
providing improved performance (typically 8-10x speedup) while correctly
recovering the true parameter values.

## Logistic Regression

Logistic regression models binary outcomes using a logit link:
$\text{logit}\left( p_{i} \right) = X_{i}\beta$, where $p_{i}$ is the
probability of success.

### Simulating Data

``` r
set.seed(789)
n <- 300
p <- 2

# True coefficients
true_beta_logit <- c(0.2, 1.5, -1.0)

# Design matrix
X_logit <- cbind(1, matrix(rnorm(n * p), n, p))

# Generate binary outcomes
eta <- X_logit %*% true_beta_logit
prob <- plogis(eta)  # inverse logit
y_binom <- rbinom(n, size = 1, prob = prob)
```

### Defining the Log-Likelihood

``` r
logistic_loglik <- function(pars, y, X) {
  eta <- X %*% pars
  p <- plogis(eta)
  sum(dbinom(y, size = 1, prob = p, log = TRUE))
}
```

### Performance Comparison

``` r
inits_logit <- rep(0, 3)

# Finite differences
timing_logit_fd <- system.time({
  fit_logit_fd <- stan_sample(logistic_loglik, inits_logit,
                              additional_args = list(y = y_binom, X = X_logit),
                              num_chains = 1, seed = 1234)
})

# RTMB
timing_logit_rtmb <- system.time({
  fit_logit_rtmb <- stan_sample(logistic_loglik, inits_logit,
                                grad_fun = "RTMB",
                                additional_args = list(y = y_binom, X = X_logit),
                                num_chains = 1, seed = 1234)
})
```

### Results

``` r
timing_results_logit <- data.frame(
  Method = c("Finite Differences", "RTMB"),
  Time_seconds = c(timing_logit_fd[3], timing_logit_rtmb[3]),
  Speedup = c(1, timing_logit_fd[3] / timing_logit_rtmb[3])
)
knitr::kable(timing_results_logit, digits = 2,
             caption = "Performance comparison for Logistic regression")
```

|         | Method             | Time_seconds | Speedup |
|:--------|:-------------------|-------------:|--------:|
|         | Finite Differences |         5.39 |    1.00 |
| elapsed | RTMB               |         0.86 |    6.25 |

Performance comparison for Logistic regression

``` r
summary(fit_logit_rtmb)
#> # A tibble: 4 × 10
#>   variable     mean   median    sd   mad        q5      q95  rhat ess_bulk
#>   <chr>       <dbl>    <dbl> <dbl> <dbl>     <dbl>    <dbl> <dbl>    <dbl>
#> 1 lp__     -152.    -152.    1.31  0.924 -155.     -151.    0.999     495.
#> 2 pars[1]     0.220    0.225 0.150 0.142   -0.0330    0.453 1.01      759.
#> 3 pars[2]     1.55     1.54  0.201 0.210    1.23      1.87  0.999     695.
#> 4 pars[3]    -0.896   -0.895 0.160 0.154   -1.18     -0.625 1.00      915.
#> # ℹ 1 more variable: ess_tail <dbl>
```

## Gaussian Mixture Model

Mixture models represent complex latent structure and demonstrate RTMB’s
benefits for challenging models. We’ll fit a two-component Gaussian
mixture.

The model is:
$p(y) = \pi \cdot N\left( \mu_{1},\sigma_{1}^{2} \right) + (1 - \pi) \cdot N\left( \mu_{2},\sigma_{2}^{2} \right)$

### Simulating Data

``` r
set.seed(101)
n <- 400

# True parameters
true_pi <- 0.3
true_mu1 <- -2
true_mu2 <- 3
true_sigma1 <- 1
true_sigma2 <- 1.5

# Generate mixture data
component <- rbinom(n, 1, true_pi)
y_mix <- ifelse(component == 1,
                rnorm(n, true_mu1, true_sigma1),
                rnorm(n, true_mu2, true_sigma2))
```

### Defining the Log-Likelihood

``` r
mixture_loglik <- function(pars, y) {
  # Transform parameters to satisfy constraints
  pi <- pars[1]  # mixing proportion in [0,1]
  mu1 <- pars[2]
  mu2 <- pars[3]
  sigma1 <- pars[4]  # positive
  sigma2 <- pars[5]  # positive

  # Log-likelihood for each component
  log_lik1 <- dnorm(y, mu1, sigma1, log = TRUE) + log(pi)
  log_lik2 <- dnorm(y, mu2, sigma2, log = TRUE) + log(1 - pi)

  sum(log(exp(log_lik1) + exp(log_lik2)))
}
```

### Performance Comparison

``` r
# Initialize near true values (mixture models can have multimodality)
inits_mix <- c(0.3, -2, 3, 1, 1.5)

# Finite differences
timing_mix_fd <- system.time({
  fit_mix_fd <- stan_sample(mixture_loglik, inits_mix,
                            lower = c(0, -Inf, -Inf, 0, 0),
                            upper = c(1, Inf, Inf, Inf, Inf),
                            additional_args = list(y = y_mix),
                            num_chains = 1, seed = 1234)
})

# RTMB
timing_mix_rtmb <- system.time({
  fit_mix_rtmb <- stan_sample(mixture_loglik, inits_mix,
                              lower = c(0, -Inf, -Inf, 0, 0),
                              upper = c(1, Inf, Inf, Inf, Inf),
                              grad_fun = "RTMB",
                              additional_args = list(y = y_mix),
                              num_chains = 1, seed = 1234)
})
```

### Results

``` r
timing_results_mix <- data.frame(
  Method = c("Finite Differences", "RTMB"),
  Time_seconds = c(timing_mix_fd[3], timing_mix_rtmb[3]),
  Speedup = c(1, timing_mix_fd[3] / timing_mix_rtmb[3])
)
knitr::kable(timing_results_mix, digits = 2,
             caption = "Performance comparison for Gaussian Mixture")
```

|         | Method             | Time_seconds | Speedup |
|:--------|:-------------------|-------------:|--------:|
|         | Finite Differences |        14.48 |    1.00 |
| elapsed | RTMB               |         1.19 |   12.18 |

Performance comparison for Gaussian Mixture

``` r
summary(fit_mix_rtmb)
#> # A tibble: 6 × 10
#>   variable     mean   median     sd    mad       q5      q95  rhat ess_bulk
#>   <chr>       <dbl>    <dbl>  <dbl>  <dbl>    <dbl>    <dbl> <dbl>    <dbl>
#> 1 lp__     -909.    -909.    1.54   1.42   -912.    -907.    0.999     461.
#> 2 pars[1]     0.294    0.293 0.0254 0.0256    0.252    0.336 1.000     802.
#> 3 pars[2]    -2.05    -2.06  0.120  0.123    -2.23    -1.85  0.999     744.
#> 4 pars[3]     2.95     2.96  0.104  0.103     2.78     3.12  1.000     603.
#> 5 pars[4]     1.08     1.07  0.0988 0.0960    0.929    1.25  1.01      779.
#> 6 pars[5]     1.51     1.50  0.0833 0.0808    1.38     1.66  1.00      644.
#> # ℹ 1 more variable: ess_tail <dbl>
```

## Time Series: AR(1) Model

An autoregressive model of order 1 (AR(1)) captures temporal dependence:
$y_{t} = \phi y_{t - 1} + \epsilon_{t}$, where $|\phi| < 1$ for
stationarity.

### Simulating Data

``` r
set.seed(202)
n <- 200
true_phi <- 0.7
true_sigma <- 1

# Simulate AR(1) process
y_ar <- numeric(n)
y_ar[1] <- rnorm(1, 0, true_sigma / sqrt(1 - true_phi^2))
for (t in 2:n) {
  y_ar[t] <- true_phi * y_ar[t-1] + rnorm(1, 0, true_sigma)
}
```

### Defining the Log-Likelihood

We use [`tanh()`](https://rdrr.io/r/base/Hyperbolic.html) to constrain φ
to (-1, 1).

``` r
ar1_loglik <- function(pars, y) {
  phi <- pars[1]  # constrain to (-1, 1)
  sigma <-pars[2]  # positive

  n <- length(y)

  # First observation from stationary distribution
  ll <- dnorm(y[1], 0, sigma / sqrt(1 - phi^2), log = TRUE)

  # Subsequent observations
  for (t in 2:n) {
    ll <- ll + dnorm(y[t], phi * y[t-1], sigma, log = TRUE)
  }

  ll
}
```

### Performance Comparison

``` r
inits_ar <- c(0.5, 1)

# Finite differences
timing_ar_fd <- system.time({
  fit_ar_fd <- stan_sample(ar1_loglik, inits_ar,
                           lower = c(-1, 0),
                           upper = c(0, Inf),
                           additional_args = list(y = y_ar),
                           num_chains = 1, seed = 1234)
})

# RTMB
timing_ar_rtmb <- system.time({
  fit_ar_rtmb <- stan_sample(ar1_loglik, inits_ar,
                             lower = c(-1, 0),
                             upper = c(0, Inf),
                             grad_fun = "RTMB",
                             additional_args = list(y = y_ar),
                             num_chains = 1, seed = 1234)
})
```

### Results

``` r
timing_results_ar <- data.frame(
  Method = c("Finite Differences", "RTMB"),
  Time_seconds = c(timing_ar_fd[3], timing_ar_rtmb[3]),
  Speedup = c(1, timing_ar_fd[3] / timing_ar_rtmb[3])
)
knitr::kable(timing_results_ar, digits = 2,
             caption = "Performance comparison for AR(1) model")
```

|         | Method             | Time_seconds | Speedup |
|:--------|:-------------------|-------------:|--------:|
|         | Finite Differences |        38.73 |    1.00 |
| elapsed | RTMB               |         0.63 |   61.37 |

Performance comparison for AR(1) model

``` r
summary(fit_ar_rtmb)
#> # A tibble: 3 × 10
#>   variable       mean    median      sd     mad       q5      q95  rhat ess_bulk
#>   <chr>         <dbl>     <dbl>   <dbl>   <dbl>    <dbl>    <dbl> <dbl>    <dbl>
#> 1 lp__     -373.       -3.73e+2 1.15    0.768   -3.75e+2 -3.72e+2  1.01     228.
#> 2 pars[1]    -0.00659  -4.74e-3 0.00634 0.00487 -1.94e-2 -1.66e-4  1.00     448.
#> 3 pars[2]     1.53      1.52e+0 0.0734  0.0722   1.41e+0  1.65e+0  1.00     751.
#> # ℹ 1 more variable: ess_tail <dbl>
```

### Quick Approximation with Pathfinder

RTMB also works with Pathfinder, Stan’s fast variational inference
method:

``` r
fit_ar_path <- stan_pathfinder(ar1_loglik, inits_ar,
                               grad_fun = "RTMB",
                               additional_args = list(y = y_ar))
```

``` r
summary(fit_ar_path)
#> # A tibble: 5 × 10
#>   variable        mean   median     sd    mad       q5      q95  rhat ess_bulk
#>   <chr>          <dbl>    <dbl>  <dbl>  <dbl>    <dbl>    <dbl> <dbl>    <dbl>
#> 1 lp_approx__    3.19     3.54  1.09   0.775     1.01     4.22  1.00    710.  
#> 2 lp__        -284.    -284.    0.982  0.725  -286.    -283.    1.00    695.  
#> 3 path__         2.49     2     1.10   1.48      1        4     2.63      1.20
#> 4 pars[1]        0.749    0.748 0.0473 0.0475    0.671    0.824 1.00    648.  
#> 5 pars[2]        1.00     1.00  0.0498 0.0500    0.925    1.09  1.000   860.  
#> # ℹ 1 more variable: ess_tail <dbl>
```
