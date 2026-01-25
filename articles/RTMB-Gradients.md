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
|         | Finite Differences |        10.12 |    1.00 |
| elapsed | RTMB               |         2.28 |    4.45 |

Performance comparison for Poisson regression

``` r
summary(fit_pois_rtmb)
#> # A tibble: 4 × 10
#>   variable     mean   median     sd    mad       q5      q95  rhat ess_bulk
#>   <chr>       <dbl>    <dbl>  <dbl>  <dbl>    <dbl>    <dbl> <dbl>    <dbl>
#> 1 lp__     -315.    -314.    1.22   1.09   -317.    -313.     1.00     453.
#> 2 pars[1]     0.543    0.544 0.0609 0.0609    0.436    0.637  1.01     482.
#> 3 pars[2]     1.15     1.15  0.0472 0.0488    1.08     1.23   1.01     515.
#> 4 pars[3]    -0.734   -0.734 0.0515 0.0521   -0.817   -0.651  1.00     580.
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

### Using stan_optimize

RTMB also works with other Stan methods. For example, we can find the
maximum likelihood estimate using `stan_optimize`:

``` r
fit_logit_mle <- stan_optimize(logistic_loglik, inits_logit,
                               grad_fun = "RTMB",
                               additional_args = list(y = y_binom, X = X_logit))
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
|         | Finite Differences |         5.37 |    1.00 |
| elapsed | RTMB               |         0.97 |    5.54 |

Performance comparison for Logistic regression

``` r
summary(fit_logit_rtmb)
#> # A tibble: 4 × 10
#>   variable     mean   median    sd   mad        q5      q95  rhat ess_bulk
#>   <chr>       <dbl>    <dbl> <dbl> <dbl>     <dbl>    <dbl> <dbl>    <dbl>
#> 1 lp__     -152.    -152.    1.31  1.09  -155.     -151.     1.00     546.
#> 2 pars[1]     0.224    0.225 0.151 0.150   -0.0283    0.462  1.00     736.
#> 3 pars[2]     1.56     1.55  0.206 0.207    1.24      1.89   1.00     740.
#> 4 pars[3]    -0.915   -0.920 0.176 0.174   -1.20     -0.614  1.00     600.
#> # ℹ 1 more variable: ess_tail <dbl>
```

``` r
# Maximum likelihood estimates
summary(fit_logit_mle)
#>        lp__   pars[1]  pars[2]    pars[3]
#> 1 -150.7363 0.2267225 1.517124 -0.8917454
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
  pi <- plogis(pars[1])  # mixing proportion in [0,1]
  mu1 <- pars[2]
  mu2 <- pars[3]
  sigma1 <- exp(pars[4])  # positive
  sigma2 <- exp(pars[5])  # positive

  # Log-likelihood for each component
  log_lik1 <- dnorm(y, mu1, sigma1, log = TRUE) + log(pi)
  log_lik2 <- dnorm(y, mu2, sigma2, log = TRUE) + log(1 - pi)

  # Log-sum-exp for numerical stability
  sum(log(exp(log_lik1) + exp(log_lik2)))
}
```

### Performance Comparison

``` r
# Initialize near true values (mixture models can have multimodality)
inits_mix <- c(qlogis(0.3), -2, 3, log(1), log(1.5))

# Finite differences
timing_mix_fd <- system.time({
  fit_mix_fd <- stan_sample(mixture_loglik, inits_mix,
                            additional_args = list(y = y_mix),
                            num_chains = 1, seed = 1234)
})

# RTMB
timing_mix_rtmb <- system.time({
  fit_mix_rtmb <- stan_sample(mixture_loglik, inits_mix,
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
|         | Finite Differences |        14.62 |    1.00 |
| elapsed | RTMB               |         1.79 |    8.16 |

Performance comparison for Gaussian Mixture

``` r
summary(fit_mix_rtmb)
#> # A tibble: 6 × 10
#>   variable      mean    median     sd    mad        q5      q95  rhat ess_bulk
#>   <chr>        <dbl>     <dbl>  <dbl>  <dbl>     <dbl>    <dbl> <dbl>    <dbl>
#> 1 lp__     -908.     -908.     1.55   1.38   -911.     -906.    1.00      412.
#> 2 pars[1]    -0.885    -0.885  0.124  0.120    -1.10     -0.688 1.00      806.
#> 3 pars[2]    -2.05     -2.06   0.123  0.119    -2.25     -1.85  1.00      804.
#> 4 pars[3]     2.95      2.96   0.102  0.104     2.78      3.12  1.00      907.
#> 5 pars[4]     0.0726    0.0704 0.0913 0.0916   -0.0729    0.217 1.000     910.
#> 6 pars[5]     0.408     0.408  0.0542 0.0545    0.322     0.498 1.000     987.
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
  phi <- tanh(pars[1])  # constrain to (-1, 1)
  sigma <- exp(pars[2])  # positive

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
inits_ar <- c(atanh(0.5), log(1))

# Finite differences
timing_ar_fd <- system.time({
  fit_ar_fd <- stan_sample(ar1_loglik, inits_ar,
                           additional_args = list(y = y_ar),
                           num_chains = 1, seed = 1234)
})

# RTMB
timing_ar_rtmb <- system.time({
  fit_ar_rtmb <- stan_sample(ar1_loglik, inits_ar,
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
|         | Finite Differences |        27.60 |    1.00 |
| elapsed | RTMB               |         0.76 |   36.32 |

Performance comparison for AR(1) model

``` r
summary(fit_ar_rtmb)
#> # A tibble: 3 × 10
#>   variable       mean      median     sd    mad       q5      q95  rhat ess_bulk
#>   <chr>         <dbl>       <dbl>  <dbl>  <dbl>    <dbl>    <dbl> <dbl>    <dbl>
#> 1 lp__     -284.         -2.84e+2 1.02   0.783  -2.86e+2 -2.83e+2 1.01      402.
#> 2 pars[1]     0.999       9.90e-1 0.117  0.117   8.31e-1  1.20e+0 1.00      894.
#> 3 pars[2]     0.00147    -1.16e-5 0.0501 0.0505 -7.97e-2  8.58e-2 1.000     890.
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
#>   variable          mean   median     sd    mad       q5      q95  rhat ess_bulk
#>   <chr>            <dbl>    <dbl>  <dbl>  <dbl>    <dbl>    <dbl> <dbl>    <dbl>
#> 1 lp_approx__    2.15e+0  2.65e+0 1.51   0.803  -9.44e-1  3.39e+0 1.00    356.  
#> 2 lp__          -2.84e+2 -2.84e+2 1.01   0.761  -2.87e+2 -2.83e+2 1.00    461.  
#> 3 path__         2.47e+0  2   e+0 1.11   1.48    1   e+0  4   e+0 2.51      1.22
#> 4 pars[1]        1.01e+0  9.93e-1 0.120  0.112   8.33e-1  1.22e+0 1.00    377.  
#> 5 pars[2]        5.57e-4 -9.35e-4 0.0495 0.0474 -7.97e-2  8.21e-2 1.000   781.  
#> # ℹ 1 more variable: ess_tail <dbl>
```
