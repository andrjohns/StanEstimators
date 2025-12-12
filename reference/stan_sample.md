# stan_sample

Estimate parameters using Stan's sampling algorithms

## Usage

``` r
stan_sample(
  fn,
  par_inits = NULL,
  n_pars = NULL,
  additional_args = list(),
  algorithm = "hmc",
  engine = "nuts",
  grad_fun = NULL,
  lower = -Inf,
  upper = Inf,
  eval_standalone = (parallel_chains > 1),
  globals = TRUE,
  packages = NULL,
  seed = NULL,
  refresh = NULL,
  quiet = FALSE,
  output_dir = NULL,
  output_basename = NULL,
  sig_figs = NULL,
  num_chains = 4,
  parallel_chains = 1,
  num_samples = 1000,
  num_warmup = 1000,
  save_warmup = NULL,
  thin = NULL,
  adapt_engaged = NULL,
  adapt_gamma = NULL,
  adapt_delta = NULL,
  adapt_kappa = NULL,
  adapt_t0 = NULL,
  adapt_init_buffer = NULL,
  adapt_term_buffer = NULL,
  adapt_window = NULL,
  int_time = NULL,
  max_treedepth = NULL,
  metric = NULL,
  metric_file = NULL,
  stepsize = NULL,
  stepsize_jitter = NULL,
  check_diagnostics = TRUE
)
```

## Arguments

- fn:

  Function to estimate parameters for

- par_inits:

  Initial values for parameters. This can either be a numeric vector of
  initial values (which will be used for all chains), a list of numeric
  vectors (of length equal to the number of chains), a function taking a
  single argument (the chain ID) and returning a numeric vector of
  initial values, or NULL (in which case Stan will generate initial
  values automatically). (must be specified if `n_pars` is NULL)

- n_pars:

  Number of parameters to estimate (must be specified if `par_inits` is
  NULL)

- additional_args:

  List of additional arguments to pass to the function

- algorithm:

  (string) The sampling algorithm. One of `"hmc"` or `"fixed_param"`.

- engine:

  (string) The `HMC` engine to use, one of `"nuts"` or `"static"`

- grad_fun:

  Either:

  - `NULL` for finite-differences (default)

  - A function for calculating gradients w.r.t. each parameter

  - "RTMB" to use the RTMB package for automatic differentiation

- lower:

  Lower bound constraint(s) for parameters

- upper:

  Upper bound constraint(s) for parameters

- eval_standalone:

  (logical) Whether to evaluate the function in a separate R session.
  Defaults to `(parallel_chains > 1)`. Must be `TRUE` if
  `parallel_chains > 1`.

- globals:

  (optional) a logical, a character vector, or a named list to control
  how globals are handled when evaluating functions in a separate R
  session. Ignored if `eval_standalone` = `FALSE`. For details, see
  section 'Globals used by future expressions' in the help for
  [`future::future()`](https://future.futureverse.org/reference/future.html).

- packages:

  (optional) a character vector specifying packages to be attached in
  the R environment evaluating the function. Ignored if
  `eval_standalone` = `FALSE`.

- seed:

  Random seed

- refresh:

  Number of iterations for printing

- quiet:

  (logical) Whether to suppress Stan's output

- output_dir:

  Directory to store outputs

- output_basename:

  Basename to use for output files

- sig_figs:

  Number of significant digits to use for printing

- num_chains:

  (positive integer) The number of Markov chains to run. The default is
  4.

- parallel_chains:

  (positive integer) The number of chains to run in parallel, the
  default is 1.

- num_samples:

  (positive integer) The number of post-warmup iterations to run per
  chain.

- num_warmup:

  (positive integer) The number of warmup iterations to run per chain.

- save_warmup:

  (logical) Should warmup iterations be saved? The default is `FALSE`.

- thin:

  (positive integer) The period between saved samples. This should
  typically be left at its default (no thinning) unless memory is a
  problem.

- adapt_engaged:

  (logical) Do warmup adaptation? The default is `TRUE`.

- adapt_gamma:

  (positive real) Adaptation regularization scale.

- adapt_delta:

  (real in `(0,1)`) The adaptation target acceptance statistic.

- adapt_kappa:

  (positive real) Adaptation relaxation exponent.

- adapt_t0:

  (positive real) Adaptation iteration offset.

- adapt_init_buffer:

  (nonnegative integer) Width of initial fast timestep adaptation
  interval during warmup.

- adapt_term_buffer:

  (nonnegative integer) Width of final fast timestep adaptation interval
  during warmup.

- adapt_window:

  (nonnegative integer) Initial width of slow timestep/metric adaptation
  interval.

- int_time:

  (positive real) Total integration time

- max_treedepth:

  (positive integer) The maximum allowed tree depth for the NUTS engine.

- metric:

  (string) One of `"diag_e"`, `"dense_e"`, or `"unit_e"`, specifying the
  geometry of the base manifold.

- metric_file:

  (character vector) The paths to JSON or Rdump files (one per chain)
  compatible with CmdStan that contain precomputed inverse metrics.

- stepsize:

  (positive real) The *initial* step size for the discrete approximation
  to continuous Hamiltonian dynamics.

- stepsize_jitter:

  (real in `(0,1)`) Allows step size to be “jittered” randomly during
  sampling to avoid any poor interactions with a fixed step size and
  regions of high curvature.

- check_diagnostics:

  (logical) Whether to check for common problems with HMC sampling
  (divergent transitions, max tree depth hits, and low Bayesian fraction
  of missing information). Default is `TRUE`.

## Value

`StanMCMC` object
