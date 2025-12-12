# stan_variational

Estimate parameters using Stan's variational inference algorithms

## Usage

``` r
stan_variational(
  fn,
  par_inits = NULL,
  n_pars = NULL,
  additional_args = list(),
  algorithm = "meanfield",
  grad_fun = NULL,
  lower = -Inf,
  upper = Inf,
  eval_standalone = FALSE,
  globals = TRUE,
  packages = NULL,
  seed = NULL,
  refresh = NULL,
  quiet = FALSE,
  output_dir = NULL,
  output_basename = NULL,
  sig_figs = NULL,
  iter = NULL,
  grad_samples = NULL,
  elbo_samples = NULL,
  eta = NULL,
  adapt_engaged = NULL,
  adapt_iter = NULL,
  tol_rel_obj = NULL,
  eval_elbo = NULL,
  output_samples = NULL
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

  (string) The variational inference algorithm. One of `"meanfield"` or
  `"fullrank"`.

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
  Defaults to `FALSE`.

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

- iter:

  (positive integer) The *maximum* number of iterations.

- grad_samples:

  (positive integer) The number of samples for Monte Carlo estimate of
  gradients.

- elbo_samples:

  (positive integer) The number of samples for Monte Carlo estimate of
  ELBO (objective function).

- eta:

  (positive real) The step size weighting parameter for adaptive step
  size sequence.

- adapt_engaged:

  (logical) Do warmup adaptation?

- adapt_iter:

  (positive integer) The *maximum* number of adaptation iterations.

- tol_rel_obj:

  (positive real) Convergence tolerance on the relative norm of the
  objective.

- eval_elbo:

  (positive integer) Evaluate ELBO every Nth iteration.

- output_samples:

  (positive integer) Number of approximate posterior samples to draw and
  save.

## Value

`StanVariational` object
