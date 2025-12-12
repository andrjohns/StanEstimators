# stan_laplace

Estimate parameters using Stan's laplace algorithm

## Usage

``` r
stan_laplace(
  fn,
  par_inits = NULL,
  n_pars = NULL,
  additional_args = list(),
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
  mode = NULL,
  jacobian = NULL,
  draws = NULL,
  opt_args = NULL
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

- mode:

  Mode for the laplace approximation, can either be a vector of values,
  a StanOptimize object, or NULL.

- jacobian:

  (logical) Whether or not to use the Jacobian adjustment for
  constrained variables.

- draws:

  (positive integer) Number of approximate posterior samples to draw and
  save.

- opt_args:

  (named list) A named list of optional arguments to pass to
  [`stan_optimize()`](https://andrjohns.github.io/StanEstimators/reference/stan_optimize.md)
  if `mode=NULL`.

## Value

`StanLaplace` object
