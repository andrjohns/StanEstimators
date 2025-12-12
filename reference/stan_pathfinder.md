# stan_pathfinder

Estimate parameters using Stan's pathfinder algorithm

## Usage

``` r
stan_pathfinder(
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
  init_alpha = NULL,
  tol_obj = NULL,
  tol_rel_obj = NULL,
  tol_grad = NULL,
  tol_rel_grad = NULL,
  tol_param = NULL,
  history_size = NULL,
  num_psis_draws = NULL,
  num_paths = NULL,
  save_single_paths = NULL,
  max_lbfgs_iters = NULL,
  num_draws = NULL,
  num_elbo_draws = NULL
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

- init_alpha:

  (positive real) The initial step size parameter.

- tol_obj:

  (positive real) Convergence tolerance on changes in objective function
  value.

- tol_rel_obj:

  (positive real) Convergence tolerance on relative changes in objective
  function value.

- tol_grad:

  (positive real) Convergence tolerance on the norm of the gradient.

- tol_rel_grad:

  (positive real) Convergence tolerance on the relative norm of the
  gradient.

- tol_param:

  (positive real) Convergence tolerance on changes in parameter value.

- history_size:

  (positive integer) The size of the history used when approximating the
  Hessian.

- num_psis_draws:

  (positive integer) Number PSIS draws to return.

- num_paths:

  (positive integer) Number of single pathfinders to run.

- save_single_paths:

  (logical) Whether to save the results of single pathfinder runs in
  multi-pathfinder.

- max_lbfgs_iters:

  (positive integer) The maximum number of iterations for LBFGS.

- num_draws:

  (positive integer) Number of draws to return after performing pareto
  smooted importance sampling (PSIS).

- num_elbo_draws:

  (positive integer) Number of draws to make when calculating the ELBO
  of the approximation at each iteration of LBFGS.

## Value

`StanPathfinder` object
