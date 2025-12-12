# stan_optimize

Estimate parameters using Stan's optimization algorithms

## Usage

``` r
stan_optimize(
  fn,
  par_inits = NULL,
  n_pars = NULL,
  additional_args = list(),
  algorithm = "lbfgs",
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
  save_iterations = NULL,
  jacobian = NULL,
  init_alpha = NULL,
  iter = NULL,
  tol_obj = NULL,
  tol_rel_obj = NULL,
  tol_grad = NULL,
  tol_rel_grad = NULL,
  tol_param = NULL,
  history_size = NULL
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

  (string) The optimization algorithm. One of `"lbfgs"`, `"bfgs"`, or
  `"newton"`.

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

- save_iterations:

  Save optimization iterations to output file

- jacobian:

  (logical) Whether or not to use the Jacobian adjustment for
  constrained variables. For historical reasons, the default is `FALSE`,
  meaning optimization yields the (regularized) maximum likelihood
  estimate. Setting it to `TRUE` yields the maximum a posteriori
  estimate.

- init_alpha:

  (positive real) The initial step size parameter.

- iter:

  (positive integer) The maximum number of iterations.

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
  Hessian. Only available for L-BFGS.

## Value

`StanOptimize` object
