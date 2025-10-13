#' Summary method for objects of class \code{StanOptimize}.
#'
#' @docType methods
#' @name summary-StanOptimize
#' @rdname summary-StanOptimize
#' @aliases summary-StanOptimize summary,StanOptimize-method
#'
#' @param object A \code{StanOptimize} object.
#' @param ... Additional arguments, currently unused.
#'
#' @export
setMethod("summary", "StanOptimize", function(object, ...) {
  object@estimates
})

#' stan_optimize
#'
#' Estimate parameters using Stan's optimization algorithms
#'
#' @param fn Function to estimate parameters for
#' @param par_inits Initial values for parameters. This can either be a numeric vector
#'  of initial values (which will be used for all chains), a list of numeric vectors (of length
#'  equal to the number of chains), a function taking a single argument (the chain ID) and
#'  returning a numeric vector of initial values, or NULL (in which case Stan will
#'  generate initial values automatically).
#'  (must be specified if `n_pars` is NULL)
#' @param n_pars Number of parameters to estimate
#'  (must be specified if `par_inits` is NULL)
#' @param additional_args List of additional arguments to pass to the function
#' @param algorithm (string) The optimization algorithm. One of `"lbfgs"`,
#'   `"bfgs"`, or `"newton"`.
#' @param grad_fun Either:
#'   - `NULL` for finite-differences (default)
#'   - A function for calculating gradients w.r.t. each parameter
#'   - "RTMB" to use the RTMB package for automatic differentiation
#' @param lower Lower bound constraint(s) for parameters
#' @param upper Upper bound constraint(s) for parameters
#' @param eval_standalone (logical) Whether to evaluate the function in a
#'    separate R session. Defaults to \code{FALSE}.
#' @param globals (optional) a logical, a character vector, or a named list
#'    to control how globals are handled when evaluating functions in a
#'    separate R session. Ignored if `eval_standalone` = `FALSE`.
#'    For details, see section 'Globals used by future expressions'
#'    in the help for [future::future()].
#' @param packages (optional) a character vector specifying packages
#'    to be attached in the \R environment evaluating the function.
#'    Ignored if `eval_standalone` = `FALSE`.
#' @param seed Random seed
#' @param refresh Number of iterations for printing
#' @param quiet (logical) Whether to suppress Stan's output
#' @param output_dir Directory to store outputs
#' @param output_basename Basename to use for output files
#' @param sig_figs Number of significant digits to use for printing
#' @param save_iterations Save optimization iterations to output file
#' @param jacobian (logical) Whether or not to use the Jacobian adjustment for
#'   constrained variables. For historical reasons, the default is `FALSE`, meaning optimization
#'   yields the (regularized) maximum likelihood estimate. Setting it to `TRUE`
#'   yields the maximum a posteriori estimate.
#' @param init_alpha (positive real) The initial step size parameter.
#' @param iter (positive integer) The maximum number of iterations.
#' @param tol_obj (positive real) Convergence tolerance on changes in objective function value.
#' @param tol_rel_obj (positive real) Convergence tolerance on relative changes in objective function value.
#' @param tol_grad (positive real) Convergence tolerance on the norm of the gradient.
#' @param tol_rel_grad (positive real) Convergence tolerance on the relative norm of the gradient.
#' @param tol_param (positive real) Convergence tolerance on changes in parameter value.
#' @param history_size (positive integer) The size of the history used when
#'   approximating the Hessian. Only available for L-BFGS.
#' @return \code{StanOptimize} object
#' @export
stan_optimize <- function(fn, par_inits = NULL, n_pars = NULL, additional_args = list(), algorithm = "lbfgs",
                          grad_fun = NULL, lower = -Inf, upper = Inf,
                          eval_standalone = FALSE,
                          globals = TRUE, packages = NULL,
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
                          history_size = NULL) {
  inputs <- prepare_inputs(fn, par_inits, n_pars, additional_args, grad_fun, lower, upper,
                            globals, packages, eval_standalone, output_dir, output_basename)
  method_args <- list(
    algorithm = algorithm,
    algorithm_args = list(
      init_alpha = init_alpha,
      tol_obj = tol_obj,
      tol_rel_obj = tol_rel_obj,
      tol_grad = tol_grad,
      tol_rel_grad = tol_rel_grad,
      tol_param = tol_param,
      history_size = history_size
    ),
    jacobian = format_bool(jacobian),
    iter = iter,
    save_iterations = format_bool(save_iterations)
  )

  output <- list(
    file = inputs$output_filepath,
    diagnostic_file = NULL,
    refresh = refresh,
    sig_figs = sig_figs,
    profile_file = NULL
  )
  args <- build_stan_call(method = "optimize",
                          method_args = method_args,
                          data_file = inputs$data_filepath,
                          init = inputs$init_filepath[1],
                          seed = seed,
                          output_args = output)
  call_stan(args, inputs, quiet)

  parsed <- parse_csv(inputs$output_filepath, lower=inputs$lower, upper=inputs$upper)

  methods::new("StanOptimize",
    metadata = parsed$metadata,
    estimates = setNames(data.frame(parsed$samples), parsed$header),
    lower_bounds = inputs$lower,
    upper_bounds = inputs$upper,
    model_methods = list(
      data_json_string = inputs$data_string,
      model_pointer = make_model_pointer(inputs$data_string, seed)
    )
  )
}
