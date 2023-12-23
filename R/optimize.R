setClass("StanOptimize",
  slots = c(
    metadata = "list",
    timing = "list",
    estimates = "data.frame",
    inputs = "list"
  )
)

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
#' @param par_inits Initial values
#' @param additional_args List of additional arguments to pass to the function
#' @param algorithm (string) The optimization algorithm. One of `"lbfgs"`,
#'   `"bfgs"`, or `"newton"`.
#' @param grad_fun Function calculating gradients w.r.t. each parameter
#' @param lower Lower bound constraint(s) for parameters
#' @param upper Upper bound constraint(s) for parameters
#' @param seed Random seed
#' @param refresh Number of iterations for printing
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
stan_optimize <- function(fn, par_inits, additional_args = list(), algorithm = "lbfgs",
                          grad_fun = NULL, lower = -Inf, upper = Inf,
                          seed = NULL,
                          refresh = NULL,
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
  inputs <- prepare_inputs(fn, par_inits, additional_args, grad_fun, lower, upper,
                            output_dir, output_basename)

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
    jacobian = jacobian,
    iter = iter,
    save_iterations = save_iterations
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
                          init = inputs$init_filepath,
                          seed = seed,
                          output_args = output)
  call_stan(args, ll_fun = inputs$ll_function, grad_fun = inputs$grad_function)

  parsed <- parse_csv(inputs$output_filepath)

  methods::new("StanOptimize",
    metadata = parsed$metadata,
    timing = parsed$timing,
    estimates = setNames(data.frame(parsed$samples), parsed$header),
    inputs = inputs
  )
}
