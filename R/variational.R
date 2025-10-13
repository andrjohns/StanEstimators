#' Summary method for objects of class \code{StanVariational}.
#'
#' @docType methods
#' @name summary-StanVariational
#' @rdname summary-StanVariational
#' @aliases summary-StanVariational summary,StanVariational-method
#'
#' @param object A \code{StanVariational} object.
#' @param ... Additional arguments, currently unused.
#'
#' @export
setMethod("summary", "StanVariational", function(object, ...) {
  posterior::summarise_draws(object@draws)
})

#' stan_variational
#'
#' Estimate parameters using Stan's variational inference algorithms
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
#' @param algorithm (string) The variational inference algorithm. One of
#'  `"meanfield"` or `"fullrank"`.
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
#' @param iter (positive integer) The _maximum_ number of iterations.
#' @param grad_samples (positive integer) The number of samples for Monte Carlo
#'   estimate of gradients.
#' @param elbo_samples (positive integer) The number of samples for Monte Carlo
#'   estimate of ELBO (objective function).
#' @param eta (positive real) The step size weighting parameter for adaptive
#'   step size sequence.
#' @param adapt_engaged (logical) Do warmup adaptation?
#' @param adapt_iter (positive integer) The _maximum_ number of adaptation
#'   iterations.
#' @param tol_rel_obj (positive real) Convergence tolerance on the relative norm
#'   of the objective.
#' @param eval_elbo (positive integer) Evaluate ELBO every Nth iteration.
#' @param output_samples (positive integer) Number of approximate posterior
#'   samples to draw and save.
#' @return \code{StanVariational} object
#' @export
stan_variational <- function(fn, par_inits = NULL, n_pars = NULL, additional_args = list(), algorithm = "meanfield",
                             grad_fun = NULL, lower = -Inf, upper = Inf,
                              eval_standalone = FALSE,
                              globals = TRUE, packages = NULL,
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
                              output_samples = NULL) {
  inputs <- prepare_inputs(fn, par_inits, n_pars, additional_args, grad_fun, lower, upper,
                            globals, packages, eval_standalone, output_dir, output_basename)
  method_args <- list(
    algorithm = algorithm,
    iter = iter,
    grad_samples = grad_samples,
    elbo_samples = elbo_samples,
    eta = eta,
    adapt = list( engaged = format_bool(adapt_engaged), iter = adapt_iter),
    tol_rel_obj = tol_rel_obj,
    eval_elbo = eval_elbo,
    output_samples = output_samples
    )

  output <- list(
    file = inputs$output_filepath,
    diagnostic_file = NULL,
    refresh = refresh,
    sig_figs = sig_figs,
    profile_file = NULL
  )
  args <- build_stan_call(method = "variational",
                          method_args = method_args,
                          data_file = inputs$data_filepath,
                          init = inputs$init_filepath[1],
                          seed = seed,
                          output_args = output)

  call_stan(args, inputs, quiet)

  parsed <- parse_csv(inputs$output_filepath, lower=inputs$lower, upper=inputs$upper)
  estimates <- setNames(data.frame(parsed$samples), parsed$header)
  methods::new("StanVariational",
    metadata = parsed$metadata,
    estimates = estimates[1,],
    draws = posterior::as_draws_df(estimates[-1,]),
    lower_bounds = inputs$lower,
    upper_bounds = inputs$upper,
    model_methods = list(
      data_json_string = inputs$data_string,
      model_pointer = make_model_pointer(inputs$data_string, seed)
    )
  )
}
