setOldClass("draws_df")
setClass("StanLaplace",
  slots = c(
    metadata = "list",
    timing = "list",
    estimates = "data.frame",
    draws = "draws_df"
  )
)

#' Summary method for objects of class \code{StanLaplace}.
#'
#' @docType methods
#' @name summary-StanLaplace
#' @rdname summary-StanLaplace
#' @aliases summary-StanLaplace summary,StanLaplace-method
#'
#' @param object A \code{StanLaplace} object.
#' @param ... Additional arguments, currently unused.
#'
#' @export
setMethod("summary", "StanLaplace", function(object, ...) {
  posterior::summarise_draws(object@draws)
})

#' stan_laplace
#'
#' Estimate parameters using Stan's laplace algorithm
#'
#' @param fn Function to estimate parameters for
#' @param par_inits Initial values
#' @param additional_args List of additional arguments to pass to the function
#' @param grad_fun Function calculating gradients w.r.t. each parameter
#' @param lower Lower bound constraint(s) for parameters
#' @param upper Upper bound constraint(s) for parameters
#' @param seed Random seed
#' @param refresh Number of iterations for printing
#' @param quiet (logical) Whether to suppress Stan's output
#' @param output_dir Directory to store outputs
#' @param output_basename Basename to use for output files
#' @param sig_figs Number of significant digits to use for printing
#' @param mode Mode for the laplace approximation, can either be a vector of
#'    values, a StanOptimize object, or NULL.
#' @param jacobian (logical) Whether or not to use the Jacobian adjustment for
#'   constrained variables.
#' @param draws (positive integer) Number of approximate posterior
#'   samples to draw and save.
#' @param opt_args (named list) A named list of optional arguments to pass to
#'  `stan_optimize()` if `mode=NULL`.
#' @return \code{StanLaplace} object
#' @export
stan_laplace <- function(fn, par_inits, additional_args = list(),
                             grad_fun = NULL, lower = -Inf, upper = Inf,
                              seed = NULL,
                              refresh = NULL,
                              quiet = FALSE,
                              output_dir = NULL,
                              output_basename = NULL,
                              sig_figs = NULL,
                              mode = NULL,
                              jacobian = NULL,
                              draws = NULL,
                              opt_args = NULL) {
  inputs <- prepare_inputs(fn, par_inits, additional_args, grad_fun, lower, upper,
                            output_dir, output_basename)
  mode_file <- paste0(inputs$output_basename, "_mode.json")
  if (!is.null(mode)) {
    if (inherits(mode, "StanOptimize")) {
      mode_vals <- mode@estimates[, -1] # First estimate is lp__
    } else {
      mode_vals <- mode
    }

  } else {
    curr_args <- list(
      fn = fn,
      par_inits = par_inits,
      additional_args = additional_args,
      grad_fun = grad_fun,
      lower = lower,
      upper = upper,
      seed = seed,
      refresh = refresh,
      output_dir = output_dir,
      output_basename = output_basename,
      sig_figs = sig_figs
    )
    opt <- do.call(stan_optimize, c(curr_args, opt_args))
    mode_vals <- opt@estimates[, -1]
  }
  mode_vals <- as.numeric(mode_vals)
  if (length(mode_vals) != length(par_inits)) {
    stop("The number of mode values does not match the number of parameter ",
          "inits!", .call = FALSE)
  }
  write_inits(mode_vals, mode_file)
  method_args <- list(
    mode = mode_file,
    jacobian = format_bool(jacobian),
    draws = draws
  )

  output <- list(
    file = inputs$output_filepath,
    diagnostic_file = NULL,
    refresh = refresh,
    sig_figs = sig_figs,
    profile_file = NULL
  )
  args <- build_stan_call(method = "laplace",
                          method_args = method_args,
                          data_file = inputs$data_filepath,
                          init = inputs$init_filepath,
                          seed = seed,
                          output_args = output)

  call_stan(args, ll_fun = inputs$ll_function, grad_fun = inputs$grad_function, quiet)

  parsed <- parse_csv(inputs$output_filepath)
  estimates <- setNames(data.frame(parsed$samples), parsed$header)

  methods::new("StanLaplace",
    metadata = parsed$metadata,
    timing = parsed$timing,
    estimates = estimates[1,],
    draws = posterior::as_draws_df(estimates[-1,])
  )
}
