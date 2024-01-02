#' stan_diagnose
#'
#' Check gradient estimation using Stan's 'Diagnose' method
#'
#' @param fn Function to estimate parameters for
#' @param par_inits Initial values
#' @param additional_args List of additional arguments to pass to the function
#' @param grad_fun Function calculating gradients w.r.t. each parameter
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
#' @return \code{StanLaplace} object
#' @export
stan_diagnose <- function(fn, par_inits, additional_args = list(),
                             grad_fun = NULL, lower = -Inf, upper = Inf,
                              eval_standalone = FALSE,
                              globals = TRUE, packages = NULL,
                              seed = NULL,
                              refresh = NULL,
                              quiet = FALSE,
                              output_dir = NULL,
                              output_basename = NULL,
                              sig_figs = NULL) {
  inputs <- prepare_inputs(fn, par_inits, additional_args, grad_fun, lower, upper,
                            globals, packages, eval_standalone, output_dir, output_basename)
  output <- list(
    file = inputs$output_filepath,
    diagnostic_file = NULL,
    refresh = refresh,
    sig_figs = sig_figs,
    profile_file = NULL
  )
  args <- build_stan_call(method = "diagnose",
                          method_args = "",
                          data_file = inputs$data_filepath,
                          init = inputs$init_filepath,
                          seed = seed,
                          output_args = output)
  call_stan(args, inputs, quiet)
}
