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
#' @param seed Random seed
#' @param refresh Number of iterations for printing
#' @param output_dir Directory to store outputs
#' @param output_basename Basename to use for output files
#' @param sig_figs Number of significant digits to use for printing
#' @return \code{StanLaplace} object
#' @export
stan_diagnose <- function(fn, par_inits, additional_args = list(),
                             grad_fun = NULL, lower = -Inf, upper = Inf,
                              seed = NULL,
                              refresh = NULL,
                              output_dir = NULL,
                              output_basename = NULL,
                              sig_figs = NULL) {
  inputs <- prepare_inputs(fn, par_inits, additional_args, grad_fun, lower, upper,
                            output_dir, output_basename)
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
                          output_args = output,
                          num_threads = NULL)
  call_stan(args, ll_fun = inputs$ll_function, grad_fun = inputs$grad_function, env = parent.frame())
}
