setClass("StanVariational",
  slots = c(
    metadata = "list",
    timing = "list",
    estimates = "data.frame",
    draws = "draws_df"
  )
)

#' @export
setMethod("summary", "StanVariational", function(object, ...) {
  posterior::summarise_draws(object@draws)
})

#' stan_variational
#'
#' Estimate parameters using Stan's variational inference algorithms
#'
#' @param fn
#' @param par_inits
#' @param ...
#' @param algorithm
#' @param output_samples
#' @param iter
#' @param grad_samples
#' @param elbo_samples
#' @param eval_elbo
#' @param output_dir
#' @param control
#' @return
#' @export
stan_variational <- function(fn, par_inits, ..., algorithm = "meanfield",
                             grad_fun = NULL, lower = -Inf, upper = Inf,
                              seed = NULL,
                              refresh = NULL,
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
  inputs <- prepare_inputs(fn, par_inits, list(...), grad_fun, lower, upper,
                            output_dir, output_basename)
  method_args <- list(
    algorithm = algorithm,
    iter = iter,
    grad_samples = grad_samples,
    elbo_samples = elbo_samples,
    eta = eta,
    adapt = list( engaged = adapt_engaged, iter = adapt_iter),
    tol_rel_obj = NULL,
    eval_elbo = NULL,
    output_samples = NULL
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
                          init = inputs$init_filepath,
                          seed = seed,
                          output_args = output,
                          num_threads = NULL)

  call_stan(args, ll_fun = inputs$ll_function, grad_fun = inputs$grad_function)

  parsed <- parse_csv(inputs$output_filepath)
  estimates <- setNames(data.frame(parsed$samples), parsed$header)
  methods::new("StanVariational",
    metadata = parsed$metadata,
    timing = parsed$timing,
    estimates = estimates[1,],
    draws = posterior::as_draws_df(estimates[-1,])
  )
}
