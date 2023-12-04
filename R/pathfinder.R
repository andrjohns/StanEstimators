setOldClass("draws_df")
setClass("StanPathfinder",
  slots = c(
    metadata = "list",
    timing = "list",
    draws = "draws_df"
  )
)

#' @export
setMethod("summary", "StanPathfinder", function(object, ...) {
  posterior::summarise_draws(object@draws)
})

#' stan_pathfinder
#'
#' Estimate parameters using Stan's pathfinder algorithm
#'
#' @param fn
#' @param par_inits
#' @param ...
#' @param num_psis_draws
#' @param num_paths
#' @param save_single_paths
#' @param max_lbfgs_iters
#' @param num_draws
#' @param num_elbo_draws
#' @param output_dir
#' @param control
#' @return
#' @export
stan_pathfinder <- function(fn, par_inits, ..., grad_fun = NULL,
                          lower = -Inf, upper = Inf,
                          seed = NULL,
                          refresh = NULL,
                          output_dir = NULL,
                          output_basename = NULL,
                          sig_figs = NULL,
                          init_alpha = NULL, tol_obj = NULL,
                          tol_rel_obj = NULL, tol_grad = NULL,
                          tol_rel_grad = NULL, tol_param = NULL,
                          history_size = NULL, num_psis_draws = NULL,
                          num_paths = NULL, save_single_paths = NULL,
                          max_lbfgs_iters = NULL, num_draws = NULL,
                          num_elbo_draws = NULL) {
  inputs <- prepare_inputs(fn, par_inits, list(...), grad_fun, lower, upper,
                            output_dir, output_basename)
  method_args <- list(
    init_alpha = init_alpha,
    tol_obj = tol_obj,
    tol_rel_obj = tol_rel_obj,
    tol_grad = tol_grad,
    tol_rel_grad = tol_rel_grad,
    tol_param = tol_param,
    history_size = history_size,
    num_psis_draws = num_psis_draws,
    num_paths = num_paths,
    save_single_paths = save_single_paths,
    max_lbfgs_iters = max_lbfgs_iters,
    num_draws = num_draws,
    num_elbo_draws = num_elbo_draws
  )

  output <- list(
    file = inputs$output_filepath,
    diagnostic_file = NULL,
    refresh = refresh,
    sig_figs = sig_figs,
    profile_file = NULL
  )

  args <- build_stan_call(method = "pathfinder",
                          method_args = method_args,
                          data_file = inputs$data_filepath,
                          init = inputs$init_filepath,
                          seed = seed,
                          output_args = output,
                          num_threads = NULL)

  call_stan(args, ll_fun = inputs$ll_function, grad_fun = inputs$grad_function)

  parsed <- parse_csv(inputs$output_filepath)

  methods::new("StanPathfinder",
    metadata = parsed$metadata,
    timing = parsed$timing,
    draws = posterior::as_draws_df(setNames(data.frame(parsed$samples), parsed$header))
  )
}
