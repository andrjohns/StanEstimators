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
stan_pathfinder <- function(fn, par_inits, ..., lower = -Inf, upper = Inf,
                          num_psis_draws = 1000,
                          num_paths = 4, save_single_paths = FALSE,
                          max_lbfgs_iters = 1000, num_draws = 1000,
                          num_elbo_draws = 25, output_dir = tempdir(),
                          control = list()) {
  fn1 <- function(v) { fn(v, ...) }
  nPars <- length(par_inits)
  finite_diff <- 1
  if ((length(par_inits) > 1) && (length(lower) == 1)) {
    lower <- rep(lower, length(par_inits))
  }
  if ((length(par_inits) > 1) && (length(upper) == 1)) {
    upper <- rep(upper, length(par_inits))
  }
  data_file <- tempfile(fileext = ".json", tmpdir = output_dir)
  output_file_base <- tempfile(tmpdir = output_dir)
  output_file <- paste0(output_file_base, ".csv")
  write_data(nPars, finite_diff, lower, upper, data_file)

  args <- c(
    "pathfinder",
    paste0("num_psis_draws=", num_psis_draws),
    paste0("num_paths=", num_paths),
    paste0("save_single_paths=", as.integer(save_single_paths)),
    paste0("max_lbfgs_iters=", max_lbfgs_iters),
    paste0("num_draws=", num_draws),
    paste0("num_elbo_draws=", num_elbo_draws),
    "data",
    paste0("file=", data_file),
    "output",
    paste0("file=", output_file)
  )

  call <- call_stan(args, ll_fun = fn1, grad_fun = fn1)

  parsed <- parse_csv(output_file)
  list(
    metadata = parsed$metadata,
    timing = parsed$timing,
    draws = posterior::as_draws_df(setNames(data.frame(parsed$samples), parsed$header))
  )
}
