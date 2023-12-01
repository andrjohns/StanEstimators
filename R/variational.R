#' @export
stan_variational <- function(fn, par_inits, ..., algorithm = "meanfield",
                             output_samples = 1000, iter = 1000,
                             grad_samples = 1, elbo_samples = 100,
                             eval_elbo = 100, output_dir = tempdir(),
                             control = list()) {
  fn1 <- function(v) { fn(v, ...) }
  nPars <- length(par_inits)
  finite_diff <- 1

  data_file <- tempfile(fileext = ".json", tmpdir = output_dir)
  output_file <- tempfile(fileext = ".csv", tmpdir = output_dir)
  output_file_base <- tools::file_path_sans_ext(output_file)
  write_data(nPars, finite_diff, data_file)

  args <- c(
    "variational",
    paste0("algorithm=", algorithm),
    paste0("output_samples=", as.integer(output_samples)),
    paste0("iter=", iter),
    paste0("grad_samples=", as.integer(grad_samples)),
    paste0("elbo_samples=", as.integer(elbo_samples)),
    paste0("eval_elbo=", as.integer(eval_elbo)),
    "data",
    paste0("file=", data_file),
    "output",
    paste0("file=", output_file)
  )

  call <- call_stan(args, ll_fun = fn1, grad_fun = fn1)
  parsed <- parse_csv(output_file)
  estimates <- setNames(data.frame(parsed$samples), parsed$header)
  list(
    metadata = parsed$metadata,
    timing = parsed$timing,
    estimates = estimates[1,],
    draws = posterior::as_draws_df(estimates[-1,])
  )
}
