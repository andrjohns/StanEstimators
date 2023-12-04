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
                             output_samples = 1000, iter = 1000,
                             grad_samples = 1, elbo_samples = 100,
                             eval_elbo = 100, output_dir = tempdir(),
                             control = list()) {
  fn1 <- prepare_function(fn, par_inits, ..., grad = FALSE)
  if (!is.null(grad_fun)) {
    gr1 <- prepare_function(grad_fun, par_inits, ..., grad = TRUE)
  } else {
    gr1 <- fn1
  }

  nPars <- length(par_inits)
  finite_diff <- as.integer(is.null(grad_fun))

  if ((length(par_inits) > 1) && (length(lower) == 1)) {
    lower <- rep(lower, length(par_inits))
  }
  if ((length(par_inits) > 1) && (length(upper) == 1)) {
    upper <- rep(upper, length(par_inits))
  }
  data_file <- tempfile(fileext = ".json", tmpdir = output_dir)
  init_file <- tempfile(fileext = ".json", tmpdir = output_dir)
  output_file <- tempfile(fileext = ".csv", tmpdir = output_dir)
  output_file_base <- tools::file_path_sans_ext(output_file)
  write_data(nPars, finite_diff, lower, upper, data_file)
  write_inits(par_inits, init_file)

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
    paste0("init=", init_file),
    "output",
    paste0("file=", output_file)
  )

  call <- call_stan(args, ll_fun = fn1, grad_fun = gr1)
  parsed <- parse_csv(output_file)
  estimates <- setNames(data.frame(parsed$samples), parsed$header)
  methods::new("StanVariational",
    metadata = parsed$metadata,
    timing = parsed$timing,
    estimates = estimates[1,],
    draws = posterior::as_draws_df(estimates[-1,])
  )
}
