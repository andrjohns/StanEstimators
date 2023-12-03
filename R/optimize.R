#' stan_optimize
#'
#' Estimate parameters using Stan's optimization algorithms
#'
#' @param fn
#' @param par_inits
#' @param ...
#' @param algorithm
#' @param jacobian
#' @param iter
#' @param save_iterations
#' @param output_dir
#' @param laplace_draws
#' @param laplace_jacobian
#' @param control
#' @return
#' @export
stan_optimize <- function(fn, par_inits, ..., algorithm = "lbfgs",
                          grad_fun = NULL, lower = -Inf, upper = Inf,
                          jacobian = FALSE, iter = 2000,
                          save_iterations = FALSE, output_dir = tempdir(),
                          laplace_draws = NULL, laplace_jacobian = NULL, control = list()) {
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
  output_file_base <- tempfile(tmpdir = output_dir)
  output_file <- paste0(output_file_base, ".csv")
  write_data(nPars, finite_diff, lower, upper, data_file)

  args <- c(
    "optimize",
    paste0("algorithm=", algorithm),
    paste0("jacobian=", as.integer(jacobian)),
    paste0("iter=", iter),
    paste0("save_iterations=", as.integer(save_iterations)),
    "data",
    paste0("file=", data_file),
    "output",
    paste0("file=", output_file)
  )

  call <- call_stan(args, ll_fun = fn1, grad_fun = gr1)
  parsed <- parse_csv(output_file)
  ret_list <- list(
    metadata = parsed$metadata,
    timing = parsed$timing,
    estimates = setNames(data.frame(parsed$samples), parsed$header)
  )

  if (!is.null(laplace_draws)) {
    laplace_output <- paste0(output_file_base, "_laplace.csv")
    laplace_args <- c(
      "laplace",
      paste0("mode=", output_file),
      paste0("jacobian=", as.integer(laplace_jacobian)),
      paste0("draws=", laplace_draws),
      "data",
      paste0("file=", data_file),
      "output",
      paste0("file=", laplace_output)
    )
    laplace_call <- call_stan(laplace_args, ll_fun = fn1, grad_fun = fn1)
    laplace_parsed <- parse_csv(laplace_output)
    ret_list$draws <- posterior::as_draws_df(setNames(data.frame(laplace_parsed$samples), laplace_parsed$header))
  }
  ret_list
}
