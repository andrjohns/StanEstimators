setClass("StanOptimize",
  slots = c(
    metadata = "list",
    timing = "list",
    estimates = "data.frame"
  )
)

setMethod("summary", "StanOptimize", function(object, ...) {
  object@estimates
})

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
  output_file_base <- tempfile(tmpdir = output_dir)
  output_file <- paste0(output_file_base, ".csv")
  write_data(nPars, finite_diff, lower, upper, data_file)
  write_inits(par_inits, init_file)

  args <- c(
    "optimize",
    paste0("algorithm=", algorithm),
    paste0("jacobian=", as.integer(jacobian)),
    paste0("iter=", iter),
    paste0("save_iterations=", as.integer(save_iterations)),
    "data",
    paste0("file=", data_file),
    paste0("init=", init_file),
    "output",
    paste0("file=", output_file)
  )

  call <- call_stan(args, ll_fun = fn1, grad_fun = gr1)
  parsed <- parse_csv(output_file)
  new("StanOptimize",
    metadata = parsed$metadata,
    timing = parsed$timing,
    estimates = setNames(data.frame(parsed$samples), parsed$header)
  )
}
