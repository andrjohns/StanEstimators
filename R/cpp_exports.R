call_stan <- function(options_vector, ll_fun, grad_fun) {
  sinkfile <- tempfile()
  sink(file = file(sinkfile, open = "wt"), type = "message")
  status <- .Call(`call_stan_`, options_vector, ll_fun, grad_fun)
  sink(file = NULL, type = "message")

  if (status == 0) {
    stop(paste(readLines(sinkfile), collapse = "\n"), call. = FALSE)
  }
  invisible(NULL)
}

parse_csv <- function(filename) {
  .Call(`parse_csv_`, filename)
}

constrain_pars <- function(pars, lower, upper) {
  .Call(`constrain_pars_`, pars, lower, upper)
}

unconstrain_pars <- function(pars, lower, upper) {
  .Call(`unconstrain_pars_`, pars, lower, upper)
}
