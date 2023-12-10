call_stan <- function(options_vector, ll_fun, grad_fun, num_threads = 1) {
  sinkfile <- tempfile()
  sink(file = file(sinkfile, open = "wt"), type = "message")
  Sys.setenv(STAN_NUM_THREADS = num_threads)
  status <- .Call(`call_stan_`, options_vector, ll_fun, grad_fun)
  sink(file = NULL, type = "message")
  sinklines <- paste(readLines(sinkfile), collapse = "\n")
  if ((status == 0) && (sinklines != "")) {
    stop(sinklines, call. = FALSE)
  }
  invisible(NULL)
}

parse_csv <- function(filename) {
  .Call(`parse_csv_`, filename)
}

constrain_pars <- function(pars, lower, upper) {
  .Call(`constrain_pars_`, as.numeric(pars), as.numeric(lower), as.numeric(upper))
}

unconstrain_pars <- function(pars, lower, upper) {
  .Call(`unconstrain_pars_`, as.numeric(pars), as.numeric(lower), as.numeric(upper))
}
