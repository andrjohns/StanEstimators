call_stan_impl <- function(options_vector, ll_fun, grad_fun) {
  status <- .Call(`call_stan_`, options_vector, ll_fun, grad_fun)
  invisible(NULL)
}

parse_csv <- function(filename) {
  .Call(`parse_csv_`, filename)
}
