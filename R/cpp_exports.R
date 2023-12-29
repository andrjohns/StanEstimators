call_stan_impl <- function(options_vector, ll_fun, grad_fun) {
  status <- .Call(`call_stan_`, options_vector, ll_fun, grad_fun)
}

parse_csv <- function(filename) {
  .Call(`parse_csv_`, filename)
}
