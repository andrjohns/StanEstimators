#' @export
call_stan <- function(options_vector, ll_fun, grad_fun) {
  .Call(`call_stan_`, options_vector, ll_fun, grad_fun)
}

#' @export
parse_csv <- function(filename) {
  .Call(`parse_csv_`, filename)
}
