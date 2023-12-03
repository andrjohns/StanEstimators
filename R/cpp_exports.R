call_stan <- function(options_vector, ll_fun, grad_fun) {
  .Call(`call_stan_`, options_vector, ll_fun, grad_fun)
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
