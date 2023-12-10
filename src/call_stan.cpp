#include <estimator/estimator_ext_header.hpp>
#include <estimator/estimator.hpp>
#include <Rcpp.h>

SEXP _cmdstan_main(SEXP options_vector);

RcppExport SEXP call_stan_(SEXP options_vector, SEXP ll_fun, SEXP grad_fun) {
  internal::ll_fun = Rcpp::Function(ll_fun);
  internal::grad_fun = Rcpp::Function(grad_fun);
  return _cmdstan_main(options_vector);
}
