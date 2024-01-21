#include <Rcpp.h>

void store_functions(SEXP ll_fun, SEXP grad_fun);

extern "C"  {
  SEXP call_stan_(SEXP options_vector);
}

RcppExport SEXP call_stan_rfunction_(SEXP options_vector, SEXP ll_fun, SEXP grad_fun) {
  BEGIN_RCPP
  store_functions(ll_fun, grad_fun);
  return call_stan_(options_vector);
  END_RCPP
}
