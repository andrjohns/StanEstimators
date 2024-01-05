#include <Rcpp.h>
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <R_ext/Visibility.h>
#include <Rversion.h>

using namespace Rcpp;

#ifdef __cplusplus
extern "C"  {
#endif

SEXP call_stan_(SEXP options_vector, SEXP ll_fun, SEXP grad_fun);
SEXP parse_csv_(SEXP filename_);
SEXP stan_versions_();
SEXP make_model_pointer_(SEXP data_json_string_, SEXP seed_);
SEXP log_prob_(SEXP ext_model_ptr, SEXP upars_, SEXP jacobian_);
SEXP grad_log_prob_(SEXP ext_model_ptr, SEXP upars_, SEXP jacobian_);
SEXP hessian_(SEXP ext_model_ptr, SEXP upars_, SEXP jacobian_);
SEXP unconstrain_variables_(SEXP ext_model_ptr, SEXP cons_json_string_);
SEXP unconstrain_draws_(SEXP ext_model_ptr, SEXP draws_matrix_);
SEXP constrain_variables_(SEXP ext_model_ptr, SEXP upars_);

#ifdef __cplusplus
}
#endif


#define CALLDEF(name, n)  {#name, (DL_FUNC) &name, n}

static const R_CallMethodDef CallEntries[] = {
  CALLDEF(call_stan_, 3),
  CALLDEF(parse_csv_, 1),
  CALLDEF(stan_versions_, 0),
  CALLDEF(make_model_pointer_, 2),
  CALLDEF(log_prob_, 3),
  CALLDEF(grad_log_prob_, 3),
  CALLDEF(hessian_, 3),
  CALLDEF(unconstrain_variables_, 2),
  CALLDEF(unconstrain_draws_, 2),
  CALLDEF(constrain_variables_, 2),
  {NULL, NULL, 0}
};

#ifdef __cplusplus
extern "C"  {
#endif
void attribute_visible R_init_StanEstimators(DllInfo *dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
#ifdef __cplusplus
}
#endif
