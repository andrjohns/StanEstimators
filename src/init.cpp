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
SEXP constrain_pars_(SEXP pars_, SEXP lower_, SEXP upper_);
SEXP unconstrain_pars_(SEXP pars_, SEXP lower_, SEXP upper_);

#ifdef __cplusplus
}
#endif


#define CALLDEF(name, n)  {#name, (DL_FUNC) &name, n}

static const R_CallMethodDef CallEntries[] = {
  CALLDEF(call_stan_, 3),
  CALLDEF(parse_csv_, 1),
  CALLDEF(constrain_pars_, 3),
  CALLDEF(unconstrain_pars_, 3),
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
