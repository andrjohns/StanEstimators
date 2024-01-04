#include <stan/math/version.hpp>
#include <stan/version.hpp>
#include <Rcpp.h>

RcppExport SEXP stan_versions_() {
  BEGIN_RCPP
  std::string math_version =
    stan::math::MAJOR_VERSION + "." +
    stan::math::MINOR_VERSION + "." +
    stan::math::PATCH_VERSION;

  std::string stan_version =
    stan::MAJOR_VERSION + "." +
    stan::MINOR_VERSION + "." +
    stan::PATCH_VERSION;

  return Rcpp::List::create(
    Rcpp::Named("Math") = math_version,
    Rcpp::Named("Stan") = stan_version
  );
  END_RCPP
}
