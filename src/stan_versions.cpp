#include <stan/math/version.hpp>
#include <stan/version.hpp>
#include <cmdstan/version.hpp>
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

  std::string cmdstan_version =
    cmdstan::MAJOR_VERSION + "." +
    cmdstan::MINOR_VERSION + "." +
    cmdstan::PATCH_VERSION;

  return Rcpp::List::create(
    Rcpp::Named("Math") = math_version,
    Rcpp::Named("Stan") = stan_version,
    Rcpp::Named("CmdStan") = cmdstan_version
  );
  END_RCPP
}
