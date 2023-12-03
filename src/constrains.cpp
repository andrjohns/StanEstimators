#include <stan/math/prim/fun/lub_constrain.hpp>
#include <stan/math/prim/fun/lub_free.hpp>
#include <Rcpp.h>
#include <RcppEigen.h>

RcppExport SEXP constrain_pars_(SEXP pars_, SEXP lower_, SEXP upper_) {
  Eigen::Map<Eigen::VectorXd> pars = Rcpp::as<Eigen::Map<Eigen::VectorXd>>(pars_);
  Eigen::Map<Eigen::VectorXd> lower = Rcpp::as<Eigen::Map<Eigen::VectorXd>>(lower_);
  Eigen::Map<Eigen::VectorXd> upper = Rcpp::as<Eigen::Map<Eigen::VectorXd>>(upper_);
  return Rcpp::wrap(stan::math::lub_constrain(pars, lower, upper));
}

RcppExport SEXP unconstrain_pars_(SEXP pars_, SEXP lower_, SEXP upper_) {
  Eigen::Map<Eigen::VectorXd> pars = Rcpp::as<Eigen::Map<Eigen::VectorXd>>(pars_);
  Eigen::Map<Eigen::VectorXd> lower = Rcpp::as<Eigen::Map<Eigen::VectorXd>>(lower_);
  Eigen::Map<Eigen::VectorXd> upper = Rcpp::as<Eigen::Map<Eigen::VectorXd>>(upper_);
  return Rcpp::wrap(stan::math::lub_free(pars, lower, upper));
}
