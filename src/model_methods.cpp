#include <headers_to_ignore.hpp>
#include <stan/io/json/json_data.hpp>
#include <stan/model/model_base.hpp>
#include <stan/model/log_prob_propto.hpp>
#include <stan/model/log_prob_grad.hpp>
#include <stan/math/rev/functor/finite_diff_hessian_auto.hpp>
#include <Rcpp.h>
#include <RcppEigen.h>
#include <sstream>

std::shared_ptr<stan::io::var_context> var_context_from_string(std::string data_json_string) {
  std::istringstream data_stream(data_json_string);
  stan::json::json_data data_context(data_stream);
  return std::make_shared<stan::json::json_data>(data_context);
}

stan::model::model_base&
new_model(stan::io::var_context& data_context, unsigned int seed,
          std::ostream* msg_stream);

RcppExport SEXP make_model_pointer_(SEXP data_json_string_, SEXP seed_) {
  BEGIN_RCPP
  std::string data_json_string = Rcpp::as<std::string>(data_json_string_);

  Rcpp::XPtr<stan::model::model_base> ptr(
    &new_model(*var_context_from_string(data_json_string),
                Rcpp::as<unsigned int>(seed_), &Rcpp::Rcout)
  );
  return ptr;
  END_RCPP
}

RcppExport SEXP log_prob_(SEXP ext_model_ptr, SEXP upars_, SEXP jacobian_) {
  BEGIN_RCPP
  Rcpp::XPtr<stan::model::model_base> ptr(ext_model_ptr);
  double rtn;
  Eigen::VectorXd upars = Rcpp::as<Eigen::Map<Eigen::VectorXd>>(upars_);
  if (Rcpp::as<bool>(jacobian_)) {
    rtn = stan::model::log_prob_propto<true>(*ptr.get(), upars, &Rcpp::Rcout);
  } else {
    rtn = stan::model::log_prob_propto<false>(*ptr.get(), upars, &Rcpp::Rcout);
  }
  return Rcpp::wrap(rtn);
  END_RCPP
}

RcppExport SEXP grad_log_prob_(SEXP ext_model_ptr, SEXP upars_, SEXP jacobian_) {
  BEGIN_RCPP
  Rcpp::XPtr<stan::model::model_base> ptr(ext_model_ptr);
  std::vector<double> gradients;
  std::vector<int> params_i;
  std::vector<double> upars = Rcpp::as<std::vector<double>>(upars_);

  double lp;
  if (Rcpp::as<bool>(jacobian_)) {
    lp = stan::model::log_prob_grad<true, true>(
      *ptr.get(), upars, params_i, gradients);
  } else {
  lp = stan::model::log_prob_grad<true, false>(
      *ptr.get(), upars, params_i, gradients);
  }
  Rcpp::NumericVector grad_rtn = Rcpp::wrap(gradients);
  grad_rtn.attr("log_prob") = lp;
  return grad_rtn;
  END_RCPP
}

RcppExport SEXP hessian_(SEXP ext_model_ptr, SEXP upars_, SEXP jacobian_) {
  BEGIN_RCPP
  Rcpp::XPtr<stan::model::model_base> ptr(ext_model_ptr);
  Eigen::Map<Eigen::VectorXd> upars = Rcpp::as<Eigen::Map<Eigen::VectorXd>>(upars_);

  auto hessian_functor = [&](auto&& x) {
    if (Rcpp::as<bool>(jacobian_)) {
      return ptr->log_prob<true, true>(x, 0);
    } else {
      return ptr->log_prob<true, false>(x, 0);
    }
  };

  double log_prob;
  Eigen::VectorXd grad;
  Eigen::MatrixXd hessian;

  stan::math::internal::finite_diff_hessian_auto(hessian_functor, upars, log_prob, grad, hessian);

  return Rcpp::List::create(
    Rcpp::Named("log_prob") = log_prob,
    Rcpp::Named("grad_log_prob") = grad,
    Rcpp::Named("hessian") = hessian);
  END_RCPP
}

RcppExport SEXP unconstrain_variables_(SEXP ext_model_ptr, SEXP variables_) {
  BEGIN_RCPP
  Rcpp::XPtr<stan::model::model_base> ptr(ext_model_ptr);
  Eigen::Map<Eigen::VectorXd> variables = Rcpp::as<Eigen::Map<Eigen::VectorXd>>(variables_);
  Eigen::VectorXd unconstrained_variables;
  ptr->unconstrain_array(variables, unconstrained_variables, &Rcpp::Rcout);
  return Rcpp::wrap(unconstrained_variables);
  END_RCPP
}

RcppExport SEXP unconstrain_draws_(SEXP ext_model_ptr, SEXP draws_matrix_) {
  BEGIN_RCPP
  Rcpp::XPtr<stan::model::model_base> ptr(ext_model_ptr);
  Eigen::Map<Eigen::MatrixXd> variables = Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(draws_matrix_);
  Eigen::MatrixXd unconstrained_draws(variables.cols(), variables.rows());
  for (int i = 0; i < variables.rows(); i++) {
    Eigen::VectorXd unconstrained_variables;
    ptr->unconstrain_array(variables.transpose().col(i), unconstrained_variables, &Rcpp::Rcout);
    unconstrained_draws.col(i) = unconstrained_variables;
  }
  return Rcpp::wrap(unconstrained_draws.transpose());
  END_RCPP
}

RcppExport SEXP constrain_variables_(SEXP ext_model_ptr, SEXP upars_) {
  BEGIN_RCPP
  Rcpp::XPtr<stan::model::model_base> ptr(ext_model_ptr);
  std::vector<double> upars = Rcpp::as<std::vector<double>>(upars_);
  std::vector<int> params_i;
  std::vector<double> pars_constrained;
  // RNG only used for *_rng calls in generated_quantities, which we don't use
  boost::ecuyer1988 dummy_rng(0);
  ptr->write_array(dummy_rng, upars, params_i, pars_constrained, false, false);
  return Rcpp::wrap(pars_constrained);
  END_RCPP
}
