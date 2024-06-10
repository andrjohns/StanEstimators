#include <Rcpp.h>
#include <cstdint>
#include <cmdstan/command.hpp>
#include <estimator/estimator_ext_header.hpp>
#include <estimator/estimator.hpp>

RcppExport SEXP call_stan_(SEXP options_vector, SEXP ll_fun, SEXP grad_fun) {
  BEGIN_RCPP
  internal::ll_fun = Rcpp::Function(ll_fun);
  internal::grad_fun = Rcpp::Function(grad_fun);
  std::vector<std::string> options = Rcpp::as<std::vector<std::string>>(options_vector);
  int argc = 1 + options.size();
  char** argv = new char*[argc];

  // Read in the name
  std::string name = "\0";
  argv[0] = new char[name.size() + 1];
  strcpy(argv[0], name.c_str());

  if (options.size() > 0) {
    // internal counter
    int counter = 1;

    // Read List into vector of char arrays
    for (int i = 0; i < options.size(); ++i) {
      std::string val = std::string(options[i]);
      argv[counter] = new char[val.size() + 1];
      strcpy(argv[counter++], val.c_str());
    }
  }
  const char** argv2 = const_cast<const char**>(argv);
  return Rcpp::wrap(cmdstan::command(argc, argv2));
  END_RCPP
}

/**
 * Opens input stream for file.
 * Throws exception if stream cannot be opened.
 *
 * Copied from cmdstan/command_helper.hpp
 *
 * @param fname name of file which exists and has read perms.
 * @return input stream
 */
std::ifstream safe_open(const std::string fname) {
  std::ifstream stream(fname.c_str());
  if (fname != "" && (stream.rdstate() & std::ifstream::failbit)) {
    std::stringstream msg;
    msg << "Can't open specified file, \"" << fname << "\"" << std::endl;
    throw std::invalid_argument(msg.str());
  }
  return stream;
}

Rcpp::List parse_metadata(std::istream& in) {
  std::stringstream ss;
  std::string line;

  if (in.peek() != '#') {
    return {};
  }
  while (in.peek() == '#') {
    std::getline(in, line);
    ss << line << '\n';
  }
  ss.seekg(std::ios_base::beg);

  char comment;
  std::string lhs;

  std::string name;
  std::string value;
  Rcpp::List metadata;

  while (ss.good()) {
    ss >> comment;
    std::getline(ss, lhs);

    size_t equal = lhs.find("=");
    if (equal != std::string::npos) {
      name = lhs.substr(0, equal);
      boost::trim(name);
      value = lhs.substr(equal + 1, lhs.size());
      boost::trim(value);
      boost::replace_first(value, " (Default)", "");
    } else {
      if (lhs.compare(" data") == 0) {
        ss >> comment;
        std::getline(ss, lhs);

        size_t equal = lhs.find("=");
        if (equal != std::string::npos) {
          name = lhs.substr(0, equal);
          boost::trim(name);
          value = lhs.substr(equal + 2, lhs.size());
          boost::replace_first(value, " (Default)", "");
        }

        continue;
      }
    }
    metadata.push_back(value, name);
  }

  return metadata;
}

RcppExport SEXP parse_csv_(SEXP filename_, SEXP lower_, SEXP upper_) {
  BEGIN_RCPP
  Rcpp::List rtn;

  std::ifstream ifstream = safe_open(Rcpp::as<std::string>(filename_));
  rtn.push_back(parse_metadata(ifstream), "metadata");

  std::vector<std::string> header;
  stan::io::stan_csv_reader::read_header(ifstream, header, nullptr);
  // Get index of first element of header that starts with "pars"
  auto it = std::find_if(header.cbegin(), header.cend(), [](const std::string& s) {
    return s.find("pars") == 0;
  });
  // Convert iterator to index
  int index = std::distance(header.cbegin(), it);


  rtn.push_back(header, "header");


  bool read_adaptation = std::forward<Rcpp::List>(rtn[0]).containsElementNamed("engaged");
  if (read_adaptation) {
    stan::io::stan_csv_adaptation adaptation;
    stan::io::stan_csv_reader::read_adaptation(ifstream, adaptation, nullptr);
    rtn.push_back(Rcpp::List::create(
      Rcpp::Named("step_size") = adaptation.step_size,
      Rcpp::Named("metric") = adaptation.metric
    ), "adaptation");
  }
  Eigen::MatrixXd samples;
  stan::io::stan_csv_timing timing;
  stan::io::stan_csv_reader::read_samples(ifstream, samples, timing, nullptr);
  Eigen::RowVectorXd lower = Rcpp::as<Eigen::RowVectorXd>(lower_);
  Eigen::RowVectorXd upper = Rcpp::as<Eigen::RowVectorXd>(upper_);

  for (int i = index; i < (index + lower.size()); ++i) {
    samples.col(i) = stan::math::lub_constrain(samples.col(i), lower[i - index], upper[i - index]);
  }

  rtn.push_back(samples, "samples");
  Rcpp::colnames(std::forward<Rcpp::NumericMatrix>(rtn["samples"])) = Rcpp::wrap(header);

  // If there is no timing information, then both elements remain at their default values of 0.
  bool rtn_timing = timing.warmup != 0 || timing.sampling != 0;
  if (rtn_timing) {
    rtn.push_back(Rcpp::List::create(
      Rcpp::Named("warmup") = timing.warmup,
      Rcpp::Named("sampling") = timing.sampling
    ), "timing");
  }

  ifstream.close();

  return rtn;
  END_RCPP
}

std::shared_ptr<stan::io::var_context> var_context_from_string(std::string data_json_string) {
  std::istringstream data_stream(data_json_string);
  stan::json::json_data data_context(data_stream);
  return std::make_shared<stan::json::json_data>(data_context);
}

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
  stan::rng_t dummy_rng(0);
  ptr->write_array(dummy_rng, upars, params_i, pars_constrained, false, false);
  return Rcpp::wrap(pars_constrained);
  END_RCPP
}

RcppExport SEXP lub_constrain_(SEXP y_, SEXP lb_, SEXP ub_) {
  using VectorMap = Eigen::Map<Eigen::VectorXd>;
  BEGIN_RCPP
  VectorMap y = Rcpp::as<VectorMap>(y_);
  VectorMap lb = Rcpp::as<VectorMap>(lb_);
  VectorMap ub = Rcpp::as<VectorMap>(ub_);
  return Rcpp::wrap(stan::math::lub_constrain(y, lb, ub));
  END_RCPP
}

RcppExport SEXP lub_free_(SEXP y_, SEXP lb_, SEXP ub_) {
  using VectorMap = Eigen::Map<Eigen::VectorXd>;
  BEGIN_RCPP
  VectorMap y = Rcpp::as<VectorMap>(y_);
  VectorMap lb = Rcpp::as<VectorMap>(lb_);
  VectorMap ub = Rcpp::as<VectorMap>(ub_);
  return Rcpp::wrap(stan::math::lub_free(y, lb, ub));
  END_RCPP
}
