#include <Rcpp.h>
#include <RcppEigen.h>
// Only the Eigen headers are needed, no need to include all Math headers
#define STAN_MATH_PRIM_HPP 1
#include <stan/io/stan_csv_reader.hpp>
#include <fstream>
#include <iostream>

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

RcppExport SEXP parse_csv_(SEXP filename_) {
  BEGIN_RCPP
  Rcpp::List rtn;

  std::ifstream ifstream = safe_open(Rcpp::as<std::string>(filename_));
  rtn.push_back(parse_metadata(ifstream), "metadata");

  std::vector<std::string> header;
  stan::io::stan_csv_reader::read_header(ifstream, header, nullptr);
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
