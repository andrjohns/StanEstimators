#include <stan/io/stan_csv_reader.hpp>
#include <Rcpp.h>
#include <RcppEigen.h>

Rcpp::List csv_to_r(const stan::io::stan_csv_metadata& metadata) {
  return Rcpp::List::create(
    Rcpp::Named("stan_version_major") = metadata.stan_version_major,
    Rcpp::Named("stan_version_minor") = metadata.stan_version_minor,
    Rcpp::Named("stan_version_patch") = metadata.stan_version_patch,
    Rcpp::Named("model") = metadata.model,
    Rcpp::Named("data") = metadata.data,
    Rcpp::Named("init") = metadata.init,
    Rcpp::Named("chain_id") = metadata.chain_id,
    Rcpp::Named("seed") = metadata.seed,
    Rcpp::Named("random_seed") = metadata.random_seed,
    Rcpp::Named("num_samples") = metadata.num_samples,
    Rcpp::Named("num_warmup") = metadata.num_warmup,
    Rcpp::Named("save_warmup") = metadata.save_warmup,
    Rcpp::Named("thin") = metadata.thin,
    Rcpp::Named("append_samples") = metadata.append_samples,
    Rcpp::Named("algorithm") = metadata.algorithm,
    Rcpp::Named("engine") = metadata.engine,
    Rcpp::Named("max_depth") = metadata.max_depth
  );
}

Rcpp::List csv_to_r(const stan::io::stan_csv_adaptation& adaptation) {
  return Rcpp::List::create(
    Rcpp::Named("step_size") = adaptation.step_size,
    Rcpp::Named("metric") = adaptation.metric
  );
}

Rcpp::List csv_to_r(const stan::io::stan_csv_timing& timing) {
  return Rcpp::List::create(
    Rcpp::Named("warmup") = timing.warmup,
    Rcpp::Named("sampling") = timing.sampling
  );
}

Rcpp::List csv_to_r(const stan::io::stan_csv& csv) {
  return Rcpp::List::create(
    Rcpp::Named("metadata") = csv_to_r(csv.metadata),
    Rcpp::Named("header") = csv.header,
    Rcpp::Named("adaptation") = csv_to_r(csv.adaptation),
    Rcpp::Named("samples") = csv.samples,
    Rcpp::Named("timing") = csv_to_r(csv.timing)
  );
}

RcppExport SEXP parse_csv_(SEXP filename_) {
  BEGIN_RCPP
  std::string filename = Rcpp::as<std::string>(filename_);
  std::ifstream ifstream;
  ifstream.open(filename);
  stan::io::stan_csv csv_parsed = stan::io::stan_csv_reader::parse(ifstream, nullptr);
  ifstream.close();
  return Rcpp::wrap(csv_to_r(csv_parsed));
  END_RCPP
}
