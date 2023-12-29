#include <Rcpp.h>
#include <cmdstan/command.hpp>

void store_functions(SEXP ll_fun, SEXP grad_fun);

RcppExport SEXP call_stan_(SEXP options_vector, SEXP ll_fun, SEXP grad_fun) {
  BEGIN_RCPP
  store_functions(ll_fun, grad_fun);
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
