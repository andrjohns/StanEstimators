#include <Rcpp.h>
#include <cmdstan/command.hpp>

SEXP _cmdstan_main(SEXP options_vector) {
  int num_threads = stan::math::internal::get_num_threads();
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
  try {
    tbb::task_arena limited(num_threads);
    tbb::task_group tg;
    int err_code;
    limited.execute([&]{
      tg.run([&]{
        err_code = cmdstan::command(argc, argv2);
      });
    });
    limited.execute([&]{ tg.wait(); });

    if (err_code == 0)
      return Rcpp::wrap(1);
    else
      return Rcpp::wrap(0);
  } catch (const std::exception& e) {
    Rcpp::Rcerr << e.what() << std::endl;
    return Rcpp::wrap(0);
  } catch (...) {
    Rcpp::Rcerr << "Unknown exception thrown!" << std::endl;
    return Rcpp::wrap(0);
  }
}
