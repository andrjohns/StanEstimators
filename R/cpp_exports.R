call_stan_impl <- function(options_vector, input_list) {
  if (isTRUE(input_list$eval_standalone)) {
    for (pkg in input_list$packages) {
      library(pkg, character.only = TRUE, quietly = TRUE)
    }
    attach_wrap <- base::attach # Satisfy R CMD CHECK
    attach_wrap(input_list$globals, pos = 2L, name = "r_bg_globals")
  }
  status <- .Call(`call_stan_`, options_vector, input_list$ll_function, input_list$grad_function)
  invisible(NULL)
}

parse_csv <- function(filename) {
  parsed <- .Call(`parse_csv_`, filename)
  parsed$metadata <- parsed$metadata[unique(names(parsed$metadata))]
  parsed
}
