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

#' stan_versions
#' @return A named list with the Stan and Stan Math library versions
#' @export
stan_versions <- function() {
  .Call(`stan_versions_`)
}

make_model_pointer <- function(data_json_string, seed) {
  seed <- ifelse(is.null(seed), 0, seed)
  .Call(`make_model_pointer_`, data_json_string, seed)
}

log_prob_impl <- function(model_ptr, upars, jacobian = TRUE) {
  .Call(`log_prob_`, model_ptr, upars, jacobian)
}

grad_log_prob_impl <- function(model_ptr, upars, jacobian = TRUE) {
  .Call(`grad_log_prob_`, model_ptr, upars, jacobian)
}

unconstrain_variables_impl <- function(model_ptr, cons_json_string) {
  .Call(`unconstrain_variables_`, model_ptr, cons_json_string)
}

constrain_variables_impl <- function(model_ptr, upars) {
  .Call(`constrain_variables_`, model_ptr, upars)
}
