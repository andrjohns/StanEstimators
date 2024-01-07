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

hessian_impl <- function(model_ptr, upars, jacobian = TRUE) {
  .Call(`hessian_`, model_ptr, upars, jacobian)
}

unconstrain_variables_impl <- function(model_ptr, variables) {
  .Call(`unconstrain_variables_`, model_ptr, variables)
}

unconstrain_draws_impl <- function(model_ptr, draws, match_format = TRUE) {
  draws_matrix <- posterior::as_draws_matrix(draws)
  par_cols <- grep("^par", colnames(draws_matrix))
  if (length(par_cols) == 0) {
    stop("No parameter columns found in draws object", call. = FALSE)
  }
  unconstrained_variables <- .Call(`unconstrain_draws_`, model_ptr, draws_matrix[, par_cols])
  draws_matrix[, par_cols] <- unconstrained_variables
  if (match_format) {
    match_draws_format(draws, draws_matrix)
  } else {
    draws_matrix
  }
}

constrain_variables_impl <- function(model_ptr, upars) {
  .Call(`constrain_variables_`, model_ptr, upars)
}

lub_constrain <- function(x, lb, ub) {
  .Call(`lub_constrain_`, x, lb, ub)
}

lub_free <- function(x, lb, ub) {
  .Call(`lub_free_`, x, lb, ub)
}
