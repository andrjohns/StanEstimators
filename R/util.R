write_data <- function(Npars, finite_diff, lower_bounds, upper_bounds, data_file) {
  no_bounds <- all(lower_bounds == -Inf) && all(upper_bounds == Inf)
  dat_string <- paste(
    '{',
    '"Npars" : ', Npars, ',',
    '"finite_diff" : ', finite_diff, ',',
    '"no_bounds" : ', as.integer(no_bounds), ',',
    '"lower_bounds" : [', paste0(lower_bounds, collapse = ','), '],',
    '"upper_bounds" : [', paste0(upper_bounds, collapse = ','), ']',
    '}'
  )
  writeLines(dat_string, con = data_file)
  invisible(NULL)
}

write_inits <- function(inits, init_filepath) {
  dat_string <- paste(
    '{',
    '"pars" : [', paste0(inits, collapse = ','), ']',
    '}'
  )
  writeLines(dat_string, con = init_filepath)
  invisible(NULL)
}

prepare_function <- function(fn, inits, ..., grad = FALSE) {
  fn_wrapper <- function(v) { fn(v, ...) }
  fn_type <- ifelse(isTRUE(grad), "Gradient", "Log-Likelihood")
  test_fn <- try(invisible(fn_wrapper(inits)), silent = TRUE)
  correct_length <- ifelse(isTRUE(grad), length(inits), 1)

  if (inherits(test_fn, "try-error")) {
    stop(fn_type, " function evaluated at initial values resulted in error: \n ",
         test_fn, call. = FALSE)
  } else if (any(!is.finite(test_fn))) {
    stop(fn_type, " function evaluated at initial values is not finite!",
         call. = FALSE)
  } else if (length(test_fn) != correct_length) {
    stop(fn_type, " function should have return of length ", correct_length,
          ", but return was length ", length(test_fn), "instead!", call. = FALSE)
  } else {
    fn_wrapper
  }
}
