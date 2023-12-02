write_data <- function(Npars, finite_diff, bound_inds,
                       lower_bounds, upper_bounds, data_file) {
  dat_string <- paste(
    '{',
    '"Npars" : ', Npars, ',',
    '"finite_diff" : ', finite_diff, ',',
    '"Nbounds" : ', length(bound_inds), ',',
    '"bound_inds" : [', paste0(bound_inds, collapse = ','), '],',
    '"lower_bounds" : [', paste0(lower_bounds, collapse = ','), '],',
    '"upper_bounds" : [', paste0(upper_bounds, collapse = ','), ']',
    '}'
  )
  writeLines(dat_string, con = data_file)
  invisible(NULL)
}

bounds <- function(inds, lower = -Inf, upper = Inf) {
  if ((length(inds) > 1) && (length(lower) == 1)) {
    lower <- rep(lower, length(inds))
  }
  if ((length(inds) > 1) && (length(upper) == 1)) {
    upper <- rep(upper, length(inds))
  }
  list(
    constrain = "lub",
    indices = inds,
    lower_bounds = lower,
    upper_bounds = upper
  )
}
