write_data <- function(Npars, finite_diff, lower_bounds, upper_bounds, data_file) {
  dat_string <- paste(
    '{',
    '"Npars" : ', Npars, ',',
    '"finite_diff" : ', finite_diff, ',',
    '"lower_bounds" : [', paste0(lower_bounds, collapse = ','), '],',
    '"upper_bounds" : [', paste0(upper_bounds, collapse = ','), ']',
    '}'
  )
  writeLines(dat_string, con = data_file)
  invisible(NULL)
}
