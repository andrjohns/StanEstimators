write_data <- function(Npars, finite_diff, output_file) {
  dat_string <- paste0("{ \"Npars\": ", Npars,", \"finite_diff\": ", as.integer(finite_diff)," }")
  writeLines(dat_string, con = output_file)
}
