stanc <- function(model_file = NULL, model_code = NULL) {
  if (is.null(model_file) && is.null(model_code)) {
    stop("Must provide either model_file or model_code", call. = FALSE)
  }
  if (!is.null(model_file) && !is.null(model_code)) {
    stop("Cannot provide both model_file and model_code", call. = FALSE)
  }
  if (!is.null(model_file)) {
    if (!file.exists(model_file)) {
      stop("File does not exist: ", model_file, call. = FALSE)
    }
    model_code <- readLines(model_file)
  }
  if (is.null(model_code)) {
    stop("Must provide either model_file or model_code", call. = FALSE)
  }
  stanc_ret <- stanc_ctx$call("stanc", "testing_model", model_code)

  if (length(stanc_ret$errors) > 0) {
    stop(paste(stanc_ret$errors, collapse = "\n"), call. = FALSE)
  }
  if (length(stanc_ret$warnings) > 0) {
    warning(paste(stanc_ret$warnings, collapse = "\n"), call. = FALSE)
  }
  stanc_ret
}
