.onLoad <- function(libname, pkgname) {
  assign("stanc_ctx", QuickJSR::JSContext$new(), envir = topenv())
  stanc_js <- system.file("stanc.js", package = "StanEstimators", mustWork = TRUE)
  stanc_ctx$source(stanc_js)
}
