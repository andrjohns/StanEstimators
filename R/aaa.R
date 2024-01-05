setOldClass("draws_df")

#' StanBase base class
#'
#' @name StanBase-class
#' @aliases StanBase
#' @docType class
NULL

setClass(
  "StanBase",
  slots = c(
    metadata = "list",
    lower_bounds = "numeric",
    upper_bounds = "numeric",
    model_methods = "list"
  ))

setClass(
  "StanLaplace",
  contains = "StanBase",
  slots = c(
    estimates = "data.frame",
    draws = "draws_df"
  )
)

setClass(
  "StanOptimize",
  contains = "StanBase",
  slots = c(
    estimates = "data.frame"
  )
)

setClass(
  "StanPathfinder",
  contains = "StanBase",
  slots = c(
    draws = "draws_df"
  )
)

setClass(
  "StanMCMC",
  contains = "StanBase",
  slots = c(
    adaptation = "list",
    timing = "list",
    diagnostics = "draws_df",
    draws = "draws_df"
  )
)

setClass(
  "StanVariational",
  contains = "StanBase",
  slots = c(
    estimates = "data.frame",
    draws = "draws_df"
  )
)

