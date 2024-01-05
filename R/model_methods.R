#' Calculate the log probability of the model given a vector of
#' unconstrained variables.
#'
#' @docType methods
#' @rdname log_prob-methods
#'
#' @param stan_object A \code{StanBase} object.
#' @param unconstrained_variables Vector of unconstrained variables.
#' @param jacobian Whether to include the Jacobian adjustment.
#'
#' @export
setGeneric(
  "log_prob",
  function(stan_object, unconstrained_variables, jacobian = TRUE) standardGeneric("log_prob")
)

#' Calculate the log probability and its gradient of the
#' model given a vector of unconstrained variables.
#'
#' @docType methods
#' @rdname grad_log_prob-methods
#'
#' @param stan_object A \code{StanBase} object.
#' @param unconstrained_variables Vector of unconstrained variables.
#' @param jacobian Whether to include the Jacobian adjustment.
#'
#' @export
setGeneric(
  "grad_log_prob",
  function(stan_object, unconstrained_variables, jacobian = TRUE) standardGeneric("grad_log_prob")
)

#' Calculate the log probability, its gradient, and its hessian matrix of the
#' model given a vector of unconstrained variables.
#'
#' @docType methods
#' @rdname hessian-methods
#'
#' @param stan_object A \code{StanBase} object.
#' @param unconstrained_variables Vector of unconstrained variables.
#' @param jacobian Whether to include the Jacobian adjustment.
#'
#' @export
setGeneric(
  "hessian",
  function(stan_object, unconstrained_variables, jacobian = TRUE) standardGeneric("hessian")
)

#' Unconstrain a vector of variables.
#'
#' @docType methods
#' @rdname unconstrain_variables-methods
#'
#' @param stan_object A \code{StanBase} object.
#' @param variables Vector of variables to be unconstrained.
#'
#' @export
setGeneric(
  "unconstrain_variables",
  function(stan_object, variables) standardGeneric("unconstrain_variables")
)

#' Unconstrain all parameter draws.
#'
#' @docType methods
#' @rdname unconstrain_draws-methods
#'
#' @param stan_object A \code{StanBase} object.
#' @param draws (optional) A `posterior::draws_*` object to be unconstrained
#'  (instead of the draws in the \code{StanBase} object).
#'
#' @export
setGeneric(
  "unconstrain_draws",
  function(stan_object, draws = NULL) standardGeneric("unconstrain_draws")
)

#' Constrain a vector of variables.
#'
#' @docType methods
#' @rdname constrain_variables-methods
#'
#' @param stan_object A \code{StanBase} object.
#' @param unconstrained_variables Vector of unconstrained variables.
#'
#' @export
setGeneric(
  "constrain_variables",
  function(stan_object, unconstrained_variables) standardGeneric("constrain_variables")
)

#' @rdname log_prob-methods
#' @aliases log_prob,StanBase,StanBase-method
setMethod("log_prob", "StanBase",
  function(stan_object, unconstrained_variables, jacobian) {
    log_prob_impl(stan_object@model_methods$model_pointer,
                  unconstrained_variables, jacobian)
  }
)

#' @rdname grad_log_prob-methods
#' @aliases grad_log_prob,StanBase,StanBase-method
setMethod("grad_log_prob", "StanBase",
  function(stan_object, unconstrained_variables, jacobian) {
    grad_log_prob_impl(stan_object@model_methods$model_pointer,
                        unconstrained_variables, jacobian)
  }
)

#' @rdname hessian-methods
#' @aliases hessian,StanBase,StanBase-method
setMethod("hessian", "StanBase",
  function(stan_object, unconstrained_variables, jacobian) {
    hessian_impl(stan_object@model_methods$model_pointer,
                        unconstrained_variables, jacobian)
  }
)

#' @rdname unconstrain_variables-methods
#' @aliases unconstrain_variables,StanBase,StanBase-method
setMethod("unconstrain_variables", "StanBase",
  function(stan_object, variables) {
    unconstrain_variables_impl(stan_object@model_methods$model_pointer,
                               variables)
  }
)

#' @rdname unconstrain_draws-methods
#' @aliases unconstrain_draws,StanBase,StanBase-method
setMethod("unconstrain_draws", "StanBase",
  function(stan_object, draws) {
    if (is.null(draws)) {
      draws <- stan_object@draws
    }
    unconstrain_draws_impl(stan_object@model_methods$model_pointer, draws)
  }
)

#' @rdname unconstrain_draws-methods
#' @aliases unconstrain_draws,StanOptimize,StanOptimize-method
setMethod("unconstrain_draws", "StanOptimize",
  function(stan_object, draws) {
    if (is.null(draws)) {
      variables <- stan_object@estimates
      unconstrain_draws_impl(stan_object@model_methods$model_pointer, stan_object@estimates, match_format = FALSE)
    } else {
      unconstrain_draws_impl(stan_object@model_methods$model_pointer, draws)
    }
  }
)

#' @rdname constrain_variables-methods
#' @aliases constrain_variables,StanBase,StanBase-method
setMethod("constrain_variables", "StanBase",
  function(stan_object, unconstrained_variables) {
    constrain_variables_impl(stan_object@model_methods$model_pointer,
                              unconstrained_variables)
  }
)
