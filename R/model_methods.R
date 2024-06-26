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
    unconstrain_variables_impl(variables, stan_object@lower_bounds, stan_object@upper_bounds)
  }
)

#' @rdname unconstrain_draws-methods
#' @aliases unconstrain_draws,StanBase,StanBase-method
setMethod("unconstrain_draws", "StanBase",
  function(stan_object, draws) {
    if (is.null(draws)) {
      draws <- stan_object@draws
    }
    unconstrain_draws_impl(draws, stan_object@lower_bounds, stan_object@upper_bounds)
  }
)

#' @rdname unconstrain_draws-methods
#' @aliases unconstrain_draws,StanOptimize,StanOptimize-method
setMethod("unconstrain_draws", "StanOptimize",
  function(stan_object, draws) {
    if (is.null(draws)) {
      variables <- stan_object@estimates
      unconstrain_draws_impl(stan_object@estimates, stan_object@lower_bounds, stan_object@upper_bounds, match_format = FALSE)
    } else {
      unconstrain_draws_impl(draws, stan_object@lower_bounds, stan_object@upper_bounds)
    }
  }
)

#' @rdname constrain_variables-methods
#' @aliases constrain_variables,StanBase,StanBase-method
setMethod("constrain_variables", "StanBase",
  function(stan_object, unconstrained_variables) {
    constrain_variables_impl(unconstrained_variables, stan_object@lower_bounds, stan_object@upper_bounds)
  }
)


#' Calculate approximate leave-one-out cross-validation (LOO-CV) for a model.
#'
#' @aliases loo
#'
#' @param x A \code{StanBase} object.
#' @param pointwise_ll_fun Function that calculates the pointwise log-likelihood
#'   given a vector of parameter values.
#' @param additional_args List of additional arguments to be passed to
#'  \code{pointwise_ll_fun}.
#' @param moment_match (logical) Whether to use a
#'   [moment-matching][loo::loo_moment_match()] correction for problematic
#'   observations.
#' @param ... Additional arguments to be passed to \code{loo::loo()}.
#'
#' @importFrom loo loo relative_eff loo.matrix loo_moment_match.default
#' @export loo
#' @export
loo.StanBase <- function(x, pointwise_ll_fun, additional_args = list(),
                          moment_match = FALSE, ...) {
  par_cols <- grep("^par", colnames(x@draws))
  log_lik_draws <- t(apply(x@draws, 1, function(curr_draw) {
    do.call(pointwise_ll_fun, c(list(curr_draw[par_cols]), additional_args))
  }))
  reff <- loo::relative_eff(exp(log_lik_draws), chain_id = x@draws$.chain)

  if (moment_match) {
    suppressWarnings(loo_res <- loo::loo.matrix(log_lik_draws, r_eff = reff, ...))
    log_lik_i <- function(x, i, parameter_name = "log_lik", ...) {
      log_lik_draws[, i]
    }

    log_lik_i_upars <- function(x, upars, i, parameter_name = "log_lik", ...) {
      apply(upars, 1, function(up_i) {
        do.call(pointwise_ll_fun, c(list(constrain_variables(x, up_i)), additional_args))[i]
      })
    }

    loo::loo_moment_match.default(
      x = x,
      loo = loo_res,
      post_draws = function(x, ...) { posterior::subset_draws(x@draws, "pars") },
      log_lik_i = log_lik_i,
      unconstrain_pars = function(x, pars, ...) { posterior::as_draws_matrix(unconstrain_draws(x)) },
      log_prob_upars = function(x, upars, ...) { apply(upars, 1, function(up_i) { log_prob(x, up_i) }) },
      log_lik_i_upars = log_lik_i_upars,
      ...
    )
  } else {
    loo::loo.matrix(log_lik_draws, r_eff = reff, ...)
  }
}
