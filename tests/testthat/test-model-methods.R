set.seed(1234)
y <- rnorm(500, 10, 2)

loglik_fun <- function(v, x) {
  sum(dnorm(x, v[1], v[2], log = TRUE))
}

inits <- c(0, 5)
grad <- function(v, x) {
  inv_sigma <- 1 / v[2]
  y_scaled = (x - v[1]) * inv_sigma
  scaled_diff = inv_sigma * y_scaled
  c(sum(scaled_diff),
    sum(inv_sigma * (y_scaled*y_scaled) - inv_sigma)
  )
}

opt_fd <- stan_optimize(loglik_fun, inits, additional_args = list(y), lower = c(-Inf, 0),
                        seed = 1234, grad_fun = grad)

test_that("log_prob correctly computes log probability", {
  true_ll <- loglik_fun(inits, y)
  expect_equal(
    log_prob(opt_fd, unconstrained_variables = c(0, log(5)), jacobian = FALSE),
    true_ll)
  expect_equal(
    log_prob(opt_fd, unconstrained_variables = c(0, log(5)), jacobian = TRUE),
    true_ll + log(5))
})

test_that("grad_log_prob correctly computes gradient of log probability", {
  true_grad <- grad(inits, y) * c(1, 5) # Gradient adjustment for lb constrain
  attr(true_grad, "log_prob") <- loglik_fun(inits, y)
  expect_equal(
    grad_log_prob(opt_fd, unconstrained_variables = c(0, log(5)), jacobian = FALSE),
    true_grad)

  true_grad[2] <- true_grad[2] + 1
  attr(true_grad, "log_prob") <- loglik_fun(inits, y) + log(5)
  expect_equal(
    grad_log_prob(opt_fd, unconstrained_variables = c(0, log(5)), jacobian = TRUE),
    true_grad)
})

test_that("variable constraining works correctly", {
  expect_equal(
    unconstrain_variables(opt_fd, c(0, 5)),
    c(0, log(5))
  )
  expect_equal(
    constrain_variables(opt_fd, c(0, log(5))),
    c(0, 5)
  )
})
