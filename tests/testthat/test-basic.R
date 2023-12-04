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

test_that("stan_sample runs", {
  expect_no_error(
    fit <- stan_sample(loglik_fun, inits, y, lower = c(-Inf, 0),
                        num_chains = 1, seed = 1234)
  )
  expect_no_error(
    fit <- stan_sample(loglik_fun, inits, y, grad_fun = grad,
                        lower = c(-Inf, 0),
                        num_chains = 1, seed = 1234)
  )
  expect_no_error(
    fit <- stan_sample(loglik_fun, inits, y, grad_fun = grad,
                        lower = c(-Inf, 0),
                        metric = "dense_e",
                        num_chains = 1, seed = 1234)
  )
})

test_that("stan_optimize runs", {
  expect_no_error(
    fit <- stan_optimize(loglik_fun, inits, y, lower = c(-Inf, 0),
                         seed = 1234)
  )
  expect_no_error(
    fit <- stan_optimize(loglik_fun, inits, y, grad_fun = grad,
                        lower = c(-Inf, 0), seed = 1234)
  )
  expect_no_error(
    fit <- stan_optimize(loglik_fun, inits, y, grad_fun = grad,
                        lower = c(-Inf, 0),
                        algorithm = "bfgs",
                        seed = 1234)
  )
})

test_that("stan_variational runs", {
  expect_no_error(
    fit <- stan_variational(loglik_fun, inits, y, lower = c(-Inf, 0),
                         seed = 1234)
  )
  expect_no_error(
    fit <- stan_variational(loglik_fun, inits, y, grad_fun = grad,
                        lower = c(-Inf, 0), seed = 1234)
  )
  expect_no_error(
    fit <- stan_variational(loglik_fun, inits, y, grad_fun = grad,
                        lower = c(-Inf, 0),
                        algorithm = "fullrank",
                        seed = 1234)
  )
})

test_that("stan_pathfinder runs", {
  expect_no_error(
    fit <- stan_pathfinder(loglik_fun, inits, y, lower = c(-Inf, 0),
                         seed = 1234)
  )
  expect_no_error(
    fit <- stan_pathfinder(loglik_fun, inits, y, grad_fun = grad,
                        lower = c(-Inf, 0), seed = 1234)
  )
})
