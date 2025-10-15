set.seed(1234)
y <- rnorm(500, 10, 2)



inits <- c(0, 5)


test_that("RTMB with explicit data - serial", {
  loglik_fun <- function(pars, x) {
    sum(dnorm(x, pars[1], pars[2], log = TRUE))
  }

  samp_fd <- stan_sample(loglik_fun, n_pars = 2, grad_fun = "RTMB",
                         additional_args = list(y), lower = c(-Inf, 0),
                         num_chains = 1, seed = 1234)
  expect_equal(mean(samp_fd@draws$`pars[1]`), 10.00, tolerance = 0.1)
  expect_equal(mean(samp_fd@draws$`pars[2]`), 2.07, tolerance = 0.1)

  samp_fd <- stan_sample(loglik_fun, inits, grad_fun = "RTMB",
                         additional_args = list(y), lower = c(-Inf, 0),
                         num_chains = 1, seed = 1234)
  expect_equal(mean(samp_fd@draws$`pars[1]`), 10.00, tolerance = 0.1)
  expect_equal(mean(samp_fd@draws$`pars[2]`), 2.07, tolerance = 0.1)

  samp_fd <- stan_sample(loglik_fun, inits, grad_fun = "RTMB",
                         additional_args = list(y), lower = c(-Inf, 0),
                         metric = "dense_e", num_chains = 1, seed = 1234)
  expect_equal(mean(samp_fd@draws$`pars[1]`), 10.00, tolerance = 0.1)
  expect_equal(mean(samp_fd@draws$`pars[2]`), 2.07, tolerance = 0.1)
})

test_that("RTMB with explicit data - parallel", {
  loglik_fun <- function(pars, x) {
    sum(dnorm(x, pars[1], pars[2], log = TRUE))
  }

  samp_fd <- stan_sample(loglik_fun, n_pars = 2, grad_fun = "RTMB",
                         additional_args = list(y), lower = c(-Inf, 0),
                         num_chains = 2, parallel_chains = 2, seed = 1234)
  expect_equal(mean(samp_fd@draws$`pars[1]`), 10.00, tolerance = 0.1)
  expect_equal(mean(samp_fd@draws$`pars[2]`), 2.07, tolerance = 0.1)

  samp_fd <- stan_sample(loglik_fun, inits, grad_fun = "RTMB",
                         additional_args = list(y), lower = c(-Inf, 0),
                         num_chains = 2, parallel_chains = 2, seed = 1234)
  expect_equal(mean(samp_fd@draws$`pars[1]`), 10.00, tolerance = 0.1)
  expect_equal(mean(samp_fd@draws$`pars[2]`), 2.07, tolerance = 0.1)

  samp_fd <- stan_sample(loglik_fun, inits, grad_fun = "RTMB",
                         additional_args = list(y), lower = c(-Inf, 0),
                         metric = "dense_e", num_chains = 2, parallel_chains = 2,
                         seed = 1234)
  expect_equal(mean(samp_fd@draws$`pars[1]`), 10.00, tolerance = 0.1)
  expect_equal(mean(samp_fd@draws$`pars[2]`), 2.07, tolerance = 0.1)
})

test_that("RTMB with captured data - serial", {
  # Check with implicit additional args
  loglik_fun <- function(pars) {
    sum(dnorm(y, pars[1], pars[2], log = TRUE))
  }

  samp_fd <- stan_sample(loglik_fun, n_pars = 2, grad_fun = "RTMB",
                         lower = c(-Inf, 0),
                         num_chains = 1, seed = 1234)
  expect_equal(mean(samp_fd@draws$`pars[1]`), 10.00, tolerance = 0.1)
  expect_equal(mean(samp_fd@draws$`pars[2]`), 2.07, tolerance = 0.1)

  samp_fd <- stan_sample(loglik_fun, inits, grad_fun = "RTMB",
                         lower = c(-Inf, 0),
                         num_chains = 1, seed = 1234)
  expect_equal(mean(samp_fd@draws$`pars[1]`), 10.00, tolerance = 0.1)
  expect_equal(mean(samp_fd@draws$`pars[2]`), 2.07, tolerance = 0.1)

  samp_fd <- stan_sample(loglik_fun, inits, grad_fun = "RTMB",
                         lower = c(-Inf, 0),
                         metric = "dense_e", num_chains = 1,
                         seed = 1234)
  expect_equal(mean(samp_fd@draws$`pars[1]`), 10.00, tolerance = 0.1)
  expect_equal(mean(samp_fd@draws$`pars[2]`), 2.07, tolerance = 0.1)
})

test_that("RTMB with captured data - parallel", {
  # Check with implicit additional args
  loglik_fun <- function(pars) {
    sum(dnorm(y, pars[1], pars[2], log = TRUE))
  }

  samp_fd <- stan_sample(loglik_fun, n_pars = 2, grad_fun = "RTMB",
                         lower = c(-Inf, 0),
                         num_chains = 2, parallel_chains = 2, seed = 1234)
  expect_equal(mean(samp_fd@draws$`pars[1]`), 10.00, tolerance = 0.1)
  expect_equal(mean(samp_fd@draws$`pars[2]`), 2.07, tolerance = 0.1)

  samp_fd <- stan_sample(loglik_fun, inits, grad_fun = "RTMB",
                         lower = c(-Inf, 0),
                         num_chains = 2, parallel_chains = 2, seed = 1234)
  expect_equal(mean(samp_fd@draws$`pars[1]`), 10.00, tolerance = 0.1)
  expect_equal(mean(samp_fd@draws$`pars[2]`), 2.07, tolerance = 0.1)

  samp_fd <- stan_sample(loglik_fun, inits, grad_fun = "RTMB",
                         lower = c(-Inf, 0),
                         metric = "dense_e", num_chains = 2, parallel_chains = 2,
                         seed = 1234)
  expect_equal(mean(samp_fd@draws$`pars[1]`), 10.00, tolerance = 0.1)
  expect_equal(mean(samp_fd@draws$`pars[2]`), 2.07, tolerance = 0.1)
})


test_that("RTMB works with data argument - standalone", {
  loglik_fun <- function(pars, x) {
    sum(dnorm(x, pars[1], pars[2], log = TRUE))
  }

  samp_fd <- stan_sample(loglik_fun, n_pars = 2, grad_fun = "RTMB",
                         additional_args = list(y), lower = c(-Inf, 0),
                         eval_standalone = TRUE,
                         num_chains = 1, seed = 1234)
  expect_equal(mean(samp_fd@draws$`pars[1]`), 10.00, tolerance = 0.1)
  expect_equal(mean(samp_fd@draws$`pars[2]`), 2.07, tolerance = 0.1)

  samp_fd <- stan_sample(loglik_fun, inits, grad_fun = "RTMB",
                         additional_args = list(y), lower = c(-Inf, 0),
                         eval_standalone = TRUE,
                         num_chains = 1, seed = 1234)
  expect_equal(mean(samp_fd@draws$`pars[1]`), 10.00, tolerance = 0.1)
  expect_equal(mean(samp_fd@draws$`pars[2]`), 2.07, tolerance = 0.1)

  samp_fd <- stan_sample(loglik_fun, inits, grad_fun = "RTMB",
                         additional_args = list(y), lower = c(-Inf, 0),
                         eval_standalone = TRUE,
                         metric = "dense_e", num_chains = 1, seed = 1234)
  expect_equal(mean(samp_fd@draws$`pars[1]`), 10.00, tolerance = 0.1)
  expect_equal(mean(samp_fd@draws$`pars[2]`), 2.07, tolerance = 0.1)
})
