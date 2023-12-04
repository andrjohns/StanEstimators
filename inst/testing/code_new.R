standata2 <- list(
  Npars = 2,
  finite_diff = 1
)
cmdstanr::write_stan_json(standata2, file="testdata2.json")

write_data <- function(Npars, finite_diff, output_file) {
  dat_string <- paste0("{ \"Npars\": ", Npars,", \"finite_diff\": ", as.integer(finite_diff)," }")
  writeLines(dat_string, con = output_file)
}

write_data(2, 1, "testdata3.json")


args <- c("sample",
          "num_chains=1",
          "data", paste0("file=",normalizePath("testdata3.json")),
          "num_threads=1")
call_stan(args, ll_fun = fn1, grad_fun = fn1)

parsed_raw <- parse_csv(normalizePath("output_test.csv"))
lapply(parsed_raw, )
devtools::load_all("~/Git/Dev/StanEstimators/", recompile = TRUE)
setwd("~/Git/Dev/StanEstimators/")

y <- rnorm(500, 10, 2)
ll <- function(v, x) {
  sum(dnorm(x, v[1], exp(v[2]), log = TRUE))
}

test_wrapper <- function(fn, arg, ...) {
  fn1 <- function(v) { fn(v, ...) }
  fn1(arg)
}

test_wrapper(ll, c(10, 2), y)


y <- rnorm(500, 10, 2)
ll <- function(v, x) { sum(dnorm(x, v[1], v[2], log = TRUE)) }

grad_fun <- function(v, x) {
  mu <- v[1]
  sigma <- v[2]

  inv_sigma <- 1 / sigma
  y_scaled = (x - mu) * inv_sigma
  scaled_diff = inv_sigma * y_scaled
  c(
    sum(scaled_diff),
    sum(inv_sigma * (y_scaled*y_scaled) - inv_sigma)
  )
}

t <- stan_optimize(ll, c(10, 2), y, lower = c(-Inf, 0), refresh=10)
t2 <- stan_sample(ll, c(10, 2), y, lower = c(-Inf, 0), grad_fun = grad_fun)

ll2 <- function(v, x) { sum(dnorm(x, v[1], exp(v[2]), log = TRUE)) }
t2 <- stan_optimize(ll2, c(10, 2), y)

args <- c(
  "optimize",
  paste0("algorithm=", "lbfgs"),
  paste0("jacobian=", 0),
  paste0("iter=", 2000),
  paste0("save_iterations=", 0),
  "data",
  paste0("file=", "file=/var/folders/1d/rjvd0h_n0yq20j13h4gdfytw0000gn/T//Rtmph3NGzQ/fileef7cf8243b4.json"),
  "output",
  paste0("file=", "file=/var/folders/1d/rjvd0h_n0yq20j13h4gdfytw0000gn/T//Rtmph3NGzQ/fileef7c312ae660.csv")
)

t_dr <- t$draws
test2 <- apply(posterior::subset_draws(t_dr, "pars"), 1, function(pars) {
  numpars <- as.numeric(pars)
  sapply(y, function(y_val) { ll(numpars, y_val) })
})

pointwise_ll <- function(v, y) {
  dnorm(y, v[1], v[2], log = TRUE)
}

loglik <- t(apply(t2$draws, 1, function(est_row) {
  pointwise_ll(as.numeric(est_row[par_inds]), y)
}))

loo::loo(loglik, r_eff = loo::relative_eff(exp(loglik), chain_id = t2$draws$.chain))
