loglik_fun <- function(v, x) {
  sum(dnorm(x, v[1], v[2], log = TRUE))
}

inits <- c(0, 5)

fit <- stan_sample(loglik_fun, inits, additional_args = list(y),lower=c(-Inf, 0),
                   num_chains = 1,
                        seed = 1234)