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

y <- rnorm(500, 10, 2)
ll <- function(v) { sum(dnorm(y, v[1], exp(v[2]), log = TRUE)) }

args <- c("sample",
          "num_chains=1",
          "data", paste0("file=",normalizePath("testdata3.json")),
          "num_threads=1")
call_stan(args, ll_fun = fn1, grad_fun = fn1)

parsed_raw <- parse_csv(normalizePath("output_test.csv"))
lapply(parsed_raw, )
devtools::load_all("~/Git/Dev/StanEstimators/")
setwd("~/Git/Dev/StanEstimators/")

y <- rnorm(500, 10, 2)
ll <- function(v, x) {
  sum(dnorm(x, v[1], exp(v[2]), log = TRUE))
}

make_wrapper <- function(fn, arg) {
  function(v) { fn(v, arg) }
}

t <- stan_optimize(ll, c(10, 2), y)
