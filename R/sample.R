stan_sample <- function(fn, par_inits, ..., algorithm = "hmc", engine = "nuts",
                        num_chains = 4, num_samples = 1000, num_warmup = 1000,
                        save_warmup = FALSE, thin = 1, output_dir = tempdir(),
                        control = list()) {
  fn1 <- function(v) { fn(v, ...) }

  nPars <- length(par_inits)
  finite_diff <- 1
  data_file <- tempfile(fileext = ".json", tmpdir = output_dir)
  output_file_base <- tempfile(tmpdir = output_dir)
  output_file <- paste0(output_file_base, ".csv")
  write_data(nPars, finite_diff, data_file)
  args <- c(
    "sample",
    paste0("num_chains=", num_chains),
    paste0("num_samples=", num_samples),
    paste0("num_warmup=", num_warmup),
    paste0("save_warmup=", as.integer(save_warmup)),
    paste0("thin=", as.integer(thin)),
    "data",
    paste0("file=", data_file),
    "output",
    paste0("file=", output_file)
  )

  call <- call_stan(args, ll_fun = fn1, grad_fun = fn1)
  output_files <- paste0(output_file_base, paste0("_", 1:num_chains, ".csv"))
  all_samples <- lapply(output_files, function(filepath) {
    parse_csv(filepath)
  })
  draw_names <- all_samples[[1]]$header
  all_draws <- lapply(all_samples, function(chain) {
    setNames(data.frame(chain$samples), chain$header)
  })
  metadata <- all_samples[[1]]$metadata
  adaptation <- lapply(all_samples, function(chain) { chain$adaptation })
  timing <- lapply(all_samples, function(chain) { chain$timing })
  draws <- lapply(all_samples, function(chain) {
    setNames(data.frame(chain$samples), chain$header)
  })
  diagnostic_vars <- c("accept_stat__", "stepsize__", "treedepth__", "n_leapfrog__", "divergent__", "energy__")
  par_vars <- draw_names[!(draw_names %in% diagnostic_vars)]
  draws <- posterior::as_draws_df(do.call(rbind.data.frame, draws))

  list(
    metadata = metadata,
    adaptation = adaptation,
    timing = timing,
    diagnostics = posterior::subset_draws(draws, variable = diagnostic_vars),
    draws = posterior::subset_draws(draws, variable = par_vars)
  )
}
