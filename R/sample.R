setOldClass("draws_df")
setClass("StanMCMC",
  slots = c(
    metadata = "list",
    adaptation = "list",
    timing = "list",
    diagnostics = "draws_df",
    draws = "draws_df",
    log_prob = "function",
    lower_bounds = "numeric",
    upper_bounds = "numeric"
  )
)

#' @export
setMethod("summary", "StanMCMC", function(object, ...) {
  posterior::summarise_draws(object@draws)
})

#' stan_sample
#'
#' Estimate parameters using Stan's sampling algorithms
#'
#' @param fn
#' @param par_inits
#' @param ...
#' @param algorithm
#' @param engine
#' @param num_chains
#' @param num_samples
#' @param num_warmup
#' @param save_warmup
#' @param thin
#' @param output_dir
#' @param control
#' @return
#' @export
stan_sample <- function(fn, par_inits, ..., algorithm = "hmc", engine = "nuts",
                          grad_fun = NULL, lower = -Inf, upper = Inf,
                          seed = NULL,
                          refresh = NULL,
                          output_dir = NULL,
                          output_basename = NULL,
                          sig_figs = NULL,
                          num_chains = 4,
                          num_samples = 1000,
                          num_warmup = 1000,
                          save_warmup = NULL,
                          thin = NULL,
                          adapt_engaged = NULL,
                          adapt_gamma = NULL,
                          adapt_delta = NULL,
                          adapt_kappa = NULL,
                          adapt_t0 = NULL,
                          adapt_init_buffer = NULL,
                          adapt_term_buffer = NULL,
                          adapt_window = NULL,
                          int_time = NULL,
                          max_treedepth = NULL,
                          metric = NULL,
                          metric_file = NULL,
                          stepsize = NULL,
                          stepsize_jitter = NULL) {
  inputs <- prepare_inputs(fn, par_inits, list(...), grad_fun, lower, upper,
                            output_dir, output_basename)
  method_args <- list(
    algorithm = algorithm,
    algorithm_args = list(
      engine = engine,
      engine_args = list(int_time = int_time, max_depth = max_treedepth),
      metric = metric,
      metric_file = metric_file,
      stepsize = stepsize,
      stepsize_jitter = stepsize_jitter
    ),
    adapt = list(
      engaged = adapt_engaged,
      gamma = adapt_gamma,
      delta = adapt_delta,
      kappa = adapt_kappa,
      t0 = adapt_t0,
      init_buffer = adapt_init_buffer,
      term_buffer = adapt_term_buffer,
      window = adapt_window
    ),
    num_samples = num_samples,
    num_warmup = num_warmup,
    save_warmup = save_warmup,
    thin = thin,
    num_chains = num_chains
  )

  output <- list(
    file = inputs$output_filepath,
    diagnostic_file = NULL,
    refresh = refresh,
    sig_figs = sig_figs,
    profile_file = NULL
  )
  args <- build_stan_call(method = "sample",
                          method_args = method_args,
                          data_file = inputs$data_filepath,
                          init = inputs$init_filepath,
                          seed = seed,
                          output_args = output,
                          num_threads = NULL)

  call_stan(args, ll_fun = inputs$ll_function, grad_fun = inputs$grad_function)

  if (num_chains > 1) {
    output_files <- paste0(inputs$output_basename, paste0("_", 1:num_chains, ".csv"))
  } else {
    output_files <- paste0(inputs$output_basename, ".csv")
  }
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

  methods::new("StanMCMC",
    metadata = metadata,
    adaptation = adaptation,
    timing = timing,
    diagnostics = posterior::subset_draws(draws, variable = diagnostic_vars),
    draws = posterior::subset_draws(draws, variable = par_vars),
    log_prob = inputs$ll_function,
    lower_bounds = lower,
    upper_bounds = upper
  )
}
