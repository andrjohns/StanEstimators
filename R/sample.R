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

loo <- function(object, ...) {
  UseMethod("loo")
}

#' @export
setMethod("summary", "StanMCMC", function(object, ...) {
  posterior::summarise_draws(object@draws)
})

#' @export
setMethod("loo", "StanMCMC", function(object, pointwise_ll_fun, data, moment_match = FALSE, ...) {
  par_inds <- grep("pars", colnames(object@draws))
  loglik <- t(apply(object@draws, 1, function(est_row) {
    apply(data, 1, function(data_row) {
      pointwise_ll_fun(as.numeric(est_row[par_inds]), data_row)
    })
  }))
  loo::loo(loglik,
            r_eff = loo::relative_eff(exp(loglik), chain_id = object@draws$.chain),
            ...)
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
                        num_chains = 4, num_threads = 1, num_samples = 1000, num_warmup = 1000,
                        save_warmup = FALSE, thin = 1, output_dir = tempdir(),
                        control = list()) {
  fn1 <- prepare_function(fn, par_inits, ..., grad = FALSE)
  if (!is.null(grad_fun)) {
    gr1 <- prepare_function(grad_fun, par_inits, ..., grad = TRUE)
  } else {
    gr1 <- fn1
  }

  nPars <- length(par_inits)
  finite_diff <- as.integer(is.null(grad_fun))
  if ((length(par_inits) > 1) && (length(lower) == 1)) {
    lower <- rep(lower, length(par_inits))
  }
  if ((length(par_inits) > 1) && (length(upper) == 1)) {
    upper <- rep(upper, length(par_inits))
  }

  data_file <- tempfile(fileext = ".json", tmpdir = output_dir)
  init_file <- tempfile(fileext = ".json", tmpdir = output_dir)
  output_file_base <- tempfile(tmpdir = output_dir)
  output_file <- paste0(output_file_base, ".csv")
  write_data(nPars, finite_diff, lower, upper, data_file)
  write_inits(par_inits, init_file)
  args <- c(
    "sample",
    paste0("num_chains=", num_chains),
    paste0("num_samples=", num_samples),
    paste0("num_warmup=", num_warmup),
    paste0("save_warmup=", as.integer(save_warmup)),
    paste0("thin=", as.integer(thin)),
    "data",
    paste0("file=", data_file),
    paste0("init=", init_file),
    "output",
    paste0("file=", output_file),
    paste0("num_threads=", num_threads)
  )

  call <- call_stan(args, ll_fun = fn1, grad_fun = gr1)
  if (num_chains > 1) {
    output_files <- paste0(output_file_base, paste0("_", 1:num_chains, ".csv"))
  } else {
    output_files <- paste0(output_file_base, ".csv")
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

  new("StanMCMC",
    metadata = metadata,
    adaptation = adaptation,
    timing = timing,
    diagnostics = posterior::subset_draws(draws, variable = diagnostic_vars),
    draws = posterior::subset_draws(draws, variable = par_vars),
    log_prob = fn1,
    lower_bounds = lower,
    upper_bounds = upper
  )
}
