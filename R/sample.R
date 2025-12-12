#' Summary method for objects of class \code{StanMCMC}.
#'
#' @docType methods
#' @name summary-StanMCMC
#' @rdname summary-StanMCMC
#' @aliases summary-StanMCMC summary,StanMCMC-method
#'
#' @param object A \code{StanMCMC} object.
#' @param ... Additional arguments, currently unused.
#'
#' @export
setMethod("summary", "StanMCMC", function(object, ...) {
  posterior::summarise_draws(object@draws)
})

#' stan_sample
#'
#' Estimate parameters using Stan's sampling algorithms
#'
#' @param fn Function to estimate parameters for
#' @param par_inits Initial values for parameters. This can either be a numeric vector
#'  of initial values (which will be used for all chains), a list of numeric vectors (of length
#'  equal to the number of chains), a function taking a single argument (the chain ID) and
#'  returning a numeric vector of initial values, or NULL (in which case Stan will
#'  generate initial values automatically).
#'  (must be specified if `n_pars` is NULL)
#' @param n_pars Number of parameters to estimate
#'  (must be specified if `par_inits` is NULL)
#' @param additional_args List of additional arguments to pass to the function
#' @param algorithm (string) The sampling algorithm. One of `"hmc"`
#'    or `"fixed_param"`.
#' @param engine (string) The `HMC` engine to use, one of `"nuts"` or `"static"`
#' @param grad_fun Either:
#'   - `NULL` for finite-differences (default)
#'   - A function for calculating gradients w.r.t. each parameter
#'   - "RTMB" to use the RTMB package for automatic differentiation
#' @param lower Lower bound constraint(s) for parameters
#' @param upper Upper bound constraint(s) for parameters
#' @param eval_standalone (logical) Whether to evaluate the function in a
#'    separate R session. Defaults to \code{(parallel_chains > 1)}.
#'    Must be `TRUE` if `parallel_chains > 1`.
#' @param globals (optional) a logical, a character vector, or a named list
#'    to control how globals are handled when evaluating functions in a
#'    separate R session. Ignored if `eval_standalone` = `FALSE`.
#'    For details, see section 'Globals used by future expressions'
#'    in the help for [future::future()].
#' @param packages (optional) a character vector specifying packages
#'    to be attached in the \R environment evaluating the function.
#'    Ignored if `eval_standalone` = `FALSE`.
#' @param seed Random seed
#' @param refresh Number of iterations for printing
#' @param quiet (logical) Whether to suppress Stan's output
#' @param output_dir Directory to store outputs
#' @param output_basename Basename to use for output files
#' @param sig_figs Number of significant digits to use for printing
#' @param num_chains (positive integer) The number of Markov chains to run. The
#'   default is 4.
#' @param parallel_chains (positive integer) The number of chains to run in
#'    parallel, the default is 1.
#' @param num_samples (positive integer) The number of post-warmup iterations
#'   to run per chain.
#' @param num_warmup (positive integer) The number of warmup iterations to run
#'   per chain.
#' @param save_warmup (logical) Should warmup iterations be saved? The default
#'   is `FALSE`.
#' @param thin (positive integer) The period between saved samples. This should
#'   typically be left at its default (no thinning) unless memory is a problem.
#' @param adapt_engaged (logical) Do warmup adaptation? The default is `TRUE`.
#' @param adapt_gamma (positive real) Adaptation regularization scale.
#' @param adapt_delta (real in `(0,1)`) The adaptation target acceptance
#'   statistic.
#' @param adapt_kappa (positive real) Adaptation relaxation exponent.
#' @param adapt_t0 (positive real) Adaptation iteration offset.
#' @param adapt_init_buffer (nonnegative integer) Width of initial fast timestep
#'   adaptation interval during warmup.
#' @param adapt_term_buffer (nonnegative integer) Width of final fast timestep
#'   adaptation interval during warmup.
#' @param adapt_window (nonnegative integer) Initial width of slow timestep/metric
#'   adaptation interval.
#' @param int_time (positive real) Total integration time
#' @param max_treedepth (positive integer) The maximum allowed tree depth for
#'   the NUTS engine.
#' @param metric (string) One of `"diag_e"`, `"dense_e"`, or `"unit_e"`,
#'   specifying the geometry of the base manifold.
#' @param metric_file (character vector) The paths to JSON or
#'   Rdump files (one per chain) compatible with CmdStan that contain
#'   precomputed inverse metrics.
#' @param stepsize (positive real) The _initial_ step size for the discrete
#'   approximation to continuous Hamiltonian dynamics.
#' @param stepsize_jitter (real in `(0,1)`) Allows step size to be “jittered”
#'    randomly during sampling to avoid any poor interactions with a
#'    fixed step size and regions of high curvature.
#' @param check_diagnostics (logical) Whether to check for common problems
#'   with HMC sampling (divergent transitions, max tree depth hits, and
#'   low Bayesian fraction of missing information). Default is `TRUE`.
#' @return \code{StanMCMC} object
#' @export
stan_sample <- function(fn, par_inits = NULL, n_pars = NULL, additional_args = list(),
                          algorithm = "hmc", engine = "nuts",
                          grad_fun = NULL, lower = -Inf, upper = Inf,
                          eval_standalone = (parallel_chains > 1),
                          globals = TRUE, packages = NULL,
                          seed = NULL,
                          refresh = NULL,
                          quiet = FALSE,
                          output_dir = NULL,
                          output_basename = NULL,
                          sig_figs = NULL,
                          num_chains = 4,
                          parallel_chains = 1,
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
                          stepsize_jitter = NULL,
                          check_diagnostics = TRUE) {
  if (!isTRUE(eval_standalone) && parallel_chains > 1) {
    stop("Cannot run parallel chains when evaluating in current R session!",
         call. = FALSE)
  }
  inputs <- prepare_inputs(fn, par_inits, n_pars, additional_args, grad_fun, lower, upper,
                            globals, packages, eval_standalone, output_dir, output_basename,
                            num_chains)

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
      engaged = format_bool(adapt_engaged),
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
    save_warmup = format_bool(save_warmup),
    thin = thin,
    num_chains = ifelse(isTRUE(eval_standalone), 1, num_chains)
  )

  if (isTRUE(eval_standalone)) {
    chain_calls <- lapply(seq_len(num_chains), function(chain) {
      output <- list(
        file = paste0(inputs$output_basename, "_", chain, ".csv"),
        diagnostic_file = NULL,
        refresh = refresh,
        sig_figs = sig_figs,
        profile_file = NULL
      )
      args <- build_stan_call(method = "sample",
                              method_args = method_args,
                              data_file = inputs$data_filepath,
                              init = inputs$init_filepath[chain],
                              seed = seed,
                              output_args = output,
                              id = chain)
      list(args, inputs)
    })

    parallel_procs <- min(parallel_chains, num_chains)
    r_bg_procs <- lapply(seq_len(parallel_procs), function(chain) {
      list(
        chain_id = chain,
        proc = callr::r_bg(call_stan_impl, args = chain_calls[[chain]], package = "StanEstimators", supervise = TRUE)
      )
    })

    chains_alive <- parallel_procs
    chains_to_run <- num_chains - parallel_procs

    # TODO: Clean this up, code-smell
    while((chains_alive > 0) || (chains_to_run > 0)) {
      for (chain in seq_len(parallel_procs)) {
        if (r_bg_procs[[chain]]$proc$is_alive()) {

          r_bg_procs[[chain]]$proc$wait(0.1)
          r_bg_procs[[chain]]$proc$poll_io(0)
          if (!quiet) {
            lines <- r_bg_procs[[chain]]$proc$read_output_lines()
            if (length(lines) > 0) {
              for (line in lines) {
                if (line != "") {
                  cat(paste0("Chain ", r_bg_procs[[chain]]$chain_id, ": ", line, "\n"))
                }
              }
            }
          }
        } else {
          errs <- r_bg_procs[[chain]]$proc$read_error_lines()
          errs <- errs[errs != ""]
          if (length(errs) > 0) {
            message(paste0(errs, collapse = " "))
          }

          if (chains_to_run > 0) {
            r_bg_procs[[chain]] <- list(
              chain_id = num_chains - chains_to_run + 1,
              proc = callr::r_bg(call_stan_impl, args = chain_calls[[num_chains - chains_to_run + 1]], package = "StanEstimators", supervise = TRUE)
            )
            chains_to_run <- chains_to_run - 1
          }
        }
      }
      chains_alive <- sum(sapply(r_bg_procs, function(proc) { proc$proc$is_alive() }))
    }
  } else {
    output <- list(
      file = paste0(inputs$output_basename, ".csv"),
      diagnostic_file = NULL,
      refresh = refresh,
      sig_figs = sig_figs,
      profile_file = NULL
    )
    args <- build_stan_call(method = "sample",
                            method_args = method_args,
                            data_file = inputs$data_filepath,
                            init = paste0(inputs$init_filepath, collapse = ","),
                            seed = seed,
                            output_args = output)
    call_stan_impl(args, inputs)
  }

  if (!isTRUE(eval_standalone) && num_chains == 1) {
    output_files <- paste0(inputs$output_basename, ".csv")
  } else {
    output_files <- paste0(inputs$output_basename, paste0("_", 1:num_chains, ".csv"))
  }

  all_samples <- lapply(output_files, function(filepath) {
    parse_csv(filepath, lower=inputs$lower, upper=inputs$upper)
  })
  draw_names <- all_samples[[1]]$header
  metadata <- all_samples[[1]]$metadata
  adaptation <- lapply(all_samples, function(chain) { chain$adaptation })
  timing <- lapply(all_samples, function(chain) { chain$timing })
  draws <- lapply(seq_len(num_chains), function(chain) {
    dr_df <- setNames(data.frame(all_samples[[chain]]$samples), draw_names)
    dr_df$.chain <- chain
    dr_df
  })
  diagnostic_vars <- c("accept_stat__", "stepsize__", "treedepth__", "n_leapfrog__", "divergent__", "energy__")
  par_vars <- draw_names[!(draw_names %in% diagnostic_vars)]
  draws <- posterior::as_draws_df(do.call(rbind.data.frame, draws))
  diagnostics <- posterior::subset_draws(draws, variable = diagnostic_vars)

  if (check_diagnostics) {
    if (isTRUE(save_warmup)) {
      check_hmc_diagnostics(diagnostics[diagnostics$.iteration > num_warmup, ],
                            as.numeric(metadata$max_depth))
    } else {
    check_hmc_diagnostics(diagnostics, as.numeric(metadata$max_depth))
    }
  }

  methods::new("StanMCMC",
    metadata = metadata,
    adaptation = adaptation,
    timing = timing,
    diagnostics = diagnostics,
    draws = posterior::subset_draws(draws, variable = par_vars),
    lower_bounds = inputs$lower,
    upper_bounds = inputs$upper,
    model_methods = list(
      data_json_string = inputs$data_string,
      model_pointer = make_model_pointer(inputs$data_string, seed)
    )
  )
}
