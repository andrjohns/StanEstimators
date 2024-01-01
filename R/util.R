write_data <- function(Npars, finite_diff, lower_bounds, upper_bounds, bounds_types, data_filepath) {
  no_bounds <- all(lower_bounds == -Inf) && all(upper_bounds == Inf)
  dat_string <- paste(
    '{',
    '"Npars" : ', Npars, ',',
    '"finite_diff" : ', finite_diff, ',',
    '"no_bounds" : ', as.integer(no_bounds), ',',
    '"lower_bounds" : [', paste0(lower_bounds, collapse = ','), '],',
    '"upper_bounds" : [', paste0(upper_bounds, collapse = ','), '],',
    '"bounds_types" : [', paste0(bounds_types, collapse = ','), ']',
    '}'
  )
  writeLines(dat_string, con = data_filepath)
  invisible(NULL)
}

write_inits <- function(inits, init_filepath) {
  dat_string <- paste(
    '{',
    '"pars" : [', paste0(inits, collapse = ','), ']',
    '}'
  )
  writeLines(dat_string, con = init_filepath)
  invisible(NULL)
}

write_file <- function(what, input_list) {
  if (what == "data") {
    args <- c("Npars", "finite_diff", "lower_bounds", "upper_bounds", "bounds_types", "data_filepath")
    write_fun <- write_data
  } else if (what == "inits") {
    args <- c("inits", "init_filepath")
    write_fun <- write_inits
  } else {
    args <- c()
    write_fun <- function() { stop("Invalid write method!", call. = FALSE) }
  }
  do.call(write_fun, input_list[args])
}

with_env <- function(f, e=parent.frame()) {
  stopifnot(is.function(f))
  environment(f) <- e
  f
}

prepare_function <- function(fn, inits, extra_args_list, grad = FALSE) {
  fn_wrapper <- function(v) { do.call(fn, c(list(v), extra_args_list)) }
  fn_type <- ifelse(isTRUE(grad), "Gradient", "Log-Likelihood")
  test_fn <- try(invisible(fn_wrapper(inits)), silent = TRUE)
  correct_length <- ifelse(isTRUE(grad), length(inits), 1)

  if (inherits(test_fn, "try-error")) {
    stop(fn_type, " function evaluated at initial values resulted in error: \n ",
         test_fn, call. = FALSE)
  } else if (any(!is.finite(test_fn))) {
    stop(fn_type, " function evaluated at initial values is not finite!",
         call. = FALSE)
  } else if (length(test_fn) != correct_length) {
    stop(fn_type, " function should have return of length ", correct_length,
          ", but return was length ", length(test_fn), "instead!", call. = FALSE)
  } else {
    fn_wrapper
  }
}

prepare_inputs <- function(fn, par_inits, extra_args_list, grad_fun, lower, upper,
                            globals, packages, output_dir, output_basename) {
  fn1 <- prepare_function(fn, par_inits, extra_args_list, grad = FALSE)
  gp <- future::getGlobalsAndPackages(fn, globals = globals)
  if (!is.null(grad_fun)) {
    gr1 <- prepare_function(grad_fun, par_inits, extra_args_list, grad = TRUE)
    gr_gp <- future::getGlobalsAndPackages(grad_fun, globals = globals)
    gp$globals <- c(gp$globals, gr_gp$globals)
    gp$packages <- c(gp$packages, gr_gp$packages)
  } else {
    gr1 <- fn1
  }

  if ((length(par_inits) > 1) && (length(lower) == 1)) {
    lower <- rep(lower, length(par_inits))
  }
  if ((length(par_inits) > 1) && (length(upper) == 1)) {
    upper <- rep(upper, length(par_inits))
  }
  bounds_types <- sapply(seq_len(length(par_inits)), function(i) {
    if (lower[i] != -Inf && upper[i] != Inf) {
      3
    } else if (lower[i] != -Inf) {
      1
    } else if (upper[i] != Inf) {
      2
    } else {
      4
    }
  })
  if (is.null(output_dir)) {
    output_dir <- tempdir()
  }
  if (is.null(output_basename)) {
    output_basename <- tempfile(tmpdir = output_dir)
  } else {
    output_basename <- file.path(output_dir, output_basename)
  }

  structured <- list(
    ll_function = fn1,
    grad_function = gr1,
    globals = gp$globals,
    packages = c(gp$packages, packages),
    inits = par_inits,
    finite_diff = as.integer(is.null(grad_fun)),
    Npars = length(par_inits),
    lower_bounds = lower,
    upper_bounds = upper,
    bounds_types = bounds_types,
    data_filepath = tempfile(fileext = ".json", tmpdir = output_dir),
    init_filepath = tempfile(fileext = ".json", tmpdir = output_dir),
    output_filepath = paste0(output_basename, ".csv"),
    output_basename = output_basename
  )
  write_file("inits", structured)
  write_file("data", structured)
  structured
}

cmdstan_syntax_tree <- list(
  "sample" = list(
    "num_samples",
    "num_warmup",
    "save_warmup",
    "thin",
    "adapt" = list(
      "engaged",
      "gamma",
      "delta",
      "kappa",
      "t0",
      "init_buffer",
      "term_buffer",
      "window"
    ),
    "algorithm" = list(
      "hmc" = list(
        "engine" = list(
          "static" = list(
            "int_time"
          ),
          "nuts" = list(
            "max_depth"
          )
        ),
        "metric",
        "metric_file",
        "stepsize",
        "stepsize_jitter"
      ),
      "fixed_param"
    ),
    "num_chains"
  ),
  "optimize" = list(
    "algorithm" = list(
      "bfgs" = list(
        "init_alpha",
        "tol_obj",
        "tol_rel_obj",
        "tol_grad",
        "tol_rel_grad",
        "tol_param"
      ),
      "lbfgs" = list(
        "init_alpha",
        "tol_obj",
        "tol_rel_obj",
        "tol_grad",
        "tol_rel_grad",
        "tol_param",
        "history_size"
      ),
      "newton"
    ),
    "jacobian",
    "iter",
    "save_iterations"
  ),
  "variational" = list(
    "algorithm",
    "iter",
    "grad_samples",
    "elbo_samples",
    "eta",
    "adapt" = list(
      "engaged",
      "iter"
    ),
    "tol_rel_obj",
    "eval_elbo",
    "output_samples"
  ),
  "pathfinder" = list(
    "init_alpha",
    "tol_obj",
    "tol_rel_obj",
    "tol_grad",
    "tol_rel_grad",
    "tol_param",
    "history_size",
    "num_psis_draws",
    "num_paths",
    "save_single_paths",
    "max_lbfgs_iters",
    "num_draws",
    "num_elbo_draws"
  ),
  "laplace" = list(
    "mode",
    "jacobian",
    "draws"
  ),
  "data" = list(
    "file"
  ),
  "init",
  "random" = list(
    "seed"
  ),
  "output" = list(
    "file",
    "diagnostic_file",
    "refresh",
    "sig_figs",
    "profile_file"
  ),
  "num_threads"
)

# Need to refactor this, what a mess
parse_method_args <- function(method, method_args) {
  method_tree <- cmdstan_syntax_tree[[method]]
  return_args <- c("")
  if (!is.null(method_tree$algorithm)) {
    algorithm <- method_args$algorithm
    algo_tree <- method_tree$algorithm[[method_args$algorithm]]
    engine_string <- ""

    if (!is.null(algo_tree$engine)) {
      engine_args <- sapply(algo_tree$engine[[method_args$algorithm_args$engine]], function(eng_arg) {
        ifelse(!is.null(method_args$algorithm_args$engine_args[[eng_arg]]),
               paste0(eng_arg, "=", method_args$algorithm_args$engine_args[[eng_arg]]),
               "")
      })
      algo_tree <- algo_tree[-grep("engine", names(algo_tree))]
      engine_string <- c(paste0("engine=", method_args$algorithm_args$engine), engine_args)
    }

    algorithm_args <- sapply(algo_tree, function(arg) {
        ifelse(!is.null(method_args$algorithm_args[[arg]]),
              paste0(arg, "=", method_args$algorithm_args[[arg]]),
              "")
    })
    return_args <- c(return_args, paste0("algorithm=", algorithm), engine_string, algorithm_args[algorithm_args != ""])
    method_tree <- method_tree[-grep("algorithm", names(method_tree))]
  }
  if (!is.null(method_tree$adapt)) {
    adapt_args <- sapply(method_tree$adapt, function(arg) {
      ifelse(!is.null(method_args$adapt[[arg]]),
             paste0(arg, "=", method_args$adapt[[arg]]),
             "")
    })
    adapt_args <- adapt_args[adapt_args != ""]
    if (length(adapt_args) > 0)
    return_args <- c(return_args, "adapt", adapt_args)
    method_tree <- method_tree[-grep("adapt", names(method_tree))]
  }

  other_args <- sapply(method_tree, function(arg) {
    ifelse(!is.null(method_args[[arg]]),
          paste0(arg, "=", method_args[[arg]]),
          "")
  })
  c(return_args, unlist(other_args[other_args != ""]))
}

parse_output_args <- function(output_args) {
  valid_args <- cmdstan_syntax_tree[["output"]]
  parsed_args <- sapply(valid_args, function(arg) {
    ifelse(!is.null(output_args[[arg]]),
      paste0(arg, "=", output_args[[arg]]),
      "")
  })
  c("output", parsed_args[parsed_args != ""])
}

build_stan_call <- function(method, method_args, data_file, init, seed, output_args) {
  if (method == "diagnose") {
    method_string <- ""
  } else {
    method_string <- parse_method_args(method, method_args)
  }
  data_string <- c("data", paste0("file=", data_file))
  init_string <- paste0("init=", init)
  if (!is.null(seed)) {
    random_string <- c("random", paste0("seed=", seed))
  } else {
    random_string <- ""
  }
  output_string <- parse_output_args(output_args)
  args <- unlist(c(method, method_string, data_string, init_string, random_string, output_string))
  args[args != ""]
}

call_stan <- function(args_list, input_list, quiet) {
  finished_metadata <- FALSE
  r_bg_args <- list(
    args_list,
    input_list
  )
  proc <- callr::r_bg(call_stan_impl, args = r_bg_args,
                      supervise = TRUE,
                      package = "StanEstimators")
  while (proc$is_alive()) {
    proc$wait(0.1)
    proc$poll_io(0)
    if (!quiet) {
      lines <- proc$read_output_lines()
      if (length(lines) > 0) {
        for (line in lines) {
          if (finished_metadata && line != "") {
            cat(line, "\n")
          }
          if (grepl("num_threads", line)) {
            finished_metadata <- TRUE
          }
        }
      }
    }
  }
  errs <- proc$read_error_lines()
  if (length(errs) > 0) {
    stop(paste0(errs, collapse = "\n"), call. = FALSE)
  }
  invisible(NULL)
}

format_bool <- function(input) {
  if (!is.null(input)) {
    input <- as.integer(input)
  }
  input
}
