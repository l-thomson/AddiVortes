# Shared helpers: error signalling, argument checks, seed resolution, and
# the JSON encoding of a configuration.

#' Signal an error of the package's condition class
#'
#' @param message The message, as `rlang::abort` takes it.
#' @param class A class to place before `thiessen_error`, or `NULL`.
#' @param call The calling environment to report.
#' @noRd
thiessen_abort <- function(message, class = NULL,
                           call = rlang::caller_env()) {
  rlang::abort(message, class = c(class, "thiessen_error"), call = call)
}

#' Re-signal an rlang input check under the package's condition class
#'
#' rlang's `check_*()` helpers signal `rlang_error`; every error this
#' package raises carries `thiessen_error` (srr BS2.15). The message and
#' the reported call are rlang's.
#'
#' @param expr One `rlang::check_*()` call.
#' @param call The calling environment to report.
#' @noRd
resignal <- function(expr, call = rlang::caller_env()) {
  rlang::try_fetch(
    expr,
    error = function(condition) {
      thiessen_abort(conditionMessage(condition), call = call)
    }
  )
}

#' Reject a value that is not a single number
#'
#' @param x The value to check.
#' @param ... Passed to [rlang::check_number_decimal()]: `min`, `max`,
#'   `allow_null`.
#' @param arg The argument name to report.
#' @param call The calling environment to report.
#' @noRd
check_number <- function(x, ..., arg = rlang::caller_arg(x),
                         call = rlang::caller_env()) {
  resignal(
    rlang::check_number_decimal(x, ..., arg = arg, call = call),
    call = call
  )
}

#' Reject a value that is not a single whole number
#'
#' @param x The value to check.
#' @param ... Passed to [rlang::check_number_whole()]: `min`, `max`,
#'   `allow_null`.
#' @param arg The argument name to report.
#' @param call The calling environment to report.
#' @noRd
check_whole_number <- function(x, ..., arg = rlang::caller_arg(x),
                               call = rlang::caller_env()) {
  resignal(
    rlang::check_number_whole(x, ..., arg = arg, call = call),
    call = call
  )
}

#' Reject a value that is not `TRUE` or `FALSE`
#'
#' @param x The value to check.
#' @param arg The argument name to report.
#' @param call The calling environment to report.
#' @noRd
check_flag <- function(x, arg = rlang::caller_arg(x),
                       call = rlang::caller_env()) {
  resignal(rlang::check_bool(x, arg = arg, call = call), call = call)
}

#' Reject a value that is not a probability in the open interval (0, 1)
#'
#' rlang's bounds are inclusive and it offers no exclusive option, so the
#' interval is checked here.
#'
#' @param x The value to check.
#' @param arg The argument name to report.
#' @param call The calling environment to report.
#' @noRd
check_probability <- function(x, arg = rlang::caller_arg(x),
                              call = rlang::caller_env()) {
  check_number(x, arg = arg, call = call)
  if (x <= 0 || x >= 1) {
    thiessen_abort(
      paste0(
        "`", arg, "` must be a number strictly between 0 and 1, not the ",
        "number ", format(x), "."
      ),
      call = call
    )
  }
  invisible(NULL)
}

#' Reject a group argument of the wrong class
#'
#' @param value The value to check.
#' @param name The argument name to report.
#' @param constructor The constructor whose class is required.
#' @param null_ok Whether `NULL` passes.
#' @param call The calling environment to report.
#' @noRd
check_group <- function(value, name, constructor, null_ok = FALSE,
                        call = rlang::caller_env()) {
  if (null_ok && is.null(value)) {
    return(invisible(NULL))
  }
  if (!inherits(value, constructor)) {
    thiessen_abort(
      paste0("`", name, "` must come from `", constructor, "()`."),
      call = call
    )
  }
  invisible(NULL)
}

# Leads the message of a core error naming the `experimental` feature.
# The extendr error channel carries a string, so the condition class
# travels as this prefix; `r/src/rust/src/lib.rs` writes it.
REQUIRES_FEATURE <- "thiessen_requires_feature: "

#' Call the core, re-signalling its errors with the package's class
#'
#' An error naming the core's `experimental` feature also carries the
#' class `thiessen_requires_feature` and the instruction for opting in,
#' so a caller can handle it apart from an invalid configuration.
#'
#' @param expr The call to the compiled core.
#' @param call The calling environment to report.
#' @noRd
core_call <- function(expr, call = rlang::caller_env()) {
  tryCatch(
    expr,
    error = function(condition) {
      message <- conditionMessage(condition)
      if (!startsWith(message, REQUIRES_FEATURE)) {
        thiessen_abort(message, call = call)
      }
      thiessen_abort(
        c(
          substring(message, nchar(REQUIRES_FEATURE) + 1L),
          i = paste0(
            "Install the package from source with `THIESSEN_EXPERIMENTAL=1` ",
            "in the environment to build the core with the feature."
          )
        ),
        class = "thiessen_requires_feature",
        call = call
      )
    }
  )
}

#' The live fitted-state pointer of a fit
#'
#' The state lives on the Rust side behind an external pointer, which
#' `readRDS` deserialises with a null address. The pointer sits in an
#' environment, so restoring it from the payload once serves every copy of
#' the fit in the session.
#'
#' @param object An object of class `"thiessen"`.
#' @param call The calling environment to report.
#' @return An external pointer to the fitted state.
#' @noRd
fit_state <- function(object, call = rlang::caller_env()) {
  state <- object$state
  if (!core_state_is_live(state$handle)) {
    state$handle <- core_call(
      core_state_restore(state$payload, fit_threads(object)),
      call = call
    )
  }
  state$handle
}

#' The thread count a fit predicts on
#'
#' @param object An object of class `"thiessen"`.
#' @return An integer; one for a fit without the field.
#' @noRd
fit_threads <- function(object) {
  threads <- object$threads
  if (is.null(threads)) {
    return(1L)
  }
  as.integer(threads)
}

#' Resolve the seed of a fit
#'
#' `NULL` draws from R's stream, so `set.seed` governs. An integer passes
#' through unchanged, so the same value reproduces the core's draws.
#'
#' @param seed `NULL` or a whole number in `[0, 2^53]`.
#' @param call The calling environment to report.
#' @return A double: the seed the core receives.
#' @noRd
resolve_seed <- function(seed, call = rlang::caller_env()) {
  if (is.null(seed)) {
    return(as.double(sample.int(.Machine$integer.max, 1L)))
  }
  check_whole_number(seed, min = 0, max = 2^53, call = call)
  as.double(seed)
}

#' Encode a control object as the core's configuration JSON
#'
#' Fields left `NULL` are omitted, so the core's defaults apply; the core
#' rejects unknown fields and validates every value.
#'
#' @param control An object of class `"thiessen_control"`.
#' @param call The calling environment to report.
#' @return A character string.
#' @noRd
config_json <- function(control, call = rlang::caller_env()) {
  if (!inherits(control, "thiessen_control")) {
    thiessen_abort(
      "`control` must come from `thiessen_control()`.",
      call = call
    )
  }
  mean_params <- term_group(control$mean_params)
  variance_params <- term_group(control$variance_params)
  # The ensembles share one covariate space; the core requires the slots
  # to declare it identically while per-ensemble geometry awaits its
  # identification argument.
  if (length(variance_params) > 0L) {
    for (shared in c("geometry", "structure")) {
      if (is.null(variance_params[[shared]]) &&
            !is.null(mean_params[[shared]])) {
        variance_params[[shared]] <- mean_params[[shared]]
      }
    }
  }
  grouped <- list(outcome = outcome_group(control$outcome))
  if (length(mean_params) > 0L) grouped$mean_params <- mean_params
  if (length(variance_params) > 0L) {
    grouped$variance_params <- variance_params
  }
  grouped$general_params <- compact(unclass(control$general_params))
  jsonlite::toJSON(grouped, auto_unbox = TRUE, digits = NA, null = "null")
}

#' The configuration group of one ensemble
#'
#' @param params An object of class `"term_params"`, or `NULL`.
#' @return A named list, `NULL` fields omitted; empty for `NULL`.
#' @noRd
term_group <- function(params) {
  if (is.null(params)) {
    return(structure(list(), names = character(0)))
  }
  group <- compact(unclass(params))
  if (!is.null(group$geometry)) {
    group$geometry <- compact(unclass(group$geometry))
  }
  if (!is.null(group$structure)) {
    group$structure <- compact(unclass(group$structure))
    if (length(group$structure) == 0L) group$structure <- NULL
  }
  group
}

#' The control object of a resolved configuration
#'
#' Rebuilds the nested groups from the grouped configuration the core
#' reports after a fit, in which every field is set.
#'
#' @param grouped The core's configuration as a named list.
#' @return An object of class `"thiessen_control"`.
#' @noRd
control_from_config <- function(grouped) {
  kind <- names(grouped$outcome)[[1L]]
  outcome <- new_outcome(kind, grouped$outcome[[1L]])
  structure(
    list(
      outcome = outcome,
      mean_params = term_from_config(grouped$mean_params),
      variance_params = term_from_config(grouped$variance_params),
      general_params = structure(grouped$general_params,
                                 class = "general_params")
    ),
    class = "thiessen_control"
  )
}

#' One resolved ensemble group as a `term_params` object
#'
#' @param group The ensemble's configuration as a named list.
#' @return An object of class `"term_params"`, or `NULL` for an absent or
#'   empty variance slot.
#' @noRd
term_from_config <- function(group) {
  if (is.null(group)) {
    return(NULL)
  }
  count <- group$tessellations
  if (!is.null(count) && count == 0L) {
    return(NULL)
  }
  if (!is.null(group$geometry)) {
    if (length(group$geometry$metric) == 0L) group$geometry$metric <- NULL
    class(group$geometry) <- "geometry_params"
  }
  if (!is.null(group$structure)) {
    class(group$structure) <- "structure_params"
  }
  class(group) <- "term_params"
  group
}

#' Coerce a design to the numeric matrix the core takes
#'
#' @param x A numeric matrix or a numeric vector, taken as one column.
#' @param argument The name to report in an error.
#' @param call The calling environment to report.
#' @return A double matrix.
#' @noRd
as_design <- function(x, argument = "x", call = rlang::caller_env()) {
  if (is.data.frame(x)) {
    thiessen_abort(
      paste0("`", argument, "` must be a numeric matrix, not a data frame."),
      call = call
    )
  }
  if (is.null(dim(x))) {
    x <- matrix(x, ncol = 1L)
  }
  if (length(dim(x)) != 2L || !is.numeric(x)) {
    thiessen_abort(
      paste0("`", argument, "` must be a numeric matrix."),
      call = call
    )
  }
  if (anyNA(x)) {
    thiessen_abort(
      paste0("`", argument, "` must not contain missing values."),
      call = call
    )
  }
  storage.mode(x) <- "double"
  x
}

#' Resolve the number of chains of a fit
#'
#' @param chains A whole number of chains.
#' @param call The calling environment to report.
#' @return An integer.
#' @noRd
resolve_chains <- function(chains, call = rlang::caller_env()) {
  check_whole_number(chains, min = 1, call = call)
  as.integer(chains)
}

#' Resolve the number of threads of a fit
#'
#' @param threads A whole number of threads.
#' @param call The calling environment to report.
#' @return An integer.
#' @noRd
resolve_threads <- function(threads, call = rlang::caller_env()) {
  if (!is.numeric(threads) || length(threads) != 1L || is.na(threads) ||
        threads < 1 || threads != trunc(threads)) {
    thiessen_abort("`threads` must be a whole number of at least 1.",
                   call = call)
  }
  as.integer(threads)
}

#' The number of progress reports a fit signals
#'
#' Progress is signalled with progressr, so a session reports it only after
#' `progressr::handlers()`; nothing is printed by default.
#'
#' @param control An object of class `"thiessen_control"`.
#' @param chains The number of chains the fit runs.
#' @return An integer: one report per sweep, to a maximum of a hundred.
#' @noRd
progress_updates <- function(control, chains = 1L) {
  schedule <- control$general_params
  sweeps <- chains *
    (schedule$burn_in + schedule$draws * schedule$thinning)
  as.integer(min(sweeps, 100L))
}

# Pooling the draws predicts at every training row for every kept draw, so
# it costs about twice what the sweeps cost, in multiples of them. A step
# apiece would leave the bar all but complete over the longer phase.
POOLING_WEIGHT <- 2L

#' The number of steps a fit's progressor takes
#'
#' @param control An object of class `"thiessen_control"`.
#' @param chains The number of chains the fit runs.
#' @return An integer: the sweep reports, pooling at its weight, and one
#'   step for the convergence summary.
#' @noRd
progress_steps <- function(control, chains = 1L) {
  updates <- progress_updates(control, chains)
  updates + POOLING_WEIGHT * updates + 1L
}

#' The message the sweeps report
#'
#' @param chains The number of chains the fit runs.
#' @param threads The number of threads the chains run on.
#' @return A string.
#' @noRd
sweep_message <- function(chains, threads) {
  if (chains == 1L) {
    return("sampling")
  }
  if (threads == 1L) {
    return(sprintf("sampling %d chains", chains))
  }
  sprintf("sampling %d chains on %d threads", chains, min(chains, threads))
}
