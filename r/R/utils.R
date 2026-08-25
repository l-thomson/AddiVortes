# Shared helpers: error signalling, seed resolution, and the JSON encoding
# of a configuration.

#' Signal an error of the package's condition class
#'
#' @param message The message, as `rlang::abort` takes it.
#' @param call The calling environment to report.
#' @noRd
thiessen_abort <- function(message, call = rlang::caller_env()) {
  rlang::abort(message, class = "thiessen_error", call = call)
}

#' Call the core, re-signalling its errors with the package's class
#'
#' @param expr The call to the compiled core.
#' @param call The calling environment to report.
#' @noRd
core_call <- function(expr, call = rlang::caller_env()) {
  tryCatch(
    expr,
    error = function(condition) {
      thiessen_abort(conditionMessage(condition), call = call)
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
    state$handle <- core_call(core_state_restore(state$payload), call = call)
  }
  state$handle
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
  if (!is.numeric(seed) || length(seed) != 1L || is.na(seed)) {
    thiessen_abort("`seed` must be `NULL` or a single number.", call = call)
  }
  seed <- as.double(seed)
  if (seed < 0 || seed != trunc(seed) || seed > 2^53) {
    thiessen_abort(
      "`seed` must be a whole number in [0, 2^53].",
      call = call
    )
  }
  seed
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
  if (!is.numeric(chains) || length(chains) != 1L || is.na(chains) ||
        chains < 1 || chains != trunc(chains)) {
    thiessen_abort("`chains` must be a whole number of at least 1.",
                   call = call)
  }
  as.integer(chains)
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
