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
  fields <- unclass(control)
  jsonlite::toJSON(fields, auto_unbox = TRUE, digits = NA, null = "null")
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
