# Wrappers for the functions exported by src/rust/src/lib.rs, one per
# `#[extendr]` function, by the registered symbol name.

#' Version of the core crate
#'
#' @return A character string: the semantic version of the Rust core the
#'   package was built with, equal to the `Config/thiessen/core-version`
#'   field of the package `DESCRIPTION`.
#' @examples
#' core_version()
#' @export
core_version <- function() .Call(wrap__core_version)

core_defaults <- function() .Call(wrap__core_defaults)

core_validate <- function(config_json) .Call(wrap__core_validate, config_json)

core_fit <- function(config_json, x, y, seed_value, report, updates) {
  .Call(wrap__core_fit, config_json, x, y, seed_value, report, updates)
}

core_predict <- function(state_json, x) .Call(wrap__core_predict, state_json, x)

core_predict_draws <- function(state_json, x, kind) {
  .Call(wrap__core_predict_draws, state_json, x, kind)
}

core_interval <- function(state_json, x, kind, level) {
  .Call(wrap__core_interval, state_json, x, kind, level)
}

core_sigma <- function(state_json) .Call(wrap__core_sigma, state_json)

core_log_lik <- function(state_json, x, y) {
  .Call(wrap__core_log_lik, state_json, x, y)
}

core_diagnostics <- function(state_json) {
  .Call(wrap__core_diagnostics, state_json)
}
