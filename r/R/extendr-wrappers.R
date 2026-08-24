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

#' Whether the core was built with its experimental feature
#'
#' The outcome families and options the core crate keeps behind its
#' `experimental` Cargo feature are absent from a build without it, so a
#' configuration or a saved fit naming one is rejected with the core's
#' message naming the feature. This package enables the feature in no
#' build, so the answer is `FALSE` in every released version; report it with
#' [core_version()] in a bug report, since a fit rejected for naming a gated
#' option looks the same either way.
#'
#' @return A logical of length one: whether the core in use was built with
#'   the `experimental` feature.
#' @examples
#' core_experimental()
#' @export
core_experimental <- function() .Call(wrap__core_experimental)

core_defaults <- function() .Call(wrap__core_defaults)

core_validate <- function(config_json) .Call(wrap__core_validate, config_json)

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

core_sampler_new <- function(config_json, x, y, seed_value, chain) {
  .Call(wrap__core_sampler_new, config_json, x, y, seed_value, chain)
}

core_sampler_step <- function(sampler, n) .Call(wrap__core_sampler_step, sampler, n)

core_sampler_keep <- function(sampler) .Call(wrap__core_sampler_keep, sampler)

core_sampler_n_kept <- function(sampler) .Call(wrap__core_sampler_n_kept, sampler)

core_sampler_set_response <- function(sampler, y) {
  .Call(wrap__core_sampler_set_response, sampler, y)
}

core_sampler_fitted_values <- function(sampler) {
  .Call(wrap__core_sampler_fitted_values, sampler)
}

core_sampler_noise_variances <- function(sampler) {
  .Call(wrap__core_sampler_noise_variances, sampler)
}

core_finish <- function(samplers) .Call(wrap__core_finish, samplers)
