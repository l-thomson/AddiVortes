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
