# The outcome families: constructor functions returning classed objects,
# the `binomial(link = "probit")` idiom. The object carries its own
# parameters and serialises as the configuration's `outcome` group, so the
# constructor arguments and the stored form share one set of names.

#' The Gaussian outcome family
#'
#' The Gaussian observation model of Stone and Gosling (2025): one sigma^2
#' drawn per sweep. Attaching a variance ensemble
#' (`variance_params = term_params(tessellations = ...)` with a positive
#' count) makes the model heteroscedastic, so the residual variance varies
#' with x. This function masks [stats::gaussian()] when the package is
#' attached; the `glm` family is still available as `stats::gaussian()`.
#'
#' @param nu Degrees of freedom nu of the sigma^2 prior,
#'   sigma^2 ~ nu lambda / chi^2_nu. Default 6. A variance ensemble
#'   requires nu > 2.
#' @param q Calibration quantile q of the sigma^2 prior,
#'   Pr(sigma < sigma_hat) = q. Default 0.85.
#'
#' @return An object of class `c("thiessen_gaussian", "thiessen_outcome")`.
#'
#' @seealso [probit()], [thiessen_control()]
#' @examples
#' gaussian(nu = 3)
#' @export
gaussian <- function(nu = 6, q = 0.85) {
  check_scalar(nu, "nu")
  check_scalar(q, "q")
  new_outcome("gaussian", list(nu = nu, q = q))
}

#' The binary probit outcome family
#'
#' P(y = 1 | x) = Phi(c + f(x)) with the Albert and Chib (1993) latent
#' augmentation. The latent scale is fixed at 1 for identification, so a
#' variance ensemble is not available under this family.
#'
#' @param offset The offset c. `NULL`, the default, resolves to
#'   Phi^-1(ybar) at fit, the BART `binaryOffset` default.
#'
#' @return An object of class `c("thiessen_probit", "thiessen_outcome")`.
#'
#' @seealso [gaussian()], [thiessen_control()]
#' @examples
#' probit()
#' @export
probit <- function(offset = NULL) {
  if (!is.null(offset)) check_scalar(offset, "offset")
  new_outcome("probit", list(offset = offset))
}

#' Construct a classed outcome family
#'
#' @param kind The core's name for the family.
#' @param fields The family's parameters, `NULL` entries kept.
#' @return A classed list carrying `kind` as an attribute.
#' @noRd
new_outcome <- function(kind, fields) {
  structure(
    fields,
    kind = kind,
    class = c(paste0("thiessen_", kind), "thiessen_outcome")
  )
}

#' The configuration's `outcome` group of a family object
#'
#' @param outcome An object of class `"thiessen_outcome"`.
#' @return A one-entry named list, the core's tagged form.
#' @noRd
outcome_group <- function(outcome) {
  stats::setNames(list(compact(unclass(outcome))), attr(outcome, "kind"))
}

#' @export
format.thiessen_outcome <- function(x, ...) {
  constructor_call(attr(x, "kind"), compact(unclass(x)))
}

#' Print an outcome family
#'
#' @param x An object of class `"thiessen_outcome"`.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @examples
#' print(gaussian())
#' @export
print.thiessen_outcome <- function(x, ...) {
  cat(format(x), "\n", sep = "")
  invisible(x)
}
