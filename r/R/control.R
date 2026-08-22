#' Configuration of a fit
#'
#' The configuration of Stone and Gosling (2025), s. 2, in the shape the
#' core stores it: an outcome family, one parameter group per ensemble,
#' and the sweep schedule. Each part has its own constructor with its own
#' documentation: [gaussian()] and [probit()] for the family,
#' [term_params()] for an ensemble, and [general_params()] for the
#' schedule. An argument left at its default gives the core's default, so
#' `thiessen_control()` is the published configuration.
#'
#' Attaching `variance_params` with a positive tessellation count selects
#' the heteroscedastic model, in which the residual variance varies with
#' x; it needs the Gaussian family with nu > 2, and the paper's count is
#' 40. The two ensembles share one covariate space: `geometry` and
#' `structure` set on `mean_params` apply to `variance_params` as well.
#'
#' One shortcut: `thiessen_control(tessellations = 200)` sets the mean
#' ensemble's size without spelling the group, since that count is the
#' single number most fits tune. Every other setting is named in its
#' group.
#'
#' The models reachable here are the published method and follow semantic
#' versioning. Everything else the core crate adds sits behind its
#' `experimental` Cargo feature, which this package does not enable, so a
#' configuration or a saved fit naming such an option is rejected with the
#' core's message naming the feature. The table of experimental items and
#' their status is `docs/experimental.md` in the repository. A graduated
#' item is accepted here as any other option, with no separate opt-in.
#'
#' The core's calibration suite covers the configurations listed in
#' `docs/calibrated.md` in the repository; component options are verified
#' in isolation, and every other combination of the documented options is
#' valid to run and is not separately verified.
#'
#' @param outcome The outcome family, from [gaussian()] or [probit()].
#' @param mean_params The ensemble describing the average, from
#'   [term_params()].
#' @param variance_params The ensemble describing the spread, from
#'   [term_params()]. `NULL`, the default, keeps the spread constant.
#' @param general_params The sweep schedule, from [general_params()].
#'   `NULL`, the default, is `general_params()`.
#' @param tessellations Shortcut for the mean ensemble's size:
#'   `thiessen_control(tessellations = 200)` is
#'   `thiessen_control(mean_params = term_params(tessellations = 200))`.
#'   An error if `mean_params` also sets a count.
#'
#' @return An object of class `"thiessen_control"` holding the four
#'   groups.
#'
#' @references
#' Stone, A. and Gosling, J. P. (2025). AddiVortes: (Bayesian) additive
#' Voronoi tessellations. *Journal of Computational and Graphical
#' Statistics* 34(3), 859-871. \doi{10.1080/10618600.2024.2414104}
#'
#' @examples
#' thiessen_control(tessellations = 50)
#'
#' thiessen_control(
#'   outcome = gaussian(nu = 10),
#'   mean_params = term_params(tessellations = 200, lambda_c = 25),
#'   variance_params = term_params(tessellations = 40),
#'   general_params = general_params(burn_in = 500, draws = 2000)
#' )
#' @export
thiessen_control <- function(outcome = gaussian(),
                             mean_params = term_params(),
                             variance_params = NULL,
                             general_params = NULL,
                             tessellations = NULL) {
  # The argument shares the constructor's name, so the default is resolved
  # here rather than in the signature, where it would be a recursive
  # promise.
  if (is.null(general_params)) {
    general_params <- thiessen::general_params()
  }
  if (!inherits(outcome, "thiessen_outcome")) {
    thiessen_abort("`outcome` must come from `gaussian()` or `probit()`.")
  }
  check_group(mean_params, "mean_params", "term_params")
  check_group(variance_params, "variance_params", "term_params",
              null_ok = TRUE)
  check_group(general_params, "general_params", "general_params")
  if (!is.null(tessellations)) {
    check_scalar(tessellations, "tessellations")
    if (!is.null(mean_params$tessellations)) {
      thiessen_abort(paste0(
        "`tessellations` is a shortcut for `mean_params`; set the count ",
        "in one place."
      ))
    }
    mean_params$tessellations <- tessellations
  }
  control <- structure(
    list(
      outcome = outcome,
      mean_params = mean_params,
      variance_params = variance_params,
      general_params = general_params
    ),
    class = "thiessen_control"
  )
  core_call(core_validate(config_json(control)))
  control
}

#' Print a control object
#'
#' @param x An object of class `"thiessen_control"`.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @examples
#' print(thiessen_control(tessellations = 50))
#' @export
print.thiessen_control <- function(x, ...) {
  cat("<thiessen_control>\n")
  shown <- list(
    outcome = format(x$outcome),
    mean_params = format(x$mean_params),
    variance_params = if (is.null(x$variance_params)) {
      "none (constant spread)"
    } else {
      format(x$variance_params)
    },
    general_params = format(x$general_params)
  )
  for (name in names(shown)) {
    cat(sprintf("  %-15s %s\n", name, shown[[name]]))
  }
  invisible(x)
}

#' One line naming each column's metric
#'
#' @param metric The `metric` field of a geometry group.
#' @return A character string.
#' @noRd
format_metric <- function(metric) {
  named <- vapply(
    metric,
    function(entry) {
      if (is.character(entry)) {
        entry
      } else {
        paste0(names(entry)[1L], "(", entry[[1L]][["sphere"]], ")")
      }
    },
    character(1)
  )
  paste(named, collapse = ", ")
}
