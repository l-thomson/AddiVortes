#' Hyperparameters and sweep schedule of a fit
#'
#' The hyperparameters of Stone and Gosling (2025), s. 2, and the sweep
#' schedule `thiessen()` runs. Every argument defaults to the core's own
#' default, reported by `core_defaults()`, so an unset argument and an
#' argument set to its default give the same fit.
#'
#' The three models are the published method and follow semantic versioning.
#' Everything else the core crate adds sits behind its `experimental` Cargo
#' feature, which this package does not enable, so a configuration or a
#' saved fit naming such an option is rejected with the core's message
#' naming the feature. The table of experimental items and their status is
#' `docs/experimental.md` in the repository. A graduated item is accepted
#' here as any other option, with no separate opt-in.
#'
#' @param model The observation model: `"gaussian"`, `"probit"` for a binary
#'   response, or `"heteroscedastic"` for a variance that varies with x.
#' @param m Ensemble size m of the mean function. Default 200.
#' @param nu Degrees of freedom nu of the sigma^2 prior,
#'   sigma^2 ~ nu lambda / chi^2_nu. Default 6. The heteroscedastic model
#'   requires nu > 2.
#' @param q Calibration quantile q of the sigma^2 prior,
#'   Pr(sigma < sigma_hat) = q. Default 0.85.
#' @param k Cell-mean prior spread k: sigma_mu = 0.5 / (k sqrt(m)) on the
#'   response scaled to \[-0.5, 0.5\], and 3 / (k sqrt(m)) on the latent
#'   scale of the probit model (Chipman, George and McCulloch 2010, s. 4).
#'   Default 3.
#' @param sigma_c Prior and proposal standard deviation sigma_c of a centre
#'   coordinate in the scaled space. Default 0.8.
#' @param omega Dimension-count prior parameter omega; omega / p is the
#'   prior probability of including a covariate. `NULL`, the default,
#'   resolves to min(3, p) at fit. Must satisfy 0 < omega <= p.
#' @param lambda_c Cell-count prior rate lambda_c: b - 1 ~ Poisson(lambda_c).
#'   Default 5, following AddiVortes 0.6.8 and later; the paper reports 25.
#' @param burn_in Sweeps discarded before the kept draws. Default 200.
#' @param draws Posterior draws kept. Default 1000.
#' @param thinning Keep every `thinning`-th sweep after burn-in. Default 1.
#' @param prior_only Switch off the likelihood, so the chain draws from the
#'   prior and `predict()` gives prior predictive draws. Default `FALSE`.
#' @param offset Probit model only: the offset c in
#'   P(y = 1 | x) = Phi(c + f(x)). `NULL`, the default, resolves to
#'   Phi^-1(ybar) at fit.
#' @param m_var Heteroscedastic model only: the number m' of variance
#'   tessellations. Default 40.
#' @param metric The metric of each covariate column, in column order: a
#'   list whose entries are `"euclidean"`, `"categorical"`, or
#'   `list(spherical = list(sphere = i))` for one coordinate of the sphere
#'   labelled `i`, its latitudes first and its longitude last, in radians.
#'   The default, an empty list, is Euclidean on every column.
#'
#' @return An object of class `"thiessen_control"`: the core's default
#'   configuration with the arguments given substituted.
#'
#' @references
#' Stone, A. and Gosling, J. P. (2025). AddiVortes: (Bayesian) additive
#' Voronoi tessellations. *Journal of Computational and Graphical
#' Statistics* 34(3), 859-871. \doi{10.1080/10618600.2024.2414104}
#'
#' @examples
#' thiessen_control(m = 50, draws = 200)
#' @export
thiessen_control <- function(model = NULL, m = NULL, nu = NULL, q = NULL,
                             k = NULL, sigma_c = NULL, omega = NULL,
                             lambda_c = NULL, burn_in = NULL, draws = NULL,
                             thinning = NULL, prior_only = NULL,
                             offset = NULL, m_var = NULL, metric = NULL) {
  defaults <- flatten_config(jsonlite::fromJSON(core_defaults(), simplifyVector = FALSE))
  given <- list(
    model = model, m = m, nu = nu, q = q, k = k, sigma_c = sigma_c,
    omega = omega, lambda_c = lambda_c, burn_in = burn_in, draws = draws,
    thinning = thinning, prior_only = prior_only, offset = offset,
    m_var = m_var, metric = metric
  )
  if (!is.null(given$metric)) {
    given$metric <- as.list(given$metric)
  }
  control <- defaults
  # The entries of `metric` are unnamed, so each field is assigned rather
  # than merged by name.
  for (name in names(given)) {
    if (!is.null(given[[name]])) {
      control[[name]] <- given[[name]]
    }
  }
  # `omega` and `offset` are null in the defaults and stay null unless
  # supplied; the core resolves both from the data at fit.
  control <- structure(control[names(defaults)], class = "thiessen_control")
  core_call(core_validate(config_json(control)))
  control
}

#' Print a control object
#'
#' @param x An object of class `"thiessen_control"`.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @examples
#' print(thiessen_control(m = 50))
#' @export
print.thiessen_control <- function(x, ...) {
  cat("<thiessen_control>\n")
  fields <- unclass(x)
  for (name in names(fields)) {
    value <- fields[[name]]
    shown <- if (is.null(value)) {
      "resolved at fit"
    } else if (name == "metric") {
      if (length(value) == 0L) "euclidean on every column" else format_metric(value)
    } else {
      format(value)
    }
    cat(sprintf("  %-11s %s\n", name, shown))
  }
  invisible(x)
}

#' One line naming each column's metric
#'
#' @param metric The `metric` field of a control object.
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
