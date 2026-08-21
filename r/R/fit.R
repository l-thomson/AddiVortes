#' Fit an AddiVortes model
#'
#' AddiVortes is Bayesian regression on a sum of Voronoi tessellations
#' (Stone and Gosling, 2025): the mean function is a sum of m tessellations,
#' each with a mean per cell, drawn by the Gibbs sampler of the paper. It
#' stands to BART (Chipman, George and McCulloch, 2010) as a tessellation
#' stands to a tree: a cell is a region of the covariate space rather than a
#' box, so a boundary oblique to the axes costs one cell rather than many
#' splits.
#'
#' @param x A numeric matrix of covariates, one row per observation. A
#'   numeric vector is taken as one column.
#' @param y The response: a numeric vector of length `nrow(x)`. Under the
#'   probit model the values must be 0 and 1.
#' @param control An object of class `"thiessen_control"`, from
#'   [thiessen_control()].
#' @param seed The seed of the chain. `NULL`, the default, draws one from
#'   R's stream, so [set.seed()] governs; a whole number in `[0, 2^53]`
#'   passes to the core unchanged, so the same value reproduces the same
#'   draws for a given package version and platform.
#' @param ... Passed to the method.
#'
#' @return An object of class `"thiessen"`: a list with the fitted state,
#'   the resolved configuration, the number of kept draws, the seed used,
#'   the design, the response, the fitted values, the residuals and the
#'   call. Use [predict.thiessen()],
#'   [summary.thiessen()] and [sigma.thiessen()] to read it.
#'
#' @references
#' Chipman, H. A., George, E. I. and McCulloch, R. E. (2010). BART: Bayesian
#' additive regression trees. *The Annals of Applied Statistics* 4(1),
#' 266-298. \doi{10.1214/09-AOAS285}
#'
#' Stone, A. and Gosling, J. P. (2025). AddiVortes: (Bayesian) additive
#' Voronoi tessellations. *Journal of Computational and Graphical
#' Statistics* 34(3), 859-871. \doi{10.1080/10618600.2024.2414104}
#'
#' @examples
#' n <- 60
#' x <- cbind(seq(0, 1, length.out = n), rep(c(0, 0.5), length.out = n))
#' y <- 2 * (x[, 1] - 0.5)^2 + 0.5 * x[, 2]
#' fit <- thiessen(x, y, thiessen_control(m = 10, burn_in = 20, draws = 40),
#'                 seed = 1)
#' fit
#' @export
thiessen <- function(x, ...) {
  UseMethod("thiessen")
}

#' @rdname thiessen
#' @export
thiessen.default <- function(x, y, control = thiessen_control(), seed = NULL,
                             ...) {
  rlang::check_dots_empty()
  design <- as_design(x)
  if (!is.numeric(y) || !is.null(dim(y))) {
    thiessen_abort("`y` must be a numeric vector.")
  }
  if (anyNA(y)) {
    thiessen_abort("`y` must not contain missing values.")
  }
  if (length(y) != nrow(design)) {
    thiessen_abort(sprintf(
      "`x` has %d rows and `y` has %d values; they must agree.",
      nrow(design), length(y)
    ))
  }
  resolved <- resolve_seed(seed)
  fit <- core_call(core_fit(
    config_json(control), design, as.double(y), resolved
  ))
  for (warning in fit$warnings) {
    rlang::warn(warning, class = "thiessen_warning")
  }
  structure(
    list(
      state = fit$state,
      control = structure(
        jsonlite::fromJSON(fit$config, simplifyVector = FALSE),
        class = "thiessen_control"
      ),
      model = fit$model,
      n_draws = fit$n_draws,
      in_sample_rmse = fit$in_sample_rmse,
      warnings = fit$warnings,
      seed = resolved,
      n_features = ncol(design),
      x = design,
      y = as.double(y),
      fitted.values = fit$fitted_values,
      residuals = as.double(y) - fit$fitted_values,
      call = match.call()
    ),
    class = "thiessen"
  )
}
