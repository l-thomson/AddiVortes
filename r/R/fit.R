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
#' A factor covariate becomes d - 1 treatment-contrast indicators, the first
#' level as reference, as `model.matrix` and upstream AddiVortes encode it.
#' Where `control` declares a `metric`, one entry per column, factors are
#' passed as integer level codes instead and each factor column must declare
#' `"categorical"`.
#'
#' A factor response must have two levels and becomes 0 and 1 with the first
#' level as the zero, as `glm` treats one.
#'
#' `stats::update()` works on a fit: the call is stored, so
#' `update(fit, seed = 2)` refits with that argument replaced.
#'
#' @param x A numeric matrix of covariates, one row per observation, or a
#'   data frame. A numeric vector is taken as one column.
#' @param formula A two-sided formula. The left side names the response and
#'   the right side the covariates, `.` for every remaining column.
#' @param data A data frame holding the columns the formula names.
#' @param y The response: a numeric vector of length `nrow(x)`, or a
#'   two-level factor. Under the probit model the values must be 0 and 1.
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
#'   the design, the response, the fitted values, the residuals, the
#'   hardhat blueprint where one applies, and the call.
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
#'
#' frame <- data.frame(y = y, a = x[, 1], b = factor(x[, 2] > 0))
#' thiessen(y ~ a + b, frame, thiessen_control(m = 10, burn_in = 20,
#'                                             draws = 40), seed = 1)
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
  response <- as_response(y)
  new_fit(design, response$y, control, seed, generic_call(match.call()),
          response_levels = response$levels)
}

#' @rdname thiessen
#' @export
thiessen.data.frame <- function(x, y, control = thiessen_control(),
                                seed = NULL, ...) {
  rlang::check_dots_empty()
  molded <- core_call(
    hardhat::mold(~ ., data = x, blueprint = blueprint_for(control))
  )
  design <- encode_predictors(
    molded$predictors, molded$blueprint$indicators, control$metric
  )
  response <- as_response(y)
  new_fit(design, response$y, control, seed, generic_call(match.call()),
          blueprint = molded$blueprint, response_levels = response$levels)
}

#' @rdname thiessen
#' @export
thiessen.formula <- function(formula, data, control = thiessen_control(),
                             seed = NULL, ...) {
  rlang::check_dots_empty()
  molded <- core_call(
    hardhat::mold(formula, data, blueprint = blueprint_for(control))
  )
  design <- encode_predictors(
    molded$predictors, molded$blueprint$indicators, control$metric
  )
  response <- encode_response(molded$outcomes)
  new_fit(design, response$y, control, seed, generic_call(match.call()),
          blueprint = molded$blueprint, response_levels = response$levels)
}

#' Coerce a response to the numeric vector the core takes
#'
#' @param y A numeric vector or a two-level factor.
#' @param call The calling environment to report.
#' @return A list of the response and, for a factor, its levels.
#' @noRd
as_response <- function(y, call = rlang::caller_env()) {
  if (is.factor(y)) {
    return(encode_response(data.frame(y = y), call = call))
  }
  if (!is.numeric(y) || !is.null(dim(y))) {
    thiessen_abort("`y` must be a numeric vector or a two-level factor.",
                   call = call)
  }
  if (anyNA(y)) {
    thiessen_abort("`y` must not contain missing values.", call = call)
  }
  list(y = as.double(y), levels = NULL)
}

#' The method call with the generic's name, so `update()` can re-evaluate it
#'
#' @param call The method's `match.call()`.
#' @return A call to `thiessen`.
#' @noRd
generic_call <- function(call) {
  call[[1L]] <- as.name("thiessen")
  call
}

#' Fit the core and assemble the object the methods return
#'
#' @param design The numeric design.
#' @param y The numeric response.
#' @param control An object of class `"thiessen_control"`.
#' @param seed The seed as the caller gave it.
#' @param call The call to store.
#' @param blueprint The hardhat blueprint, or `NULL` for a matrix fit.
#' @param response_levels The response's factor levels, or `NULL`.
#' @param call_env The calling environment to report.
#' @return An object of class `"thiessen"`.
#' @noRd
new_fit <- function(design, y, control, seed, call, blueprint = NULL,
                    response_levels = NULL, call_env = rlang::caller_env()) {
  if (length(y) != nrow(design)) {
    thiessen_abort(
      sprintf(
        "The design has %d rows and the response has %d values; they must agree.",
        nrow(design), length(y)
      ),
      call = call_env
    )
  }
  resolved <- resolve_seed(seed, call = call_env)
  fit <- core_call(
    core_fit(config_json(control), design, y, resolved),
    call = call_env
  )
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
      blueprint = blueprint,
      response_levels = response_levels,
      x = design,
      y = y,
      fitted.values = fit$fitted_values,
      residuals = y - fit$fitted_values,
      call = call
    ),
    class = "thiessen"
  )
}
