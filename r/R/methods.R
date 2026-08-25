#' Posterior predictions from a fitted model
#'
#' @param object An object of class `"thiessen"`.
#' @param newdata New covariates. A fit from a formula or a data frame takes
#'   a data frame, whose columns are matched to the fitted design by name and
#'   type; a fit from a matrix takes a numeric matrix with the fitted
#'   columns. `NULL`, the default, predicts at the training rows.
#' @param type The quantity: `"mean"`, the posterior mean of the response
#'   (the probability under the probit model); `"draws"`, that quantity for
#'   every kept draw; `"latent"`, the mean function f for every kept draw;
#'   `"variance"`, the variance of y given f for every kept draw.
#' @param interval `"none"`, the default; `"credible"` for the interval of
#'   the posterior mean; `"prediction"` for the posterior predictive
#'   interval. Only with `type = "mean"`.
#' @param level The mass of a central interval. Default 0.95.
#' @param ... Ignored.
#'
#' @return For `type = "mean"` and `interval = "none"`, a numeric vector of
#'   length `nrow(newdata)`; with an interval, a matrix of that many rows
#'   with columns `fit`, `lower` and `upper`. For the other types, a matrix
#'   of one row per kept draw and one column per row of `newdata`.
#'
#' @examples
#' n <- 60
#' x <- cbind(seq(0, 1, length.out = n), rep(c(0, 0.5), length.out = n))
#' y <- 2 * (x[, 1] - 0.5)^2 + 0.5 * x[, 2]
#' control <- thiessen_control(
#'   tessellations = 10,
#'   general_params = general_params(burn_in = 20, draws = 40)
#' )
#' fit <- thiessen(x, y, control, seed = 1)
#' head(predict(fit))
#' head(predict(fit, interval = "credible"))
#' @importFrom stats predict
#' @export
predict.thiessen <- function(object, newdata = NULL,
                             type = c("mean", "draws", "latent", "variance"),
                             interval = c("none", "credible", "prediction"),
                             level = 0.95, ...) {
  type <- match.arg(type)
  interval <- match.arg(interval)
  design <- predict_design(object, newdata)
  if (interval != "none" && type != "mean") {
    thiessen_abort("An interval is available for `type = \"mean\"` only.")
  }
  if (!is.numeric(level) || length(level) != 1L || is.na(level) ||
        level <= 0 || level >= 1) {
    thiessen_abort("`level` must be a single number in (0, 1).")
  }
  state <- fit_state(object)
  if (type != "mean") {
    return(core_call(core_predict_draws(state, design, type)))
  }
  if (interval == "none") {
    return(core_call(core_predict(state, design)))
  }
  out <- core_call(core_predict_interval(state, design, interval, level))
  colnames(out) <- c("fit", "lower", "upper")
  out
}

#' Residual standard deviation of a fitted model
#'
#' @param object An object of class `"thiessen"`.
#' @param ... Ignored.
#'
#' @return A single number: the posterior mean of sigma under the Gaussian
#'   model, and 1 under the probit model, whose latent scale is fixed. The
#'   heteroscedastic model has no single residual scale; use
#'   `predict(type = "variance")`.
#'
#' @examples
#' n <- 60
#' x <- cbind(seq(0, 1, length.out = n), rep(c(0, 0.5), length.out = n))
#' y <- 2 * (x[, 1] - 0.5)^2 + 0.5 * x[, 2]
#' control <- thiessen_control(
#'   tessellations = 10,
#'   general_params = general_params(burn_in = 20, draws = 40)
#' )
#' sigma(thiessen(x, y, control, seed = 1))
#' @importFrom stats sigma
#' @export
sigma.thiessen <- function(object, ...) {
  if (object$model == "probit") {
    return(1)
  }
  draws <- core_call(core_sigma(fit_state(object)))
  if (length(draws) == 0L) {
    thiessen_abort(paste0(
      "The ", object$model, " model has no single residual scale; ",
      "use `predict(type = \"variance\")`."
    ))
  }
  mean(draws)
}

#' Fitted values, residuals and observation count of a fitted model
#'
#' @param object An object of class `"thiessen"`.
#' @param ... Ignored.
#'
#' @return For `fitted()` and `residuals()`, a numeric vector of length
#'   `nobs(object)`: the posterior mean of the response at each training row,
#'   and the response less that mean. For `nobs()`, the number of
#'   observations.
#'
#' @examples
#' n <- 60
#' x <- cbind(seq(0, 1, length.out = n), rep(c(0, 0.5), length.out = n))
#' y <- 2 * (x[, 1] - 0.5)^2 + 0.5 * x[, 2]
#' control <- thiessen_control(
#'   tessellations = 10,
#'   general_params = general_params(burn_in = 20, draws = 40)
#' )
#' fit <- thiessen(x, y, control, seed = 1)
#' nobs(fit)
#' head(residuals(fit))
#' @importFrom stats fitted
#' @export
fitted.thiessen <- function(object, ...) {
  object$fitted.values
}

#' @rdname fitted.thiessen
#' @importFrom stats residuals
#' @export
residuals.thiessen <- function(object, ...) {
  object$residuals
}

#' @rdname fitted.thiessen
#' @importFrom stats nobs
#' @export
nobs.thiessen <- function(object, ...) {
  length(object$y)
}

#' Print a fitted model
#'
#' @param x An object of class `"thiessen"`.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @examples
#' n <- 60
#' x <- cbind(seq(0, 1, length.out = n), rep(c(0, 0.5), length.out = n))
#' y <- 2 * (x[, 1] - 0.5)^2 + 0.5 * x[, 2]
#' control <- thiessen_control(
#'   tessellations = 10,
#'   general_params = general_params(burn_in = 20, draws = 40)
#' )
#' print(thiessen(x, y, control, seed = 1))
#' @export
print.thiessen <- function(x, ...) {
  cat("AddiVortes fit\n")
  cat("Call: ", paste(deparse(x$call), collapse = " "), "\n", sep = "")
  cat(sprintf(
    "%s model, %d observations, %d covariates\n",
    x$model, nobs(x), x$n_features
  ))
  cat(sprintf(
    "%d tessellations, %d draws kept after %d burn-in, thinning %d\n",
    x$control$mean_params$tessellations, x$n_draws,
    x$control$general_params$burn_in, x$control$general_params$thinning
  ))
  cat(sprintf("In-sample RMSE %.4g, seed %.0f\n", x$in_sample_rmse, x$seed))
  cat(convergence_line(x), "\n", sep = "")
  message <- convergence_message(x$convergence)
  if (!is.null(message)) {
    cat("Warning: ", message, "\n", sep = "")
  }
  invisible(x)
}

#' Summarise a fitted model
#'
#' @param object An object of class `"thiessen"`.
#' @param ... Ignored.
#'
#' @return An object of class `"summary.thiessen"`: a list of the model, the
#'   dimensions of the fit, the sweep schedule, the in-sample root mean
#'   squared error, the quantiles of the residuals, the quantiles of the
#'   posterior draws of sigma where the model has one, and the convergence
#'   diagnostics where two or more chains ran.
#'
#' @examples
#' n <- 60
#' x <- cbind(seq(0, 1, length.out = n), rep(c(0, 0.5), length.out = n))
#' y <- 2 * (x[, 1] - 0.5)^2 + 0.5 * x[, 2]
#' control <- thiessen_control(
#'   tessellations = 10,
#'   general_params = general_params(burn_in = 20, draws = 40)
#' )
#' summary(thiessen(x, y, control, seed = 1))
#' @importFrom stats quantile
#' @export
summary.thiessen <- function(object, ...) {
  probs <- c(0.025, 0.25, 0.5, 0.75, 0.975)
  draws <- core_call(core_sigma(fit_state(object)))
  structure(
    list(
      call = object$call,
      model = object$model,
      nobs = nobs(object),
      n_features = object$n_features,
      control = object$control,
      n_chains = object$n_chains,
      n_draws = object$n_draws,
      convergence = object$convergence,
      in_sample_rmse = object$in_sample_rmse,
      residuals = quantile(object$residuals, probs),
      sigma = if (length(draws) > 0L) quantile(draws, probs) else NULL,
      warnings = object$warnings
    ),
    class = "summary.thiessen"
  )
}

#' @param x An object of class `"summary.thiessen"`.
#' @return `x`, invisibly.
#' @rdname summary.thiessen
#' @export
print.summary.thiessen <- function(x, ...) {
  cat("AddiVortes fit\n")
  cat("Call: ", paste(deparse(x$call), collapse = " "), "\n", sep = "")
  cat(sprintf(
    "%s model, %d observations, %d covariates\n",
    x$model, x$nobs, x$n_features
  ))
  cat(sprintf(
    "%d tessellations, %d draws kept after %d burn-in, thinning %d\n",
    x$control$mean_params$tessellations, x$n_draws,
    x$control$general_params$burn_in, x$control$general_params$thinning
  ))
  cat("\nResiduals:\n")
  print(x$residuals)
  if (!is.null(x$sigma)) {
    cat("\nsigma:\n")
    print(x$sigma)
  }
  cat(sprintf("\nIn-sample RMSE %.4g\n", x$in_sample_rmse))
  cat(convergence_line(x), "\n", sep = "")
  message <- convergence_message(x$convergence)
  if (!is.null(message)) {
    cat("Warning: ", message, "\n", sep = "")
  }
  for (warning in x$warnings) {
    cat("Warning: ", warning, "\n", sep = "")
  }
  invisible(x)
}
