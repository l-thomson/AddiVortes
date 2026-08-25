# Methods on the rstantools generics, re-exported so a caller needs no
# library(rstantools), as rstanarm and brms do.

#' @importFrom rstantools posterior_predict
#' @export
rstantools::posterior_predict

#' @importFrom rstantools posterior_epred
#' @export
rstantools::posterior_epred

#' @importFrom rstantools log_lik
#' @export
rstantools::log_lik

#' @importFrom rstantools predictive_interval
#' @export
rstantools::predictive_interval

#' Draw from the posterior predictive distribution
#'
#' One replicate per kept draw: the mean function of that draw plus a
#' residual under the model, Bernoulli labels under the probit model. The
#' residuals are drawn from R's stream, so [set.seed()] governs them; they
#' are not part of the chain the core draws.
#'
#' @param object An object of class `"thiessen"`.
#' @param newdata New covariates, as [predict.thiessen()] takes them.
#'   `NULL`, the default, is the training rows.
#' @param ... Ignored.
#'
#' @return A double matrix, one row per kept draw and one column per row of
#'   `newdata`.
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
#' dim(posterior_predict(fit))
#' @exportS3Method rstantools::posterior_predict
posterior_predict.thiessen <- function(object, newdata = NULL, ...) {
  design <- predict_design(object, newdata)
  state <- fit_state(object)
  draws <- core_call(core_predict_draws(state, design, "draws"))
  if (object$model == "probit") {
    labels <- stats::rbinom(length(draws), 1L, pmin(pmax(draws, 0), 1))
    return(matrix(as.double(labels), nrow = nrow(draws)))
  }
  variance <- core_call(core_predict_draws(state, design, "variance"))
  noise <- stats::rnorm(length(draws), 0, sqrt(as.vector(variance)))
  draws + matrix(noise, nrow = nrow(draws))
}

#' Draw from the posterior distribution of the expected response
#'
#' @param object An object of class `"thiessen"`.
#' @param newdata New covariates, as [predict.thiessen()] takes them.
#'   `NULL`, the default, is the training rows.
#' @param ... Ignored.
#'
#' @return A double matrix, one row per kept draw and one column per row of
#'   `newdata`: the mean of the response, the probability under the probit
#'   model.
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
#' dim(posterior_epred(fit))
#' @exportS3Method rstantools::posterior_epred
posterior_epred.thiessen <- function(object, newdata = NULL, ...) {
  design <- predict_design(object, newdata)
  core_call(core_predict_draws(fit_state(object), design, "draws"))
}

#' Pointwise log-likelihood of a fitted model
#'
#' The matrix `loo::loo()` takes.
#'
#' @param object An object of class `"thiessen"`.
#' @param newdata New covariates, as [predict.thiessen()] takes them.
#'   `NULL`, the default, is the training rows.
#' @param y The response at `newdata`. Taken from `newdata` where the fit
#'   came from a formula and `newdata` carries the response column.
#' @param ... Ignored.
#'
#' @return A double matrix, one row per kept draw and one column per
#'   observation.
#'
#' @examples
#' if (requireNamespace("loo", quietly = TRUE)) {
#'   n <- 60
#'   x <- cbind(seq(0, 1, length.out = n), rep(c(0, 0.5), length.out = n))
#'   y <- 2 * (x[, 1] - 0.5)^2 + 0.5 * x[, 2]
#'   control <- thiessen_control(
#'     tessellations = 10,
#'     general_params = general_params(burn_in = 20, draws = 40)
#'   )
#'   fit <- thiessen(x, y, control, seed = 1)
#'   loo::loo(log_lik(fit))
#' }
#' @exportS3Method rstantools::log_lik
log_lik.thiessen <- function(object, newdata = NULL, y = NULL, ...) {
  if (is.null(newdata)) {
    return(core_call(core_log_lik(fit_state(object), object$x, object$y)))
  }
  design <- predict_design(object, newdata)
  response <- log_lik_response(object, newdata, y)
  if (length(response) != nrow(design)) {
    thiessen_abort(
      "`y` must have one value per row of `newdata`."
    )
  }
  core_call(core_log_lik(fit_state(object), design, response))
}

#' Central posterior predictive interval
#'
#' @param object An object of class `"thiessen"`.
#' @param prob The mass of the interval. Default 0.9.
#' @param newdata New covariates, as [predict.thiessen()] takes them.
#'   `NULL`, the default, is the training rows.
#' @param ... Ignored.
#'
#' @return A double matrix of one row per row of `newdata`, with the lower
#'   and upper bounds as columns named by their percentiles.
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
#' head(predictive_interval(fit))
#' @exportS3Method rstantools::predictive_interval
predictive_interval.thiessen <- function(object, prob = 0.9, newdata = NULL,
                                         ...) {
  if (!is.numeric(prob) || length(prob) != 1L || is.na(prob) ||
        prob <= 0 || prob >= 1) {
    thiessen_abort("`prob` must be a single number in (0, 1).")
  }
  design <- predict_design(object, newdata)
  bounds <- core_call(
    core_interval(fit_state(object), design, "prediction", prob)
  )
  tail <- (1 - prob) / 2
  colnames(bounds) <- paste0(
    format(100 * c(tail, 1 - tail), trim = TRUE), "%"
  )
  bounds
}

#' The design a prediction method should use
#'
#' @param object An object of class `"thiessen"`.
#' @param newdata `NULL` for the training rows, otherwise new covariates.
#' @param call The calling environment to report.
#' @return A double matrix.
#' @noRd
predict_design <- function(object, newdata, call = rlang::caller_env()) {
  if (is.null(newdata)) {
    return(object$x)
  }
  if (is.null(object$blueprint)) {
    return(as_design(newdata, "newdata", call = call))
  }
  forge_design(object, newdata, call = call)
}

#' The response at `newdata` for the log-likelihood
#'
#' @param object An object of class `"thiessen"`.
#' @param newdata New covariates.
#' @param y The response as the caller gave it, or `NULL`.
#' @param call The calling environment to report.
#' @return A double vector.
#' @noRd
log_lik_response <- function(object, newdata, y, call = rlang::caller_env()) {
  if (!is.null(y)) {
    return(as_response(y, call = call)$y)
  }
  if (is.null(object$blueprint)) {
    thiessen_abort(
      "`y` is required: the fit came from a matrix, so `newdata` carries no response.",
      call = call
    )
  }
  forged <- core_call(
    hardhat::forge(newdata, object$blueprint, outcomes = TRUE),
    call = call
  )
  encode_response(forged$outcomes, call = call)$y
}
