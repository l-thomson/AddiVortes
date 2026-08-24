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
#' level as the zero, as `glm` treats one. An ordered factor is encoded as
#' an unordered one; the ordering is not used.
#'
#' Missing (`NA`) and non-finite values in the covariates or the response
#' are rejected with an error; no row is dropped silently.
#'
#' `stats::update()` works on a fit: the call is stored, so
#' `update(fit, seed = 2)` refits with that argument replaced.
#'
#' With `chains` of two or more, the chains are run in turn with the seeds
#' the core derives from `seed`, their draws are pooled, and the fit carries
#' rank-normalised split R-hat and the bulk and tail effective sample sizes
#' of sigma and of the mean function at up to twenty training rows
#' (`posterior::summarise_draws()`). A fit warns, and `print()` and
#' `summary()` repeat the warning, where R-hat exceeds 1.01 or an effective
#' sample size falls below 400 (Vehtari and others, 2021). A fit of one
#' chain says so instead.
#'
#' Progress over the sweep schedule is signalled with progressr, so a
#' session reports it after `progressr::handlers()` and nothing is printed
#' by default. The draws do not depend on whether a handler is set.
#'
#' A fit is a plain R object holding the sampler state, so [saveRDS()]
#' writes one and a later session reads it without a refit. A fit written by
#' a build with the core's `experimental` feature and read by a build
#' without it errors with the condition class `thiessen_error`, naming the
#' feature, at the first call that needs the state.
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
#' @param chains The number of chains to run, a whole number. Each chain
#'   has its own seed, derived from `seed` in the core, and the draws of the
#'   chains are pooled. Two or more chains give the convergence
#'   diagnostics; one chain does not.
#' @param seed The seed of the chain. `NULL`, the default, draws one from
#'   R's stream, so [set.seed()] governs; a whole number in `[0, 2^53]`
#'   passes to the core unchanged, so the same value reproduces the same
#'   draws for a given package version and platform.
#' @param ... Passed to the method.
#'
#' @return An object of class `"thiessen"`: a list with the fitted state,
#'   the resolved configuration, the number of chains and of kept draws, the
#'   convergence diagnostics where two or more chains ran, the seed used,
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
#' Vehtari, A., Gelman, A., Simpson, D., Carpenter, B. and Buerkner, P.-C.
#' (2021). Rank-normalization, folding, and localization: an improved R-hat
#' for assessing convergence of MCMC. *Bayesian Analysis* 16(2), 667-718.
#' \doi{10.1214/20-BA1221}
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
#' fit
#'
#' # Two chains add the convergence diagnostics; one chain reports none.
#' thiessen(x, y, control, seed = 1, chains = 2)
#'
#' frame <- data.frame(y = y, a = x[, 1], b = factor(x[, 2] > 0))
#' thiessen(y ~ a + b, frame, control, seed = 1)
#' @export
thiessen <- function(x, ...) {
  UseMethod("thiessen")
}

#' @rdname thiessen
#' @export
thiessen.default <- function(x, y, control = thiessen_control(), seed = NULL,
                             chains = 1, ...) {
  rlang::check_dots_empty()
  design <- as_design(x)
  response <- as_response(y)
  new_fit(design, response$y, control, seed, chains,
          generic_call(match.call()), response_levels = response$levels)
}

#' @rdname thiessen
#' @export
thiessen.data.frame <- function(x, y, control = thiessen_control(),
                                seed = NULL, chains = 1, ...) {
  rlang::check_dots_empty()
  molded <- core_call(
    hardhat::mold(~ ., data = x, blueprint = blueprint_for(control))
  )
  design <- encode_predictors(
    molded$predictors, molded$blueprint$indicators,
    control$mean_params$geometry$metric
  )
  response <- as_response(y)
  new_fit(design, response$y, control, seed, chains,
          generic_call(match.call()), blueprint = molded$blueprint,
          response_levels = response$levels)
}

#' @rdname thiessen
#' @export
thiessen.formula <- function(formula, data, control = thiessen_control(),
                             seed = NULL, chains = 1, ...) {
  rlang::check_dots_empty()
  molded <- core_call(
    hardhat::mold(formula, data, blueprint = blueprint_for(control))
  )
  design <- encode_predictors(
    molded$predictors, molded$blueprint$indicators,
    control$mean_params$geometry$metric
  )
  response <- encode_response(molded$outcomes)
  new_fit(design, response$y, control, seed, chains,
          generic_call(match.call()), blueprint = molded$blueprint,
          response_levels = response$levels)
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
#' @param chains The number of chains to run.
#' @param call The call to store.
#' @param blueprint The hardhat blueprint, or `NULL` for a matrix fit.
#' @param response_levels The response's factor levels, or `NULL`.
#' @param call_env The calling environment to report.
#' @return An object of class `"thiessen"`.
#' @noRd
new_fit <- function(design, y, control, seed, chains, call, blueprint = NULL,
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
  chains <- resolve_chains(chains, call = call_env)
  resolved <- resolve_seed(seed, call = call_env)
  fit <- core_call(
    run_schedule(control, design, y, resolved, chains),
    call = call_env
  )
  assemble_fit(fit, design, y, resolved, call,
               blueprint = blueprint, response_levels = response_levels,
               call_env = call_env)
}

#' Run the sweep schedule of every chain and pool the draws
#'
#' The schedule runs here rather than in the core because progressr reports
#' by signalling a condition, and extendr evaluates an R callback through
#' `R_tryEval`, which clears R's handler stack: a condition raised inside a
#' callback the core calls reaches no handler established outside the
#' `.Call`. Driving the sampler from R signals where the handlers are live.
#' The sweep order is the core's own, so the draws are unchanged.
#'
#' @param control An object of class `"thiessen_control"`.
#' @param design The numeric design.
#' @param y The numeric response.
#' @param seed The resolved seed.
#' @param chains The number of chains to run.
#' @return The list `core_finish()` returns.
#' @noRd
run_schedule <- function(control, design, y, seed, chains) {
  schedule <- control$general_params
  config <- config_json(control)
  thinning <- as.integer(schedule$thinning)
  sweeps <- schedule$burn_in + schedule$draws * thinning
  total <- chains * sweeps
  updates <- progress_updates(control, chains)
  report <- progressr::progressor(steps = updates)
  emitted <- 0L
  done <- 0L
  # `updates` reports spread evenly over the sweeps of every chain, and the
  # sweeps to the next of them, so burn-in advances in one call per report.
  gap <- function(completed) {
    ceiling((emitted + 1L) * total / updates) - done - completed
  }
  tick <- function(completed) {
    while (emitted < updates &&
             (emitted + 1L) * total <= (done + completed) * updates) {
      emitted <<- emitted + 1L
      report()
    }
  }
  samplers <- vector("list", chains)
  for (index in seq_len(chains)) {
    handle <- core_sampler_new(config, design, y, seed, index - 1L)
    samplers[[index]] <- handle
    completed <- 0L
    while (completed < schedule$burn_in) {
      run <- max(1L, min(schedule$burn_in - completed, gap(completed)))
      core_sampler_step(handle, as.integer(run))
      completed <- completed + run
      tick(completed)
    }
    for (draw in seq_len(schedule$draws)) {
      core_sampler_step(handle, thinning)
      completed <- completed + thinning
      core_sampler_keep(handle)
      tick(completed)
    }
    done <- done + sweeps
  }
  core_finish(samplers)
}

#' Assemble the object the methods return from the core's fit list
#'
#' @param fit The list `core_finish()` returns.
#' @param design The numeric design.
#' @param y The numeric response.
#' @param seed The resolved seed.
#' @param call The call to store.
#' @param blueprint The hardhat blueprint, or `NULL` for a matrix fit.
#' @param response_levels The response's factor levels, or `NULL`.
#' @param call_env The calling environment to report.
#' @return An object of class `"thiessen"`.
#' @noRd
assemble_fit <- function(fit, design, y, seed, call, blueprint = NULL,
                         response_levels = NULL,
                         call_env = rlang::caller_env()) {
  for (warning in fit$warnings) {
    rlang::warn(warning, class = "thiessen_warning")
  }
  fit_object <- structure(
    list(
      state = fit$state,
      control = control_from_config(
        jsonlite::fromJSON(fit$config, simplifyVector = FALSE)
      ),
      model = fit$model,
      n_chains = fit$n_chains,
      n_draws = fit$n_draws,
      in_sample_rmse = fit$in_sample_rmse,
      warnings = fit$warnings,
      seed = seed,
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
  fit_object$convergence <- convergence_of(fit_object)
  warn_convergence(fit_object, call = call_env)
  fit_object
}
