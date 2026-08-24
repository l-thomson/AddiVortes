# The default plot method: traces of the per-draw sampler diagnostics.

#' Trace plots of a fitted model
#'
#' The per-draw quantities of [thiessen_diagnostics()] as traces, one panel
#' per quantity and one line per chain: `sigma` where the model has one, the
#' mean cells per mean tessellation and the mean active covariates per mean
#' tessellation. Burn-in sweeps are discarded before the first draw is kept,
#' so a trace shows the kept draws only.
#'
#' The covariate trace is flat by construction where p is 3 or fewer: the
#' default `omega` of `min(3, p)` makes `omega / p` equal to 1, so every
#' tessellation holds every covariate on every draw and the count cannot
#' move. It is the same property that makes [variable_inclusion()] uniform
#' at small p, and it is not a stalled chain.
#'
#' For traces of the mean function and for distributional displays, pass
#' [posterior::as_draws_df()] to bayesplot: `mcmc_trace()` plots sequences,
#' `mcmc_areas()` and `mcmc_dens()` plot posterior densities, and
#' `mcmc_combo()` plots both in one figure.
#'
#' @param x An object of class `"thiessen"`.
#' @param ... Ignored.
#'
#' @return `x`, invisibly.
#'
#' @examples
#' n <- 60
#' x <- cbind(seq(0, 1, length.out = n), rep(c(0, 0.5), length.out = n))
#' y <- 2 * (x[, 1] - 0.5)^2 + 0.5 * x[, 2]
#' control <- thiessen_control(
#'   tessellations = 10,
#'   general_params = general_params(burn_in = 20, draws = 40)
#' )
#' plot(thiessen(x, y, control, seed = 1))
#' @importFrom graphics plot
#' @export
plot.thiessen <- function(x, ...) {
  trace <- thiessen_diagnostics(x)
  quantities <- setdiff(names(trace), c("chain", "draw"))
  previous <- graphics::par(
    mfrow = c(length(quantities), 1L),
    mar = c(4, 4, 1, 1) + 0.1
  )
  on.exit(graphics::par(previous))
  iterations <- max(trace$draw)
  for (quantity in quantities) {
    graphics::matplot(
      seq_len(iterations),
      matrix(trace[[quantity]], nrow = iterations),
      type = "l", lty = 1L,
      xlab = "kept draw", ylab = quantity
    )
  }
  invisible(x)
}
