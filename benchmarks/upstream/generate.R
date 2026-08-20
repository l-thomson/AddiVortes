# Regenerates the upstream comparison fixtures under
# crates/thiessen/tests/fixtures/upstream from CRAN AddiVortes, pinned by
# renv.lock. Fixtures change only through this script; the pull request
# that regenerates them states why.
#
# Usage, from benchmarks/upstream: Rscript generate.R
#
# Per dataset: the data as CSV (covariate columns, then y), and a summary
# CSV with one row per posterior summary (mean and sd of f at fixed
# training rows; sigma mean and quantiles 0.05, 0.5, 0.95), each with its
# Monte Carlo standard error from the effective sample size. The row
# indices in the summary names are zero-based.

library(AddiVortes)
library(coda)

fixture_dir <- file.path("..", "..", "crates", "thiessen", "tests", "fixtures", "upstream")
dir.create(fixture_dir, recursive = TRUE, showWarnings = FALSE)

# Per-draw f at the given rows, caller scale: the loop of
# predict.AddiVortes without the summarising, through the exported
# cellIndices.
f_draws_at <- function(fit, x_new) {
  x_scaled <- sweep(sweep(as.matrix(x_new), 2, fit$xCentres), 2, fit$xRanges, "/")
  draws <- length(fit$posteriorTess)
  out <- matrix(0, draws, nrow(x_new))
  for (s in seq_len(draws)) {
    total <- numeric(nrow(x_new))
    for (j in seq_along(fit$posteriorTess[[s]])) {
      idx <- cellIndices(
        x_scaled, fit$posteriorTess[[s]][[j]], fit$posteriorDim[[s]][[j]],
        fit$metric_red, fit$member_red
      )
      total <- total + fit$posteriorPred[[s]][[j]][idx]
    }
    out[s, ] <- total * fit$yRange + fit$yCentre
  }
  out
}

# MCSE of a type 7 quantile: sqrt(p (1 - p) / ESS) over the density at the
# quantile, the density by the central difference of the quantile function
# with half-width h.
quantile_mcse <- function(series, p, h = 0.025) {
  ess <- max(1, effectiveSize(series))
  qs <- quantile(series, c(p - h, p + h), type = 7, names = FALSE)
  sqrt(p * (1 - p) / ess) * max(qs[2] - qs[1], 1e-12) / (2 * h)
}

summarise_fit <- function(fit, x, points, name) {
  f <- f_draws_at(fit, x[points, , drop = FALSE])
  sigma <- sqrt(as.numeric(fit$posteriorSigma)) * fit$yRange
  rows <- data.frame(summary = character(), value = numeric(), mcse = numeric())
  add <- function(summary, value, mcse) {
    rows[nrow(rows) + 1, ] <<- list(summary, value, mcse)
  }
  for (i in seq_along(points)) {
    series <- f[, i]
    ess <- max(1, effectiveSize(series))
    add(sprintf("f_mean_r%d", points[i] - 1), mean(series), sd(series) / sqrt(ess))
    add(sprintf("f_sd_r%d", points[i] - 1), sd(series), sd(series) / sqrt(2 * ess))
  }
  ess_sigma <- max(1, effectiveSize(sigma))
  add("sigma_mean", mean(sigma), sd(sigma) / sqrt(ess_sigma))
  for (p in c(0.05, 0.5, 0.95)) {
    add(
      sprintf("sigma_q%02d", round(100 * p)),
      quantile(sigma, p, type = 7, names = FALSE),
      quantile_mcse(sigma, p)
    )
  }
  write.csv(rows, file.path(fixture_dir, sprintf("%s_summary.csv", name)),
    row.names = FALSE, quote = FALSE
  )
}

# Friedman (1991) benchmark: n = 200, p = 10, sigma = 1.
set.seed(42)
n <- 200
x <- matrix(runif(n * 10), n, 10)
y <- 10 * sin(pi * x[, 1] * x[, 2]) + 20 * (x[, 3] - 0.5)^2 +
  10 * x[, 4] + 5 * x[, 5] + rnorm(n)
write.csv(data.frame(x, y = y), file.path(fixture_dir, "friedman_data.csv"),
  row.names = FALSE, quote = FALSE
)
set.seed(7)
fit <- AddiVortes(y, x, m = 200, totalMCMCIter = 1200, mcmcBurnIn = 200,
  showProgress = FALSE
)
summarise_fit(fit, x, c(1, 50, 100, 150, 200), "friedman")

# attitude (base R): n = 30, p = 6, all numeric.
x <- as.matrix(attitude[, -1])
y <- attitude$rating
write.csv(data.frame(x, y = y), file.path(fixture_dir, "attitude_data.csv"),
  row.names = FALSE, quote = FALSE
)
set.seed(8)
fit <- AddiVortes(y, x, m = 200, totalMCMCIter = 1200, mcmcBurnIn = 200,
  showProgress = FALSE
)
summarise_fit(fit, x, c(1, 8, 15, 22, 30), "attitude")

cat("fixtures written to", normalizePath(fixture_dir), "\n")
