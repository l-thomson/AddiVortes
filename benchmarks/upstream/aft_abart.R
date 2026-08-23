# Informational comparison of the AFT model against BART::abart, the
# lognormal accelerated failure time model with trees in place of
# tessellations. The priors differ (tree splits against Voronoi
# tessellations; abart defaults k = 2, sigdf = 3, sigquant = 0.90 against
# the crate's k = 3, nu = 6, q = 0.85, matched here where the surface
# allows), so the posteriors are close but not equal: the summaries are
# reported in the pull request, not asserted by a test.
#
# Usage, from benchmarks/upstream: Rscript aft_abart.R
#
# Writes under ../../target/variants: the dataset (covariate columns,
# time, delta) and one row per summary (mean of yhat, the log-time f, at
# fixed training rows, and the mean of sigma, with Monte Carlo standard
# errors from the effective sample size). The row indices in the summary
# names are zero-based. The crate side and the comparison table come from
# the ignored test in crates/thiessen/tests/variants.rs:
#
#     cargo test --release --features experimental --test variants -- --ignored --nocapture

library(coda)
library(BART)

out_dir <- file.path("..", "..", "target", "variants")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Lognormal AFT Friedman function: log T = f(x) + e, f the centred
# Friedman (1991) function scaled to standard deviation 0.5, e ~
# N(0, 0.3^2); independent lognormal censoring calibrated to censor
# roughly a third of the rows. n = 200, p = 5.
set.seed(42)
n <- 200
x <- matrix(runif(n * 5), n, 5)
raw <- 10 * sin(pi * x[, 1] * x[, 2]) + 20 * (x[, 3] - 0.5)^2 +
  10 * x[, 4] + 5 * x[, 5]
f <- 0.5 * (raw - mean(raw)) / sd(raw)
log_t <- f + 0.3 * rnorm(n)
log_c <- quantile(log_t, 0.75) + 0.3 * rnorm(n)
delta <- as.numeric(log_t <= log_c)
times <- exp(pmin(log_t, log_c))
cat(sprintf("censored share: %.2f\n", mean(1 - delta)))
write.csv(data.frame(x, time = times, delta = delta),
  file.path(out_dir, "aft_friedman_data.csv"),
  row.names = FALSE, quote = FALSE
)

fit <- abart(x, times, delta,
  ntree = 50L, k = 3, nskip = 200L, ndpost = 1000L, seed = 99L
)

points <- c(1, 50, 100, 150, 200)
summaries <- data.frame(summary = character(), value = numeric(), mcse = numeric())
for (r in points) {
  draws <- fit$yhat.train[, r]
  summaries <- rbind(summaries, data.frame(
    summary = sprintf("f_mean_r%d", r - 1),
    value = mean(draws),
    mcse = sd(draws) / sqrt(effectiveSize(mcmc(draws)))
  ))
}
sigma <- fit$sigma[-seq_len(200)]
summaries <- rbind(summaries, data.frame(
  summary = "sigma_mean",
  value = mean(sigma),
  mcse = sd(sigma) / sqrt(effectiveSize(mcmc(sigma)))
))
write.csv(summaries, file.path(out_dir, "aft_friedman_script_summary.csv"),
  row.names = FALSE, quote = FALSE
)
print(summaries)
