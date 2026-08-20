# Informational comparison of the heteroscedastic model against the
# authors' H-AddiVortes research script
# (anonymous2738/Heteroscedastic_AddiVortes at commit
# 54168fd3c9cced8ebb6fbbf382918e03b9b98eee). The script carries the
# structural-move terms that CRAN AddiVortes corrected in 0.6.8, so it
# targets a different posterior from the crate: the summaries are reported
# in the pull request, not asserted by a test.
#
# Usage, from benchmarks/upstream: Rscript heteroscedastic_variant.R
#
# Writes under ../../target/variants: the dataset (covariate columns, then
# y) and one row per summary (posterior mean of f(x) and of s^2(x) at
# fixed training rows, with the Monte Carlo standard error from the
# effective sample size). The row indices in the summary names are
# zero-based. The crate side and the comparison table come from the
# ignored test in crates/thiessen/tests/variants.rs:
#
#     cargo test --release --test variants -- --ignored --nocapture

library(coda)
library(invgamma)
library(FNN)

commit <- "54168fd3c9cced8ebb6fbbf382918e03b9b98eee"
url <- sprintf(
  "https://raw.githubusercontent.com/anonymous2738/Heteroscedastic_AddiVortes/%s/Algorithm.R",
  commit
)
source_text <- readLines(url)
# Drop the package installation block and the energy dependency, which
# the script uses only for its predictive Q-Q plot; return the per-draw
# test values of f and s^2 before the script sorts them.
source_text <- source_text[seq(grep("^library\\(", source_text)[1], length(source_text))]
source_text <- source_text[!grepl("library\\('energy'\\)", source_text)]
source_text <- sub(
  "  LowerConfidenceTESTValue<-vector(length=length(mean_yhat_Test))",
  "  TestDraws<-TestMatrix\n  LowerConfidenceTESTValue<-vector(length=length(mean_yhat_Test))",
  source_text,
  fixed = TRUE
)
source_text <- sub(
  "      In_sample_RMSE = sqrt(mean((y-mean_yhat)^2)),",
  "      TestDraws = TestDraws, VarianceDraws = Sigma_squared_test, In_sample_RMSE = sqrt(mean((y-mean_yhat)^2)),",
  source_text,
  fixed = TRUE
)
stopifnot(sum(grepl("TestDraws", source_text)) == 2)
eval(parse(text = source_text))

out_dir <- file.path("..", "..", "target", "variants")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Heteroscedastic Friedman function: y = f(x) + s(x) e, f the centred
# Friedman (1991) function scaled to unit standard deviation,
# s(x) = 0.3 + 0.7 x_1; n = 200, p = 5.
set.seed(42)
n <- 200
x <- matrix(runif(n * 5), n, 5)
raw <- 10 * sin(pi * x[, 1] * x[, 2]) + 20 * (x[, 3] - 0.5)^2 +
  10 * x[, 4] + 5 * x[, 5]
f <- (raw - mean(raw)) / sd(raw)
s <- 0.3 + 0.7 * x[, 1]
y <- f + s * rnorm(n)
write.csv(data.frame(x, y = y), file.path(out_dir, "heteroscedastic_friedman_data.csv"),
  row.names = FALSE, quote = FALSE
)

points <- c(1, 50, 100, 150, 200)
set.seed(7)
invisible(capture.output(
  fit <- AddiVortes_Algorithm(
    y, x,
    m = 50, m_var = 20, max_iter = 1200, burn_in = 200, nu = 6, q = 0.85,
    k = 3, sd = 0.8, Omega = 3, lambda_rate = 5,
    YTest = y[points], XTest = x[points, , drop = FALSE], plot_qq = FALSE
  )
))

f_draws <- fit$TestDraws * (max(y) - min(y)) + (max(y) + min(y)) / 2
s2_draws <- fit$VarianceDraws
rows <- data.frame(summary = character(), value = numeric(), mcse = numeric())
summarise <- function(label, series) {
  ess <- max(1, effectiveSize(series))
  rows[nrow(rows) + 1, ] <<- list(label, mean(series), sd(series) / sqrt(ess))
}
for (i in seq_along(points)) {
  summarise(sprintf("f_mean_r%d", points[i] - 1), f_draws[i, ])
}
for (i in seq_along(points)) {
  summarise(sprintf("s2_mean_r%d", points[i] - 1), s2_draws[i, ])
}
write.csv(rows, file.path(out_dir, "heteroscedastic_friedman_script_summary.csv"),
  row.names = FALSE, quote = FALSE
)
cat("summaries written to", normalizePath(out_dir), "\n")
