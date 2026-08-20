# Informational comparison of the probit model against the authors'
# Binary AddiVortes research script (Adam-Stone2/Binary_AddiVortes at
# commit ffc914b70d2105faf033af9e92f2e35dd18ba415). The script carries the
# structural-move terms that CRAN AddiVortes corrected in 0.6.8, has no
# offset and initialises the latent fit at the share of ones, so it targets
# a different posterior from the crate: the summaries are reported in the
# pull request, not asserted by a test.
#
# Usage, from benchmarks/upstream: Rscript binary_variant.R
#
# Writes under ../../target/variants: the dataset (covariate columns, then
# y) and one row per summary (mean of P(y = 1 | x) at fixed training rows,
# with its Monte Carlo standard error from the effective sample size). The
# row indices in the summary names are zero-based. The crate side and the
# comparison table come from the ignored test in
# crates/thiessen/tests/variants.rs:
#
#     cargo test --release --test variants -- --ignored --nocapture

library(coda)
library(truncnorm)
library(FNN)

commit <- "ffc914b70d2105faf033af9e92f2e35dd18ba415"
url <- sprintf(
  "https://raw.githubusercontent.com/Adam-Stone2/Binary_AddiVortes/%s/Binary_AddiVortes.R",
  commit
)
source_text <- readLines(url)
# Drop the package installation block before the first library() call and
# return the per-draw test probabilities alongside the script's own
# summaries.
source_text <- source_text[seq(grep("^library\\(", source_text)[1], length(source_text))]
source_text <- sub(
  "ProbTrain = mean_yhat,",
  "ProbTrain = mean_yhat, ProbTestDraws = probabilities_test,",
  source_text,
  fixed = TRUE
)
stopifnot(any(grepl("ProbTestDraws", source_text)))
eval(parse(text = source_text))

out_dir <- file.path("..", "..", "target", "variants")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Probit Friedman function: P(y = 1 | x) = Phi(f(x)), f the centred
# Friedman (1991) function scaled to unit standard deviation; n = 200,
# p = 5.
set.seed(42)
n <- 200
x <- matrix(runif(n * 5), n, 5)
raw <- 10 * sin(pi * x[, 1] * x[, 2]) + 20 * (x[, 3] - 0.5)^2 +
  10 * x[, 4] + 5 * x[, 5]
prob <- pnorm((raw - mean(raw)) / sd(raw))
y <- as.numeric(runif(n) < prob)
write.csv(data.frame(x, y = y), file.path(out_dir, "probit_friedman_data.csv"),
  row.names = FALSE, quote = FALSE
)

points <- c(1, 50, 100, 150, 200)
set.seed(7)
invisible(capture.output(
  fit <- AddiVortes_Algorithm(
    y, x,
    m = 50, max_iter = 1200, burn_in = 200, k = 3, var = 0.8, Omega = 3,
    lambda_rate = 5, YTest = y[points], XTest = x[points, , drop = FALSE]
  )
))

draws <- fit$ProbTestDraws
rows <- data.frame(summary = character(), value = numeric(), mcse = numeric())
for (i in seq_along(points)) {
  series <- draws[i, ]
  ess <- max(1, effectiveSize(series))
  rows[nrow(rows) + 1, ] <- list(
    sprintf("p_mean_r%d", points[i] - 1), mean(series), sd(series) / sqrt(ess)
  )
}
write.csv(rows, file.path(out_dir, "probit_friedman_script_summary.csv"),
  row.names = FALSE, quote = FALSE
)
cat("summaries written to", normalizePath(out_dir), "\n")
