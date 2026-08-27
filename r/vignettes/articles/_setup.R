# Shared setup for the precomputed experimental articles: the chunk
# options, the schedule, the seeds, two data generators, the paired fits
# and the scoring. Every article sources it in a visible chunk.

library(thiessen)
library(ggplot2)

options(mc.cores = 4)
theme_set(theme_minimal())
knitr::opts_chunk$set(
  collapse = TRUE, comment = "#>",
  fig.width = 7, fig.height = 4, dpi = 96, out.width = "100%"
)

schedule <- general_params(burn_in = 300, draws = 300)
seeds <- 1:5

# Friedman #1 (Friedman 1991): five informative covariates, p - 5 noise
# covariates, Gaussian noise of standard deviation `sd`. Training rows
# and a held-out set drawn from the same law, with the noise-free truth.
friedman <- function(n, p = 10, sd = 1, seed, n_test = 500) {
  set.seed(seed)
  draw <- function(n) {
    x <- matrix(runif(n * p), n, p,
                dimnames = list(NULL, paste0("x", seq_len(p))))
    f <- 10 * sin(pi * x[, 1] * x[, 2]) + 20 * (x[, 3] - 0.5)^2 +
      10 * x[, 4] + 5 * x[, 5]
    list(x = x, f = f, y = f + rnorm(n, sd = sd))
  }
  list(train = draw(n), test = draw(n_test))
}

# A smooth surface on the unit square, sin(pi x1) cos(pi x2), with
# Gaussian noise of standard deviation `sd`.
smooth_surface <- function(n, sd = 0.1, seed, n_test = 500) {
  set.seed(seed)
  draw <- function(n) {
    x <- cbind(x1 = runif(n), x2 = runif(n))
    f <- sin(pi * x[, "x1"]) * cos(pi * x[, "x2"])
    list(x = x, f = f, y = f + rnorm(n, sd = sd))
  }
  list(train = draw(n), test = draw(n_test))
}

# One fit per seed for each named control, on the training rows.
paired_fits <- function(data, controls, seeds) {
  lapply(controls, function(control) {
    lapply(seeds, function(seed) {
      thiessen(data$train$x, data$train$y, control, seed = seed)
    })
  })
}

# Held-out root mean squared error against the noise-free truth, the
# coverage and mean width of the central predictive interval, and the
# log score; the mean and standard error over the seeds.
score <- function(fits, data, level = 0.95) {
  one <- function(fit) {
    interval <- predict(fit, data$test$x, interval = "prediction",
                        level = level)
    log_likelihood <- log_lik(fit, newdata = data$test$x, y = data$test$y)
    c(rmse = sqrt(mean((interval[, "fit"] - data$test$f)^2)),
      coverage = mean(data$test$y >= interval[, "lower"] &
                        data$test$y <= interval[, "upper"]),
      width = mean(interval[, "upper"] - interval[, "lower"]),
      log_score = mean(log(colMeans(exp(log_likelihood)))))
  }
  rows <- lapply(names(fits), function(name) {
    scores <- t(vapply(fits[[name]], one, numeric(4)))
    data.frame(model = name, metric = colnames(scores),
               mean = colMeans(scores),
               se = apply(scores, 2, sd) / sqrt(nrow(scores)),
               row.names = NULL)
  })
  do.call(rbind, rows)
}
