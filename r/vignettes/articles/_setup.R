# Shared setup for the precomputed experimental articles: the chunk
# options, the schedule, the seeds, the data generators, the paired fits
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

# The Friedman #1 function (Friedman 1991) over the first five columns.
friedman_f <- function(x) {
  10 * sin(pi * x[, 1] * x[, 2]) + 20 * (x[, 3] - 0.5)^2 +
    10 * x[, 4] + 5 * x[, 5]
}

# Friedman #1: five informative covariates, p - 5 noise covariates, and
# Gaussian noise of standard deviation `sd`. A share `contaminate` of the
# training rows is moved by three noise standard deviations either way.
# Training rows and a held-out set drawn from the same law, with the
# noise-free truth.
friedman <- function(n, p = 10, sd = 1, seed, n_test = 500, contaminate = 0) {
  set.seed(seed)
  draw <- function(n) {
    x <- matrix(runif(n * p), n, p,
                dimnames = list(NULL, paste0("x", seq_len(p))))
    list(x = x, f = friedman_f(x), y = friedman_f(x) + rnorm(n, sd = sd))
  }
  data <- list(train = draw(n), test = draw(n_test))
  moved <- sample(n, round(contaminate * n))
  data$train$y[moved] <- data$train$y[moved] + 3 * sd * sample(c(-1, 1), length(moved), TRUE)
  data
}

# A surface on the unit square with Gaussian noise of standard deviation
# `sd`: sin(pi x1) cos(pi x2), or 2 x1 - x2 + 0.5 sin(2 pi x1) under
# `f = "linear"`.
smooth_surface <- function(n, sd = 0.1, seed, n_test = 500, f = c("sine", "linear")) {
  f <- match.arg(f)
  set.seed(seed)
  draw <- function(n) {
    x <- cbind(x1 = runif(n), x2 = runif(n))
    truth <- switch(f,
      sine = sin(pi * x[, "x1"]) * cos(pi * x[, "x2"]),
      linear = 2 * x[, "x1"] - x[, "x2"] + 0.5 * sin(2 * pi * x[, "x1"])
    )
    list(x = x, f = truth, y = truth + rnorm(n, sd = sd))
  }
  list(train = draw(n), test = draw(n_test))
}

# Lognormal accelerated failure times on Friedman #1 over p columns, log
# time = f / 10 + N(0, sd^2), with independent lognormal censoring times
# scaled so that about `censored` of the training rows are censored. The
# training response is a `survival::Surv()`; the held-out rows carry
# their true log times and are uncensored.
lognormal_friedman <- function(n, p = 5, sd = 0.5, seed, n_test = 500, censored = 1 / 3) {
  set.seed(seed)
  draw <- function(n) {
    x <- matrix(runif(n * p), n, p,
                dimnames = list(NULL, paste0("x", seq_len(p))))
    f <- friedman_f(cbind(x, matrix(0.5, n, max(0, 5 - p)))) / 10
    list(x = x, f = f, log_time = f + rnorm(n, sd = sd))
  }
  train <- draw(n)
  cutoff <- train$log_time + rnorm(n, mean = stats::qnorm(censored, lower.tail = FALSE) * sd, sd = sd)
  train$event <- as.numeric(train$log_time <= cutoff)
  train$time <- exp(pmin(train$log_time, cutoff))
  train$y <- survival::Surv(train$time, train$event)
  test <- draw(n_test)
  test$y <- exp(test$log_time)
  list(train = train, test = test)
}

# Ordered categories from a latent Friedman #1 surface over p columns:
# z = f / 10 + N(0, 1) cut at the latent quartiles into `categories`
# levels. The training response is an ordered factor; the held-out rows
# carry their true category probabilities.
ordinal_friedman <- function(n, p = 5, categories = 4, seed, n_test = 500) {
  set.seed(seed)
  draw <- function(n) {
    x <- matrix(runif(n * p), n, p,
                dimnames = list(NULL, paste0("x", seq_len(p))))
    list(x = x, f = friedman_f(cbind(x, matrix(0.5, n, max(0, 5 - p)))) / 10)
  }
  train <- draw(n)
  test <- draw(n_test)
  cutpoints <- quantile(train$f + rnorm(n), seq_len(categories - 1) / categories)
  categorise <- function(z) factor(findInterval(z, cutpoints), levels = 0:(categories - 1), ordered = TRUE)
  train$y <- categorise(train$f + rnorm(n))
  test$probs <- t(sapply(test$f, function(f) diff(c(0, pnorm(cutpoints - f), 1))))
  test$y <- categorise(test$f + rnorm(n_test))
  list(train = train, test = test, cutpoints = cutpoints)
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
