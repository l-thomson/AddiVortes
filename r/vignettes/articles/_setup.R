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

# A surface on the unit square: sin(pi x1) cos(pi x2), or
# 2 x1 - x2 + 0.5 sin(2 pi x1) under `f = "linear"`.
surface_f <- function(x, f = c("sine", "linear")) {
  f <- match.arg(f)
  switch(f,
    sine = sin(pi * x[, "x1"]) * cos(pi * x[, "x2"]),
    linear = 2 * x[, "x1"] - x[, "x2"] + 0.5 * sin(2 * pi * x[, "x1"])
  )
}

# The surface observed with Gaussian noise of standard deviation `sd`.
smooth_surface <- function(n, sd = 0.1, seed, n_test = 500, f = c("sine", "linear")) {
  f <- match.arg(f)
  set.seed(seed)
  draw <- function(n) {
    x <- cbind(x1 = runif(n), x2 = runif(n))
    truth <- surface_f(x, f)
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

# The predictive distribution of a fit at the rows of `x` as a mixture of
# normals with equal weights: a draws by rows matrix of the mean function
# and one of the residual standard deviation. Every method the articles
# score is reduced to this pair, so one scorer serves them all.
predictive <- function(fit, x) UseMethod("predictive")

predictive.thiessen <- function(fit, x) {
  list(mean = predict(fit, x, type = "draws"),
       sd = sqrt(predict(fit, x, type = "variance")))
}

# SoftBart's predict method wants the response column in the new data.
predictive.softbart_regression <- function(fit, x) {
  newdata <- as.data.frame(x)
  newdata[[all.vars(fit$formula)[1]]] <- 0
  mean <- predict(fit, newdata)$mu
  list(mean = mean, sd = matrix(fit$sigma, nrow(mean), ncol(mean)))
}

# The smoothing uncertainty of a GAM enters through draws of the
# coefficients from their Gaussian approximation (Wood 2017, section
# 6.10); the residual scale is the fitted one.
predictive.gam <- function(fit, x, draws = 400, seed = 1) {
  set.seed(seed)
  design <- predict(fit, as.data.frame(x), type = "lpmatrix")
  beta <- mgcv::rmvn(draws, coef(fit), vcov(fit))
  mean <- beta %*% t(design)
  list(mean = mean, sd = matrix(sqrt(fit$sig2), nrow(mean), ncol(mean)))
}

# The quantile of each row's mixture, by bisection on the mixture CDF.
mixture_quantile <- function(p, m, s) {
  lower <- apply(m - 8 * s, 1, min)
  upper <- apply(m + 8 * s, 1, max)
  for (step in seq_len(60)) {
    mid <- (lower + upper) / 2
    below <- rowMeans(pnorm(mid, m, s)) < p
    lower[below] <- mid[below]
    upper[!below] <- mid[!below]
  }
  (lower + upper) / 2
}

# Held-out scores of one predictive distribution: the root mean squared
# error of the posterior mean against the noise-free truth, the coverage
# and mean width of the central predictive interval, and the CRPS and the
# log score of the response, exact for the mixture (scoringRules). The
# mixture is thinned to at most 400 draws, since the CRPS of a mixture
# costs the square of its size.
score_one <- function(p, data, level = 0.95) {
  keep <- unique(round(seq(1, nrow(p$mean), length.out = min(nrow(p$mean), 400))))
  m <- t(p$mean[keep, , drop = FALSE])
  s <- t(p$sd[keep, , drop = FALSE])
  y <- data$test$y
  lower <- mixture_quantile((1 - level) / 2, m, s)
  upper <- mixture_quantile((1 + level) / 2, m, s)
  c(rmse = sqrt(mean((colMeans(p$mean) - data$test$f)^2)),
    coverage = mean(y >= lower & y <= upper),
    width = mean(upper - lower),
    crps = mean(scoringRules::crps_mixnorm(y, m, s)),
    log_score = -mean(scoringRules::logs_mixnorm(y, m, s)))
}

# One row per fit and metric for every named list of fits. A method with
# no sampling variation is passed as a list of one fit.
score <- function(fits, data, level = 0.95) {
  rows <- lapply(names(fits), function(model) {
    per_fit <- lapply(seq_along(fits[[model]]), function(i) {
      scores <- score_one(predictive(fits[[model]][[i]], data$test$x), data, level)
      data.frame(model = model, fit = i, metric = names(scores), value = unname(scores))
    })
    do.call(rbind, per_fit)
  })
  scores <- do.call(rbind, rows)
  scores$model <- factor(scores$model, names(fits))
  scores$metric <- factor(scores$metric, c("rmse", "coverage", "width", "crps", "log_score"))
  scores
}

# The mean over fits with its standard error in parentheses, one row per
# model; a model with one fit shows the value alone.
score_table <- function(scores) {
  cell <- function(value) {
    if (length(value) == 1) return(sprintf("%.3f", value))
    sprintf("%.3f (%.3f)", mean(value), sd(value) / sqrt(length(value)))
  }
  wide <- tapply(scores$value, scores[c("model", "metric")], cell)
  table <- data.frame(model = rownames(wide), as.data.frame.matrix(wide), row.names = NULL)
  names(table) <- c("model", "RMSE", "coverage", "width", "CRPS", "log score")
  table
}

# Held-out root mean squared error of one fit against the truth.
rmse <- function(fit, data) {
  sqrt(mean((colMeans(predictive(fit, data$test$x)$mean) - data$test$f)^2))
}
