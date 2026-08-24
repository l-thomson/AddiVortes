gaussian_fit <- function() {
  fixture <- small_fixture()
  thiessen(fixture$x, fixture$y, small_control(), seed = 1)
}

probit_fit <- function() {
  fixture <- small_fixture()
  labels <- as.double(fixture$y >= stats::median(fixture$y))
  thiessen(
    fixture$x, labels, small_control(outcome = probit_outcome()), seed = 1
  )
}

test_that("the draws carry the mean function, sigma and the counts", {
  skip_if_not_installed("posterior")
  fit <- gaussian_fit()

  draws <- posterior::as_draws_df(fit)

  expect_s3_class(draws, "draws_df")
  expect_identical(posterior::ndraws(draws), 20L)
  expect_true(all(
    c("mu[1]", "mu[40]", "sigma", "cell_count", "dimension_count") %in%
      posterior::variables(draws)
  ))
})

test_that("the probit model exposes no sigma variable", {
  skip_if_not_installed("posterior")

  draws <- posterior::as_draws_df(probit_fit())

  expect_false("sigma" %in% posterior::variables(draws))
})

test_that("the heteroscedastic model exposes no sigma variable", {
  skip_if_not_installed("posterior")
  fixture <- small_fixture()
  fit <- thiessen(
    fixture$x, fixture$y,
    small_control(variance_params = term_params(tessellations = 4)), seed = 1
  )

  expect_false("sigma" %in% posterior::variables(posterior::as_draws_df(fit)))
})

test_that("the sigma draws are the core's", {
  skip_if_not_installed("posterior")
  fit <- gaussian_fit()

  draws <- posterior::as_draws_df(fit)

  expect_identical(as.double(draws$sigma), core_sigma(fit$state))
})

test_that("the draws array has one chain", {
  skip_if_not_installed("posterior")
  fit <- gaussian_fit()

  array <- posterior::as_draws_array(fit)

  expect_s3_class(array, "draws_array")
  expect_identical(posterior::nchains(array), 1L)
  expect_identical(posterior::nchains(fit), 1L)
  expect_identical(posterior::ndraws(fit), 20L)
})

test_that("summarise_draws reports on the fit", {
  skip_if_not_installed("posterior")
  fit <- gaussian_fit()

  summary <- posterior::summarise_draws(
    posterior::as_draws_df(fit), "mean", "sd"
  )

  expect_true("sigma" %in% summary$variable)
  expect_true(all(summary$sd >= 0))
})

test_that("posterior_epred is the per-draw mean", {
  fit <- gaussian_fit()

  epred <- posterior_epred(fit)

  expect_identical(dim(epred), c(20L, 40L))
  expect_identical(epred, predict(fit, fit$x, type = "draws"))
})

test_that("posterior_predict is the mean plus a residual", {
  fit <- gaussian_fit()

  set.seed(2)
  replicates <- posterior_predict(fit)

  expect_identical(dim(replicates), c(20L, 40L))
  expect_false(identical(replicates, posterior_epred(fit)))
})

test_that("posterior_predict is governed by set.seed", {
  fit <- gaussian_fit()

  set.seed(3)
  first <- posterior_predict(fit)
  set.seed(3)
  again <- posterior_predict(fit)

  expect_identical(again, first)
})

test_that("the probit replicates are labels", {
  fit <- probit_fit()

  set.seed(4)
  replicates <- posterior_predict(fit)

  expect_true(all(replicates %in% c(0, 1)))
})

test_that("posterior_predict takes new rows", {
  fit <- gaussian_fit()
  rows <- fit$x[1:5, ]

  expect_identical(dim(posterior_predict(fit, rows)), c(20L, 5L))
  expect_identical(dim(posterior_epred(fit, rows)), c(20L, 5L))
})

test_that("log_lik is one column per observation", {
  fit <- gaussian_fit()

  values <- log_lik(fit)

  expect_identical(dim(values), c(20L, 40L))
  expect_true(all(is.finite(values)))
})

test_that("log_lik on new rows needs the response", {
  fit <- gaussian_fit()
  rows <- fit$x[1:5, ]

  expect_error(log_lik(fit, rows), class = "thiessen_error")
  expect_identical(dim(log_lik(fit, rows, fit$y[1:5])), c(20L, 5L))
  expect_error(log_lik(fit, rows, fit$y[1:4]), class = "thiessen_error")
})

test_that("log_lik takes the response from a formula fit's newdata", {
  fixture <- small_fixture()
  frame <- data.frame(y = fixture$y, a = fixture$x[, 1], b = fixture$x[, 2])
  fit <- thiessen(y ~ a + b, frame, small_control(), seed = 1)

  expect_identical(dim(log_lik(fit, frame)), c(20L, 40L))
})

test_that("loo runs on the log-likelihood", {
  skip_if_not_installed("loo")
  fit <- gaussian_fit()

  estimate <- suppressWarnings(loo::loo(log_lik(fit)))

  expect_s3_class(estimate, "loo")
  expect_true(is.finite(estimate$estimates["elpd_loo", "Estimate"]))
})

test_that("predictive_interval reports its percentiles", {
  fit <- gaussian_fit()

  bounds <- predictive_interval(fit, prob = 0.8)

  expect_identical(dim(bounds), c(40L, 2L))
  expect_identical(colnames(bounds), c("10%", "90%"))
  expect_true(all(bounds[, 1] <= bounds[, 2]))
})

test_that("a wider interval contains a narrower one", {
  fit <- gaussian_fit()

  narrow <- predictive_interval(fit, prob = 0.5)
  wide <- predictive_interval(fit, prob = 0.95)

  expect_true(all(wide[, 1] <= narrow[, 1]))
  expect_true(all(wide[, 2] >= narrow[, 2]))
})

test_that("the interval mass must be a probability", {
  fit <- gaussian_fit()

  expect_error(predictive_interval(fit, prob = 1), class = "thiessen_error")
})

test_that("bayesplot draws from the fit unmodified", {
  skip_if_not_installed("bayesplot")
  skip_if_not_installed("posterior")
  fit <- gaussian_fit()

  plot <- bayesplot::mcmc_hist(posterior::as_draws_df(fit), pars = "sigma")

  expect_s3_class(plot, "ggplot")
})

test_that("tidybayes gathers the draws from the fit unmodified", {
  skip_if_not_installed("tidybayes")
  fit <- gaussian_fit()

  gathered <- tidybayes::spread_draws(fit, sigma)

  expect_true(all(c(".draw", "sigma") %in% names(gathered)))
  expect_identical(nrow(gathered), 20L)
})
