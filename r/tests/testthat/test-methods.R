test_that("predict at the training rows is the fitted values", {
  fixture <- small_fixture()

  fit <- thiessen(fixture$x, fixture$y, small_control(), seed = 1)

  expect_identical(predict(fit), fitted(fit))
})

test_that("each draw type has one row per draw and one column per row", {
  fixture <- small_fixture()
  fit <- thiessen(fixture$x, fixture$y, small_control(), seed = 1)
  rows <- fixture$x[1:5, ]

  for (type in c("draws", "latent", "variance")) {
    expect_identical(dim(predict(fit, rows, type = type)), c(20L, 5L))
  }
})

test_that("an interval carries the fit and its bounds", {
  fixture <- small_fixture()
  fit <- thiessen(fixture$x, fixture$y, small_control(), seed = 1)
  rows <- fixture$x[1:5, ]

  interval <- predict(fit, rows, interval = "credible")

  expect_identical(colnames(interval), c("fit", "lower", "upper"))
  expect_identical(dim(interval), c(5L, 3L))
  expect_true(all(interval[, "lower"] <= interval[, "fit"]))
  expect_true(all(interval[, "fit"] <= interval[, "upper"]))
})

test_that("a prediction interval is at least as wide as a credible one", {
  fixture <- small_fixture()
  fit <- thiessen(fixture$x, fixture$y, small_control(), seed = 1)
  rows <- fixture$x[1:5, ]

  credible <- predict(fit, rows, interval = "credible")
  prediction <- predict(fit, rows, interval = "prediction")

  expect_true(all(
    prediction[, "upper"] - prediction[, "lower"] >=
      credible[, "upper"] - credible[, "lower"]
  ))
})

test_that("an interval is refused for a per-draw type", {
  fixture <- small_fixture()
  fit <- thiessen(fixture$x, fixture$y, small_control(), seed = 1)

  expect_error(
    predict(fit, fixture$x, type = "draws", interval = "credible"),
    class = "thiessen_error"
  )
})

test_that("the level must be a probability", {
  fixture <- small_fixture()
  fit <- thiessen(fixture$x, fixture$y, small_control(), seed = 1)

  expect_error(
    predict(fit, fixture$x, interval = "credible", level = 1),
    class = "thiessen_error"
  )
  expect_error(
    predict(fit, fixture$x, interval = "credible", level = c(0.5, 0.9)),
    class = "thiessen_error"
  )
})

test_that("a column count the fit does not have is refused by the core", {
  fixture <- small_fixture()
  fit <- thiessen(fixture$x, fixture$y, small_control(), seed = 1)

  expect_error(
    predict(fit, fixture$x[, 1, drop = FALSE]),
    class = "thiessen_error"
  )
})

test_that("sigma is the posterior mean under the Gaussian model", {
  fixture <- small_fixture()
  fit <- thiessen(fixture$x, fixture$y, small_control(), seed = 1)

  expect_identical(sigma(fit), mean(core_sigma(fit$state)))
  expect_true(sigma(fit) > 0)
})

test_that("sigma is one under the probit model", {
  fixture <- small_fixture()
  labels <- as.double(fixture$y >= stats::median(fixture$y))

  fit <- thiessen(fixture$x, labels, small_control(outcome = probit()), seed = 1)

  expect_identical(sigma(fit), 1)
})

test_that("the heteroscedastic model has no single residual scale", {
  fixture <- small_fixture()

  fit <- thiessen(
    fixture$x, fixture$y,
    small_control(variance_params = term_params(tessellations = 4)),
    seed = 1
  )

  expect_error(sigma(fit), class = "thiessen_error")
})

test_that("printing reports the model and the schedule", {
  fixture <- small_fixture()
  fit <- thiessen(fixture$x, fixture$y, small_control(), seed = 1)

  expect_output(print(fit), "gaussian model, 40 observations, 2 covariates")
  expect_output(print(fit), "8 tessellations, 20 draws kept after 10 burn-in")
})

test_that("summary carries the residual and sigma quantiles", {
  fixture <- small_fixture()
  fit <- thiessen(fixture$x, fixture$y, small_control(), seed = 1)

  summary <- summary(fit)

  expect_s3_class(summary, "summary.thiessen")
  expect_length(summary$residuals, 5L)
  expect_length(summary$sigma, 5L)
  expect_output(print(summary), "sigma:")
})

test_that("summary reports no sigma where the model has none", {
  fixture <- small_fixture()

  fit <- thiessen(
    fixture$x, fixture$y,
    small_control(variance_params = term_params(tessellations = 4)),
    seed = 1
  )

  expect_null(summary(fit)$sigma)
})

test_that("plot draws the traces and returns the fit invisibly", {
  fixture <- small_fixture()
  fit <- suppressWarnings(
    thiessen(fixture$x, fixture$y, small_control(), seed = 1, chains = 2)
  )

  grDevices::pdf(NULL)
  on.exit(grDevices::dev.off())
  returned <- withVisible(plot(fit))

  expect_identical(returned$value, fit)
  expect_false(returned$visible)
})

test_that("plot covers a model with no sigma", {
  fixture <- small_fixture()
  fit <- thiessen(
    fixture$x, fixture$y,
    small_control(variance_params = term_params(tessellations = 4)),
    seed = 1
  )

  grDevices::pdf(NULL)
  on.exit(grDevices::dev.off())
  expect_no_error(plot(fit))
})
