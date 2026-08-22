test_that("the diagnostics carry one row per draw", {
  fixture <- small_fixture()

  fit <- thiessen(fixture$x, fixture$y, small_control(), seed = 1)
  diagnostics <- thiessen_diagnostics(fit)

  expect_identical(
    names(diagnostics),
    c("chain", "draw", "sigma", "cell_count", "dimension_count")
  )
  expect_identical(nrow(diagnostics), 20L)
  expect_identical(diagnostics$chain, rep(1L, 20L))
  expect_identical(diagnostics$draw, 1:20)
  expect_true(all(diagnostics$sigma > 0))
  expect_true(all(diagnostics$cell_count >= 1))
})

test_that("the sigma column is absent where the model has no sigma", {
  fixture <- small_fixture()
  y <- as.numeric(fixture$y > stats::median(fixture$y))

  probit <- thiessen(fixture$x, y, small_control(outcome = probit()), seed = 1)
  heteroscedastic <- thiessen(
    fixture$x, fixture$y,
    small_control(variance_params = term_params(tessellations = 4)), seed = 1
  )

  expect_false("sigma" %in% names(thiessen_diagnostics(probit)))
  expect_false("sigma" %in% names(thiessen_diagnostics(heteroscedastic)))
})

test_that("the sigma column is the sigma draws of the draws object", {
  skip_if_not_installed("posterior")
  fixture <- small_fixture()

  fit <- thiessen(fixture$x, fixture$y, small_control(), seed = 1)
  draws <- posterior::as_draws_df(fit)

  expect_identical(thiessen_diagnostics(fit)$sigma, draws$sigma)
})

test_that("the inclusion proportions sum to one and name the columns", {
  fixture <- small_fixture()
  frame <- data.frame(y = fixture$y, a = fixture$x[, 1], b = fixture$x[, 2])

  fit <- thiessen(y ~ a + b, frame, small_control(), seed = 1)
  inclusion <- variable_inclusion(fit)

  expect_identical(names(inclusion), c("a", "b"))
  expect_equal(sum(inclusion), 1, tolerance = 1e-12)
})

test_that("the accessors refuse an object that is not a fit", {
  expect_error(thiessen_diagnostics(1), class = "thiessen_error")
  expect_error(variable_inclusion(1), class = "thiessen_error")
})
