test_that("a fit survives a round trip through saveRDS", {
  fixture <- small_fixture()
  fit <- thiessen(fixture$x, fixture$y, small_control(), seed = 1)
  path <- tempfile(fileext = ".rds")
  on.exit(unlink(path), add = TRUE)

  saveRDS(fit, path)
  restored <- readRDS(path)

  expect_identical(predict(restored, fixture$x), predict(fit, fixture$x))
  expect_identical(thiessen_diagnostics(restored), thiessen_diagnostics(fit))
})

test_that("a fit read in a new session predicts the same values", {
  skip_on_cran()
  skip_if_not_installed("callr")
  fixture <- small_fixture()
  fit <- thiessen(fixture$x, fixture$y, small_control(), seed = 1)
  path <- tempfile(fileext = ".rds")
  on.exit(unlink(path), add = TRUE)
  saveRDS(fit, path)

  predictions <- callr::r(
    function(path, rows) predict(readRDS(path), rows),
    args = list(path = path, rows = fixture$x[1:5, ]),
    libpath = .libPaths(),
    package = "thiessen"
  )

  expect_identical(predictions, predict(fit, fixture$x[1:5, ]))
})

test_that("a state the build cannot read errors with the package's class", {
  fixture <- small_fixture()
  fit <- thiessen(fixture$x, fixture$y, small_control(), seed = 1)
  fit$state <- sub('"gaussian"', '"crackle"', fit$state, fixed = TRUE)

  expect_error(predict(fit), class = "thiessen_error")
})
