test_that("the defaults are the core's", {
  control <- thiessen_control()

  expect_s3_class(control, "thiessen_control")
  expect_identical(control$model, "gaussian")
  expect_identical(control$m, 200L)
  expect_identical(control$draws, 1000L)
  expect_identical(control$lambda_c, 5)
  expect_null(control$omega)
  expect_identical(control$metric, list())
})

test_that("an argument given replaces the default and the rest stand", {
  control <- thiessen_control(m = 50, draws = 200)

  expect_identical(control$m, 50)
  expect_identical(control$draws, 200)
  expect_identical(control$burn_in, thiessen_control()$burn_in)
})

test_that("the field order is the core's", {
  expect_identical(
    names(thiessen_control()),
    c("model", "m", "nu", "q", "k", "sigma_c", "omega", "lambda_c",
      "burn_in", "draws", "thinning", "prior_only", "offset", "m_var",
      "metric")
  )
})

test_that("the core rejects an invalid value at construction", {
  expect_error(thiessen_control(q = 1.5), class = "thiessen_error")
  expect_error(thiessen_control(m = 0), class = "thiessen_error")
  expect_error(thiessen_control(model = "nonesuch"), class = "thiessen_error")
})

test_that("a metric is carried through as the core names it", {
  control <- thiessen_control(metric = c("euclidean", "categorical"))

  expect_identical(control$metric, list("euclidean", "categorical"))
})

test_that("a spherical metric carries its sphere label", {
  control <- thiessen_control(
    metric = list("euclidean", list(spherical = list(sphere = 0)))
  )

  expect_length(control$metric, 2L)
  expect_identical(control$metric[[2]]$spherical$sphere, 0)
})

test_that("printing reports every field", {
  expect_output(print(thiessen_control(m = 50)), "m *50")
  expect_output(print(thiessen_control()), "resolved at fit")
  expect_output(print(thiessen_control()), "euclidean on every column")
})

test_that("a control object is required", {
  fixture <- small_fixture()

  expect_error(
    thiessen(fixture$x, fixture$y, control = list(m = 5)),
    class = "thiessen_error"
  )
})
