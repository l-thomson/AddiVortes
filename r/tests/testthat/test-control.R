test_that("the default control is the published configuration", {
  control <- thiessen_control()

  expect_s3_class(control, "thiessen_control")
  expect_s3_class(control$outcome, "thiessen_gaussian")
  expect_identical(control$outcome$nu, 6)
  expect_identical(control$outcome$q, 0.85)
  expect_s3_class(control$mean_params, "term_params")
  expect_null(control$mean_params$tessellations)
  expect_identical(control$mean_params$k, 3)
  expect_identical(control$mean_params$lambda_c, 5)
  expect_null(control$variance_params)
  expect_identical(control$general_params$burn_in, 200)
  expect_identical(control$general_params$draws, 1000)
})

test_that("the groups are the four of the core's configuration", {
  expect_identical(
    names(thiessen_control()),
    c("outcome", "mean_params", "variance_params", "general_params")
  )
})

test_that("the tessellations shortcut promotes into mean_params", {
  control <- thiessen_control(tessellations = 50)

  expect_identical(control$mean_params$tessellations, 50)
})

test_that("the shortcut and an explicit count together are refused", {
  expect_error(
    thiessen_control(
      tessellations = 50,
      mean_params = term_params(tessellations = 100)
    ),
    class = "thiessen_error"
  )
})

test_that("a group of the wrong class is refused, naming the group", {
  expect_error(thiessen_control(outcome = "gaussian"), "outcome")
  expect_error(thiessen_control(mean_params = list(k = 3)), "mean_params")
  expect_error(
    thiessen_control(variance_params = list(tessellations = 40)),
    "variance_params"
  )
  expect_error(
    thiessen_control(general_params = list(draws = 10)),
    "general_params"
  )
})

test_that("an unknown name in a group is an unused argument", {
  expect_error(term_params(zeta = 1), "unused argument")
  expect_error(geometry_params(zeta = 1), "unused argument")
  expect_error(structure_params(zeta = 1), "unused argument")
  expect_error(general_params(zeta = 1), "unused argument")
  expect_error(gaussian(zeta = 1), "unused argument")
  expect_error(thiessen_control(zeta = 1), "unused argument")
})

test_that("the core rejects an invalid value at construction", {
  expect_error(thiessen_control(outcome = gaussian(q = 1.5)),
               class = "thiessen_error")
  expect_error(thiessen_control(tessellations = 0),
               class = "thiessen_error")
})

test_that("a variance ensemble under the probit family is refused", {
  expect_error(
    thiessen_control(
      outcome = probit(),
      variance_params = term_params(tessellations = 40)
    ),
    "fixed at 1 for identification",
    class = "thiessen_error"
  )
})

test_that("the ensembles share the geometry declared on mean_params", {
  control <- thiessen_control(
    mean_params = term_params(geometry = geometry_params(sigma_c = 0.5)),
    variance_params = term_params(tessellations = 4)
  )
  grouped <- jsonlite::fromJSON(config_json(control), simplifyVector = FALSE)

  expect_identical(grouped$variance_params$geometry,
                   grouped$mean_params$geometry)
})

test_that("a metric is carried through as the core names it", {
  geometry <- geometry_params(metric = c("euclidean", "categorical"))

  expect_identical(geometry$metric, list("euclidean", "categorical"))
})

test_that("a spherical metric carries its sphere label", {
  geometry <- geometry_params(
    metric = list("euclidean", list(spherical = list(sphere = 0)))
  )

  expect_length(geometry$metric, 2L)
  expect_identical(geometry$metric[[2]]$spherical$sphere, 0)
})

test_that("the outcome constructors print as their calls", {
  expect_identical(format(gaussian()), "gaussian(nu = 6, q = 0.85)")
  expect_identical(format(probit()), "probit()")
  expect_identical(format(probit(offset = 0.5)), "probit(offset = 0.5)")
  expect_output(print(gaussian(nu = 3)), "gaussian(nu = 3", fixed = TRUE)
})

test_that("the parameter groups print as their calls", {
  expect_identical(
    format(term_params(tessellations = 40)),
    "term_params(tessellations = 40, k = 3, lambda_c = 5)"
  )
  expect_output(print(structure_params(omega = 2)), "omega = 2")
})

test_that("printing a control reports each group", {
  expect_output(print(thiessen_control(tessellations = 50)),
                "tessellations = 50")
  expect_output(print(thiessen_control()), "none \\(constant spread\\)")
  expect_output(print(thiessen_control()), "gaussian\\(")
})

test_that("a control object is required", {
  fixture <- small_fixture()

  expect_error(
    thiessen(fixture$x, fixture$y, control = list(tessellations = 5)),
    class = "thiessen_error"
  )
})
