test_that("the default control is the published configuration", {
  control <- thiessen_control()

  expect_s3_class(control, "thiessen_control")
  expect_null(control$outcome)
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
  expect_error(gaussian_outcome(zeta = 1), "unused argument")
  expect_error(thiessen_control(zeta = 1), "unused argument")
})

test_that("the core rejects an invalid value at construction", {
  expect_error(thiessen_control(outcome = gaussian_outcome(q = 1.5)),
               class = "thiessen_error")
  expect_error(thiessen_control(tessellations = 0),
               class = "thiessen_error")
})

test_that("a variance ensemble under the probit family is refused", {
  expect_error(
    thiessen_control(
      outcome = probit_outcome(),
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
  expect_identical(
    format(gaussian_outcome()), "gaussian_outcome(nu = 6, q = 0.85)"
  )
  expect_identical(format(probit_outcome()), "probit_outcome()")
  expect_identical(
    format(probit_outcome(offset = 0.5)), "probit_outcome(offset = 0.5)"
  )
  expect_output(
    print(gaussian_outcome(nu = 3)), "gaussian_outcome(nu = 3", fixed = TRUE
  )
})

test_that("the parameter groups print as their calls", {
  expect_identical(
    format(term_params(tessellations = 40)),
    "term_params(tessellations = 40, k = 3, lambda_c = 5)"
  )
  expect_output(print(structure_params(omega = 2)), "omega = 2")
})

test_that("a formatted group parses back to the group", {
  groups <- list(
    gaussian_outcome(nu = 3),
    probit_outcome(),
    probit_outcome(offset = 0.5),
    general_params(burn_in = 10, draws = 20, prior_only = TRUE),
    geometry_params(
      metric = list("euclidean", list(spherical = list(sphere = 0))),
      sigma_c = 0.5
    ),
    term_params(
      tessellations = 40,
      geometry = geometry_params(metric = list("categorical")),
      structure = structure_params(omega = 2)
    )
  )

  for (group in groups) {
    expect_identical(eval(parse(text = format(group))), group)
  }
})

test_that("the scalar arguments reject a value of the wrong shape", {
  expect_error(term_params(k = "3"), class = "thiessen_error")
  expect_error(term_params(k = c(3, 4)), class = "thiessen_error")
  expect_error(term_params(tessellations = 2.5), class = "thiessen_error")
  expect_error(general_params(burn_in = -1), class = "thiessen_error")
  expect_error(general_params(prior_only = NA), class = "thiessen_error")
  expect_error(general_params(prior_only = "yes"), class = "thiessen_error")
  expect_error(gaussian_outcome(nu = NULL), class = "thiessen_error")
  expect_error(probit_outcome(offset = c(0, 1)), class = "thiessen_error")
})

test_that("printing a control reports each group", {
  expect_output(print(thiessen_control(tessellations = 50)),
                "tessellations = 50")
  expect_output(print(thiessen_control()), "none \\(constant spread\\)")
  expect_output(print(thiessen_control()), "from the response")
  expect_output(
    print(thiessen_control(outcome = gaussian_outcome())), "gaussian_outcome\\("
  )
})

test_that("a control object is required", {
  fixture <- small_fixture()

  expect_error(
    thiessen(fixture$x, fixture$y, control = list(tessellations = 5)),
    class = "thiessen_error"
  )
})

test_that("a fit resolves every default the control-surface article prints", {
  # The article reads these from the core at knit time rather than stating
  # them, so a renamed or moved field must fail here.
  fixture <- small_fixture()
  resolved <- thiessen(
    fixture$x, fixture$y,
    thiessen_control(general_params = general_params(burn_in = 1, draws = 1)),
    seed = 1
  )$control

  expect_true(is.numeric(resolved$mean_params$tessellations))
  expect_true(is.numeric(resolved$mean_params$k))
  expect_true(is.numeric(resolved$mean_params$lambda_c))
  expect_true(is.numeric(resolved$mean_params$geometry$sigma_c))
  expect_true(is.numeric(resolved$mean_params$structure$omega))
  expect_true(is.numeric(resolved$outcome$nu))
  expect_true(is.numeric(resolved$outcome$q))
})
