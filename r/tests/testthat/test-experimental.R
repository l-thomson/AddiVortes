# The exposure policy for items behind the core's `experimental` feature.
# A constructor exists in every build and the core reports a gated item
# with the condition class `thiessen_requires_feature`, so the tests read
# the families from the core and hold no list of their own: an item added
# or graduated needs no change here.

# The outcome families the core in use carries, by their stored names.
core_families <- function() {
  catalogue <- jsonlite::fromJSON(core_outcome_defaults(), simplifyVector = FALSE)
  vapply(catalogue, function(family) names(family)[[1L]], character(1))
}

# The families this package constructs, by their stored names.
surface_families <- function() {
  exports <- grep("_outcome$", getNamespaceExports("thiessen"), value = TRUE)
  sub("_outcome$", "", exports)
}

# A family's constructor, called at its defaults.
outcome_of <- function(kind) {
  do.call(paste0(kind, "_outcome"), list())
}

test_that("the published models are accepted in either build", {
  expect_s3_class(
    thiessen_control(outcome = gaussian_outcome()), "thiessen_control"
  )
  expect_s3_class(
    thiessen_control(outcome = probit_outcome()), "thiessen_control"
  )
  expect_s3_class(
    thiessen_control(variance_params = term_params(tessellations = 40)),
    "thiessen_control"
  )
})

test_that("every family the core carries has a constructor", {
  expect_true(all(core_families() %in% surface_families()))
})

test_that("a build with the feature turns no family away", {
  skip_if_not(core_experimental())

  expect_setequal(surface_families(), core_families())
  # A family may still be rejected on its own terms, the tobit outcome
  # needing a censoring limit; none is rejected for the feature.
  for (kind in surface_families()) {
    condition <- rlang::catch_cnd(thiessen_control(outcome = outcome_of(kind)))

    expect_false(inherits(condition, "thiessen_requires_feature"), label = kind)
  }
})

test_that("a build without the feature reports each gated family", {
  skip_if(core_experimental())

  gated <- setdiff(surface_families(), core_families())
  expect_gt(length(gated), 0L)
  for (kind in gated) {
    expect_error(
      thiessen_control(outcome = outcome_of(kind)),
      class = "thiessen_requires_feature"
    )
  }
})

test_that("a degrees-of-freedom grid crosses as an array", {
  skip_if_not(core_experimental())

  expect_s3_class(
    thiessen_control(outcome = student_t_outcome(df = c(3, 6, 12))),
    "thiessen_control"
  )
})

test_that("a gated component option reports the feature", {
  skip_if(core_experimental())
  config <- '{"mean_params": {"geometry": {"membership": {"soft": {}}}}}'

  expect_error(
    core_call(core_validate(config)),
    class = "thiessen_requires_feature"
  )
})

test_that("the published default of a gated field is accepted", {
  published <- c(
    '{"mean_params": {"geometry": {"membership": "hard"}}}',
    '{"mean_params": {"structure": {"inclusion": "uniform"}}}',
    '{"mean_params": {"cell": {"basis": "constant"}}}'
  )
  for (config in published) {
    expect_no_error(core_call(core_validate(config)))
  }
})

test_that("an invalid configuration keeps the plain condition class", {
  config <- '{"mean_params": {"geometry": {"sigma_c": -1}}}'

  condition <- rlang::catch_cnd(core_call(core_validate(config)))

  expect_s3_class(condition, "thiessen_error")
  expect_false(inherits(condition, "thiessen_requires_feature"))
})

test_that("a saved fit naming a gated model reports the feature", {
  skip_if(core_experimental())
  fixture <- small_fixture()
  fit <- thiessen(fixture$x, fixture$y, small_control(), seed = 1)
  fit$state$payload <- swap_payload_name(
    fit$state$payload, "gaussian", "laplace"
  )
  fit <- unserialize(serialize(fit, NULL))

  expect_error(predict(fit), class = "thiessen_requires_feature")
})

test_that("a saved fit naming an unknown model fails to load", {
  fixture <- small_fixture()
  fit <- thiessen(fixture$x, fixture$y, small_control(), seed = 1)
  fit$state$payload <- swap_payload_name(
    fit$state$payload, "gaussian", "robust"
  )
  fit <- unserialize(serialize(fit, NULL))

  expect_error(predict(fit), class = "thiessen_error")
})
