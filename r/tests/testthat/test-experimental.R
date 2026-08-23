# The exposure policy for items behind the core's `experimental` feature.
# The package builds the core without the feature, so only the published
# models are reachable: a gated outcome has no constructor here, and a
# configuration naming a gated field or variant is rejected by the core, so
# the policy holds without a change here when an item is added or graduates.

# Names reserved for items behind the feature; none is accepted here.
GATED <- c("soft", "robust_t", "dart", "minkowski", "manhattan",
           "mahalanobis", "gower", "cosine", "weighted", "composite")

test_that("the package builds the core without the feature", {
  expect_false(core_experimental())
})

test_that("the published models are accepted", {
  expect_s3_class(thiessen_control(outcome = gaussian()), "thiessen_control")
  expect_s3_class(thiessen_control(outcome = probit()), "thiessen_control")
  expect_s3_class(
    thiessen_control(variance_params = term_params(tessellations = 40)),
    "thiessen_control"
  )
})

test_that("a gated outcome fails to deserialise in the core", {
  for (name in GATED) {
    config <- sprintf('{"outcome": {"%s": {}}}', name)
    expect_error(core_validate(config), "unknown variant")
  }
})

test_that("no constructor exists for a gated outcome", {
  exposed <- tolower(getNamespaceExports("thiessen"))

  expect_length(intersect(exposed, GATED), 0L)
  expect_false("experimental" %in% exposed)
})

test_that("a gated field fails to deserialise in the core", {
  gated_fields <- list(
    '{"mean_params": {"geometry": {"membership": "soft"}}}',
    '{"mean_params": {"geometry": {"precision": [1.0]}}}',
    '{"mean_params": {"structure": {"inclusion": "uniform"}}}',
    '{"mean_params": {"cell": {"basis": "linear"}}}'
  )
  for (config in gated_fields) {
    expect_error(core_validate(config))
  }
})

test_that("a saved fit naming a gated model fails to load", {
  fixture <- small_fixture()
  fit <- thiessen(fixture$x, fixture$y, small_control(), seed = 1)
  fit$state <- sub('"gaussian"', '"soft"', fit$state, fixed = TRUE)

  expect_error(predict(fit), class = "thiessen_error")
})
