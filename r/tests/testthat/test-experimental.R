# The exposure policy for items behind the core's `experimental` feature.
# The package builds the core without the feature, so only the published
# models are accepted. It keeps no list of model names: every name is
# validated by the core, so the policy holds without a change here when an
# item is added or graduates.

PUBLISHED <- c("gaussian", "probit", "heteroscedastic")

# Names reserved for items behind the feature; none is accepted here.
GATED <- c("soft", "robust_t", "dart", "minkowski", "manhattan",
           "mahalanobis", "gower", "cosine", "weighted", "composite")

test_that("the package builds the core without the feature", {
  expect_false(core_experimental())
})

test_that("the published models are accepted", {
  for (name in PUBLISHED) {
    expect_s3_class(thiessen_control(model = name), "thiessen_control")
  }
})

test_that("a gated name is rejected", {
  for (name in GATED) {
    expect_error(thiessen_control(model = name), class = "thiessen_error")
  }
})

test_that("the rejection comes from the core, not from a list here", {
  expect_error(thiessen_control(model = "soft"), "unknown variant")
  expect_false(any(grepl("soft", deparse(body(thiessen_control)), fixed = TRUE)))
})

test_that("a saved fit naming a gated model fails to load", {
  fixture <- small_fixture()
  fit <- thiessen(fixture$x, fixture$y, small_control(), seed = 1)
  fit$state <- sub('"gaussian"', '"soft"', fit$state, fixed = TRUE)

  expect_error(predict(fit), class = "thiessen_error")
})

test_that("the package exposes no gated name", {
  exposed <- tolower(getNamespaceExports("thiessen"))

  expect_length(intersect(exposed, GATED), 0L)
  expect_false("experimental" %in% exposed)
})
