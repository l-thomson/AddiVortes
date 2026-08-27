test_that("the feature-error prefix is the one the core writes", {
  expect_identical(core_requires_feature_prefix(), REQUIRES_FEATURE)
})

test_that("the linked core is the version DESCRIPTION declares", {
  declared <- utils::packageDescription("thiessen")[["Config/thiessen/core-version"]]
  expect_match(core_version(), "^[0-9]+\\.[0-9]+\\.[0-9]+")
  expect_identical(core_version(), declared)
})
