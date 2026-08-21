# The short test schedule never meets the convergence thresholds, so the
# multi-chain fits here are wrapped to keep the warning out of the report.
chain_fit <- function(seed = 1, chains = 2, control = small_control()) {
  fixture <- small_fixture()
  suppressWarnings(
    thiessen(fixture$x, fixture$y, control, seed = seed, chains = chains)
  )
}

test_that("chains pool their draws", {
  one <- chain_fit(chains = 1)
  two <- chain_fit()

  expect_identical(one$n_chains, 1L)
  expect_identical(two$n_chains, 2L)
  expect_identical(two$n_draws, 2L * one$n_draws)
  expect_identical(nrow(thiessen_diagnostics(two)), 40L)
  expect_identical(thiessen_diagnostics(two)$chain, rep(1:2, each = 20L))
})

test_that("the first chain is the single-chain fit", {
  one <- chain_fit(chains = 1)
  two <- chain_fit()

  # Chain 0 is the seed itself, so the first chain repeats the one-chain fit.
  expect_identical(
    thiessen_diagnostics(two)$sigma[1:20],
    thiessen_diagnostics(one)$sigma
  )
})

test_that("the same seed reproduces every chain", {
  first <- chain_fit(seed = 3, chains = 3)
  second <- chain_fit(seed = 3, chains = 3)

  expect_identical(predict(second), predict(first))
  expect_identical(thiessen_diagnostics(second), thiessen_diagnostics(first))
})

test_that("the pooled prediction is the mean of the pooled draws", {
  fixture <- small_fixture()
  rows <- fixture$x[1:5, ]
  pooled <- chain_fit()

  draws <- predict(pooled, rows, type = "draws")

  expect_identical(dim(draws), c(40L, 5L))
  expect_equal(predict(pooled, rows), colMeans(draws), tolerance = 1e-12)
})

test_that("the draws object carries the chains", {
  fit <- chain_fit()

  draws <- posterior::as_draws_array(fit)
  summary <- posterior::summarise_draws(draws)

  expect_identical(posterior::nchains(fit), 2L)
  expect_identical(posterior::ndraws(fit), 40L)
  expect_identical(posterior::nchains(draws), 2L)
  expect_false(is.na(summary$rhat[summary$variable == "sigma"]))
})

test_that("chains must be a whole number of at least one", {
  fixture <- small_fixture()

  for (chains in list(0, -1, 1.5, "two", NA)) {
    expect_error(
      thiessen(fixture$x, fixture$y, small_control(), seed = 1,
               chains = chains),
      class = "thiessen_error"
    )
  }
})
