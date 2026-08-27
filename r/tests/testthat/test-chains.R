# The short test schedule never meets the convergence thresholds, so the
# multi-chain fits here are wrapped to keep the warning out of the report.
chain_fit <- function(seed = 1, chains = 2, control = small_control(),
                      threads = 1) {
  fixture <- small_fixture()
  suppressWarnings(
    thiessen(fixture$x, fixture$y, control, seed = seed, chains = chains,
             threads = threads)
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

# Threaded fits against the serial fit of the same seed.
expect_threads_alike <- function(serial, chains, threads) {
  fixture <- small_fixture()
  for (count in threads) {
    threaded <- chain_fit(seed = 5, chains = chains, threads = count)

    expect_identical(threaded$threads, as.integer(count))
    expect_identical(threaded$state$payload, serial$state$payload)
    expect_identical(fitted(threaded), fitted(serial))
    expect_identical(thiessen_diagnostics(threaded), thiessen_diagnostics(serial))
    expect_identical(
      predict(threaded, fixture$x, type = "draws"),
      predict(serial, fixture$x, type = "draws")
    )
    expect_identical(
      predict(threaded, fixture$x, interval = "prediction"),
      predict(serial, fixture$x, interval = "prediction")
    )
  }
}

# The CRAN policy allows a check two cores, so the wider counts are kept
# off it.
test_that("threaded chains draw what the chains run in turn draw", {
  serial <- chain_fit(seed = 5, chains = 3)

  expect_threads_alike(serial, chains = 3, threads = 2)
  expect_identical(serial$threads, 1L)
})

test_that("more threads than chains, or than cores, draw the same", {
  skip_on_cran()
  serial <- chain_fit(seed = 5, chains = 3)

  expect_threads_alike(serial, chains = 3, threads = c(3, 8))
})

test_that("predict() takes a thread count of its own", {
  fixture <- small_fixture()
  fit <- chain_fit(chains = 2)
  mean <- predict(fit, fixture$x)
  interval <- predict(fit, fixture$x, interval = "prediction")

  expect_identical(predict(fit, fixture$x, threads = 2), mean)
  expect_identical(
    predict(fit, fixture$x, interval = "prediction", threads = 2),
    interval
  )
  expect_identical(predict(fit, fixture$x), mean)
  expect_error(
    predict(fit, fixture$x, threads = 0),
    class = "thiessen_error"
  )
})

test_that("a threaded fit predicts alike after a save and a load", {
  fixture <- small_fixture()
  threaded <- chain_fit(chains = 2, threads = 2)
  path <- tempfile(fileext = ".rds")
  on.exit(unlink(path), add = TRUE)
  saveRDS(threaded, path)

  loaded <- readRDS(path)

  expect_identical(loaded$threads, 2L)
  expect_identical(predict(loaded, fixture$x), predict(threaded, fixture$x))
})

test_that("threads must be a whole number of at least one", {
  fixture <- small_fixture()

  for (threads in list(0, -1, 1.5, "two", NA)) {
    expect_error(
      thiessen(fixture$x, fixture$y, small_control(), seed = 1,
               threads = threads, chains = 1),
      class = "thiessen_error"
    )
  }
})

test_that("a fit defaults to four chains on one thread", {
  fixture <- small_fixture()
  withr::local_options(mc.cores = NULL)

  fit <- suppressWarnings(
    thiessen(fixture$x, fixture$y, small_control(), seed = 1)
  )

  expect_identical(fit$n_chains, 4L)
  expect_identical(fit$threads, 1L)
  expect_identical(fit$convergence$n_chains, 4L)
})

test_that("mc.cores sets the default threads and leaves the draws alone", {
  fixture <- small_fixture()
  serial <- chain_fit(seed = 5, chains = 4)
  withr::local_options(mc.cores = 2)

  threaded <- suppressWarnings(
    thiessen(fixture$x, fixture$y, small_control(), seed = 5)
  )

  expect_identical(threaded$threads, 2L)
  expect_identical(threaded$state$payload, serial$state$payload)
  expect_identical(fitted(threaded), fitted(serial))
  expect_identical(thiessen_diagnostics(threaded), thiessen_diagnostics(serial))
})
