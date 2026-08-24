# progressr signals nothing in a non-interactive session unless it is
# enabled, and `with_progress()` muffles every condition before a handler
# established outside it runs, so count the conditions with the option set
# and no progressr handler in the way.
count_updates <- function(expr) {
  old <- options(progressr.enable = TRUE)
  on.exit(options(old), add = TRUE)
  types <- character(0)
  withCallingHandlers(
    expr,
    progression = function(condition) {
      types <<- c(types, condition$type)
    }
  )
  sum(types == "update")
}

test_that("a fit signals every progression it requests", {
  fixture <- small_fixture()
  control <- thiessen_control(
    tessellations = 8,
    general_params = general_params(burn_in = 40, draws = 80)
  )

  expect_identical(progress_updates(control), 100L)
  expect_identical(
    count_updates(thiessen(fixture$x, fixture$y, control, seed = 1)),
    100L
  )
})

test_that("a fit shorter than a hundred sweeps signals one per sweep", {
  fixture <- small_fixture()

  expect_identical(progress_updates(small_control()), 30L)
  expect_identical(
    count_updates(thiessen(fixture$x, fixture$y, small_control(), seed = 1)),
    30L
  )
})

test_that("the sweeps of every chain are reported", {
  fixture <- small_fixture()

  expect_identical(
    count_updates(suppressWarnings(
      thiessen(fixture$x, fixture$y, small_control(), chains = 2, seed = 1)
    )),
    progress_updates(small_control(), 2L)
  )
})

test_that("the number of updates does not exceed the sweeps", {
  expect_identical(
    progress_updates(
      thiessen_control(general_params = general_params(burn_in = 1, draws = 2))
    ),
    3L
  )
  expect_identical(
    progress_updates(
      thiessen_control(general_params = general_params(burn_in = 1, draws = 2)),
      chains = 2
    ),
    6L
  )
  expect_identical(
    progress_updates(
      thiessen_control(
        general_params = general_params(burn_in = 100, draws = 100)
      )
    ),
    100L
  )
})

# progressr's terminal handlers write nothing in a non-interactive session,
# so an assertion on captured output reports on the printed fit rather than
# on the progress bar. The counts above are the check on the mechanism; this
# one is that the handler pipeline runs at all.
test_that("a fit runs under a handler", {
  fixture <- small_fixture()

  fit <- progressr::with_progress(
    thiessen(fixture$x, fixture$y, small_control(), seed = 1),
    handlers = progressr::handler_void()
  )

  expect_s3_class(fit, "thiessen")
})

test_that("nothing is printed without a handler", {
  fixture <- small_fixture()

  expect_silent(thiessen(fixture$x, fixture$y, small_control(), seed = 1))
})

test_that("reporting progress does not change the draws", {
  fixture <- small_fixture()

  plain <- thiessen(fixture$x, fixture$y, small_control(), seed = 1)
  reported <- progressr::with_progress(
    thiessen(fixture$x, fixture$y, small_control(), seed = 1),
    handlers = progressr::handler_void()
  )

  expect_identical(fitted(reported), fitted(plain))
  expect_identical(thiessen_diagnostics(reported), thiessen_diagnostics(plain))
})
