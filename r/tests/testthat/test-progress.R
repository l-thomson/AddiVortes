test_that("the core calls the report function as many times as asked", {
  fixture <- small_fixture()
  calls <- 0L

  core_fit(config_json(small_control()), fixture$x, fixture$y, 1, 1L,
           function() calls <<- calls + 1L, 7L)

  expect_identical(calls, 7L)
})

test_that("the number of updates does not exceed the sweeps", {
  expect_identical(
    progress_reporter(
      thiessen_control(general_params = general_params(burn_in = 1, draws = 2))
    )$updates,
    3L
  )
  expect_identical(
    progress_reporter(
      thiessen_control(general_params = general_params(burn_in = 1, draws = 2)),
      chains = 2
    )$updates,
    6L
  )
  expect_identical(
    progress_reporter(
      thiessen_control(
        general_params = general_params(burn_in = 100, draws = 100)
      )
    )$updates,
    100L
  )
})

test_that("a handler reports the progress of a fit", {
  fixture <- small_fixture()

  output <- capture.output(
    progressr::with_progress(
      thiessen(fixture$x, fixture$y, small_control(), seed = 1),
      handlers = progressr::handler_txtprogressbar()
    )
  )

  expect_true(any(nzchar(output)))
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
