# progressr signals nothing in a non-interactive session unless it is
# enabled, and `with_progress()` muffles every condition before a handler
# established outside it runs, so collect the conditions with the option set
# and no progressr handler in the way.
progressions <- function(expr) {
  old <- options(progressr.enable = TRUE)
  on.exit(options(old), add = TRUE)
  seen <- data.frame(
    type = character(0), amount = numeric(0), message = character(0)
  )
  withCallingHandlers(
    expr,
    progression = function(condition) {
      seen[nrow(seen) + 1L, ] <<- list(
        condition$type,
        if (is.null(condition$amount)) NA_real_ else condition$amount,
        paste(conditionMessage(condition), collapse = "")
      )
    }
  )
  seen
}

updates_of <- function(seen) {
  seen[seen$type == "update", ]
}

# The steps the whole fit takes, sweeps and the phases that follow them.
total_advance <- function(seen) {
  as.integer(sum(updates_of(seen)$amount))
}

# The steps the sweeps take: those the first phase after them precedes.
sweep_updates <- function(seen) {
  updates <- updates_of(seen)
  pooling <- which(updates$message == "pooling the draws")[1L]
  as.integer(sum(updates$amount[seq_len(pooling - 1L)]))
}

phase_messages <- function(seen) {
  updates <- updates_of(seen)
  updates$message[nzchar(updates$message)]
}

test_that("a fit signals every progression it requests", {
  fixture <- small_fixture()
  control <- thiessen_control(
    tessellations = 8,
    general_params = general_params(burn_in = 40, draws = 80)
  )

  expect_identical(progress_updates(control), 100L)
  seen <- progressions(thiessen(fixture$x, fixture$y, control, seed = 1))
  expect_identical(sweep_updates(seen), 100L)
  expect_identical(total_advance(seen), progress_steps(control))
})

test_that("a fit shorter than a hundred sweeps signals one per sweep", {
  fixture <- small_fixture()

  expect_identical(progress_updates(small_control()), 30L)
  seen <- progressions(
    thiessen(fixture$x, fixture$y, small_control(), seed = 1)
  )
  expect_identical(sweep_updates(seen), 30L)
})

test_that("the sweeps of every chain are reported", {
  fixture <- small_fixture()

  seen <- progressions(suppressWarnings(
    thiessen(fixture$x, fixture$y, small_control(), chains = 2, seed = 1)
  ))

  expect_identical(sweep_updates(seen), progress_updates(small_control(), 2L))
  expect_identical(total_advance(seen), progress_steps(small_control(), 2L))
})

# The sweeps once took every step of the progressor, which finished the
# report at the last sweep and left pooling, the longest phase of a long
# fit, and the convergence summary running under a closed handler.
test_that("the sweeps leave a step for each phase that follows them", {
  fixture <- small_fixture()

  seen <- progressions(
    thiessen(fixture$x, fixture$y, small_control(), seed = 1)
  )

  expect_identical(
    total_advance(seen) - sweep_updates(seen),
    as.integer(PROGRESS_PHASES)
  )
  expect_identical(
    phase_messages(seen),
    c("sampling", "pooling the draws", "summarising the draws")
  )
})

test_that("the chains of a fit are named as they are sampled", {
  fixture <- small_fixture()

  seen <- progressions(suppressWarnings(
    thiessen(fixture$x, fixture$y, small_control(), chains = 2, seed = 1)
  ))

  expect_identical(
    phase_messages(seen),
    c(
      "sampling chain 1 of 2", "sampling chain 2 of 2",
      "pooling the draws", "summarising the draws"
    )
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

test_that("a sampler assembles a fit without a progressor", {
  fixture <- small_fixture()

  sampler <- thiessen_sampler(fixture$x, fixture$y, small_control(), seed = 1)
  sampler$step(5)
  sampler$keep()

  expect_s3_class(sampler$finish(), "thiessen")
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
