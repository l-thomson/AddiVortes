# progressr signals nothing in a non-interactive session unless it is
# enabled, and `with_progress()` muffles every condition before a handler
# established outside it runs, so collect the conditions with the option set
# and no progressr handler in the way.
progressions <- function(expr) {
  old <- options(progressr.enable = TRUE)
  on.exit(options(old), add = TRUE)
  seen <- data.frame(
    type = character(0), amount = numeric(0), message = character(0),
    sticky = logical(0)
  )
  withCallingHandlers(
    expr,
    progression = function(condition) {
      seen[nrow(seen) + 1L, ] <<- list(
        condition$type,
        if (is.null(condition$amount)) NA_real_ else condition$amount,
        paste(conditionMessage(condition), collapse = ""),
        inherits(condition, "sticky")
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
test_that("the sweeps leave the budget the phases after them need", {
  fixture <- small_fixture()
  sweeps <- progress_updates(small_control())

  seen <- progressions(
    thiessen(fixture$x, fixture$y, small_control(), seed = 1)
  )

  expect_identical(sweep_updates(seen), sweeps)
  expect_identical(
    total_advance(seen) - sweeps,
    POOLING_WEIGHT * sweeps + 1L
  )
  expect_identical(
    phase_messages(seen),
    c("sampling", "pooling the draws", "summarising the draws")
  )
})

# Pooling is one call into the core, so the bar rests where the sweeps
# leave it for as long as pooling runs. A bar that rests all but complete
# reads as a fit that has hung rather than one still working.
test_that("the sweeps leave the bar around a third along, not all but complete", {
  for (chains in 1:2) {
    fraction <- progress_updates(small_control(), chains) /
      progress_steps(small_control(), chains)
    expect_gt(fraction, 0.25)
    expect_lt(fraction, 0.45)
  }
})

# Under a terminal handler a plain message is overwritten by the next bar
# redraw, so only a sticky one leaves the phase on screen.
test_that("every phase names itself in a sticky progression", {
  fixture <- small_fixture()

  seen <- progressions(
    thiessen(fixture$x, fixture$y, small_control(), seed = 1)
  )
  named <- seen[nzchar(seen$message), ]

  expect_identical(
    named$message,
    c("sampling", "pooling the draws", "summarising the draws")
  )
  expect_true(all(named$sticky))
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
