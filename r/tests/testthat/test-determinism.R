# The draws of a fit against the chain the core commits. Bit-exact on the
# reference target x86_64-unknown-linux-gnu, which is where the core checks
# its own snapshot; the comparison is skipped elsewhere.

# The fixture rows the core records f(x) at, one-based.
POINTS <- c(1L, 18L, 34L)

stored_chain <- function(name = "core-gaussian-chain.txt") {
  lines <- readLines(test_path(name))
  body <- lines[-1]
  body <- body[nzchar(body)]
  matrix(
    as.double(unlist(strsplit(body, " ", fixed = TRUE))),
    nrow = length(body), byrow = TRUE
  )
}

skip_off_reference_target <- function() {
  skip_if_not(
    Sys.info()[["sysname"]] == "Linux" && R.version$arch == "x86_64",
    "the chain is bit-exact on x86_64-unknown-linux-gnu only"
  )
}

core_control <- function(...) {
  thiessen_control(
    tessellations = 15,
    general_params = general_params(burn_in = 50, draws = 60),
    ...
  )
}

test_that("the draws equal the core's committed chain", {
  skip_off_reference_target()
  fixture <- core_fixture()
  stored <- stored_chain()

  fit <- thiessen(fixture$x, fixture$y, core_control(), seed = CORE_SEED)

  expect_identical(fit$n_draws, nrow(stored))
  expect_identical(core_sigma(fit$state), stored[, 1])
  expect_identical(
    unname(predict(fit, fixture$x[POINTS, ], type = "draws")),
    stored[, -1]
  )
})


test_that("the probit draws equal the core's committed chain", {
  skip_off_reference_target()
  fixture <- core_fixture()
  stored <- stored_chain("core-probit-chain.txt")
  # The core's fixture thresholds at the upper middle order statistic.
  threshold <- sort(fixture$y)[length(fixture$y) / 2 + 1]
  labels <- as.double(fixture$y >= threshold)

  fit <- thiessen(fixture$x, labels, core_control(outcome = probit()),
                  seed = CORE_SEED)

  expect_identical(fit$n_draws, nrow(stored))
  expect_identical(
    unname(predict(fit, fixture$x[POINTS, ], type = "latent")),
    stored
  )
})

test_that("the heteroscedastic draws equal the core's committed chain", {
  skip_off_reference_target()
  fixture <- core_fixture()
  stored <- stored_chain("core-heteroscedastic-chain.txt")
  i <- seq_along(fixture$y) - 1L
  noise <- 0.3 * (((i * 29) %% 17) / 16 - 0.5)
  y <- fixture$y - noise + noise * (0.2 + 2 * fixture$x[, 1])

  fit <- thiessen(
    fixture$x, y,
    core_control(variance_params = term_params(tessellations = 5)),
    seed = CORE_SEED
  )

  expect_identical(fit$n_draws, nrow(stored))
  expect_identical(
    unname(predict(fit, fixture$x[POINTS, ], type = "draws")),
    stored[, 1:3]
  )
  expect_identical(
    unname(predict(fit, fixture$x[POINTS, ], type = "variance")),
    stored[, 4:6]
  )
})
