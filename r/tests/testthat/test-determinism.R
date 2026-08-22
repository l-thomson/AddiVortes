# The draws of a fit against the chain the core commits. Bit-exact on the
# reference target x86_64-unknown-linux-gnu, which is where the core checks
# its own snapshot; the comparison is skipped elsewhere.

# The fixture rows the core records f(x) at, one-based.
POINTS <- c(1L, 18L, 34L)

stored_chain <- function() {
  lines <- readLines(test_path("core-gaussian-chain.txt"))
  body <- lines[-1]
  body <- body[nzchar(body)]
  matrix(
    as.double(unlist(strsplit(body, " ", fixed = TRUE))),
    nrow = length(body), byrow = TRUE
  )
}

test_that("the draws equal the core's committed chain", {
  skip_if_not(
    Sys.info()[["sysname"]] == "Linux" && R.version$arch == "x86_64",
    "the chain is bit-exact on x86_64-unknown-linux-gnu only"
  )
  fixture <- core_fixture()
  stored <- stored_chain()

  fit <- thiessen(
    fixture$x, fixture$y,
    thiessen_control(m = 15, burn_in = 50, draws = 60),
    seed = CORE_SEED
  )

  expect_identical(fit$n_draws, nrow(stored))
  expect_identical(core_sigma(fit$state), stored[, 1])
  expect_identical(
    unname(predict(fit, fixture$x[POINTS, ], type = "draws")),
    stored[, -1]
  )
})
