# Statistical behaviour at tolerances: recovery of a known signal, stability
# over seeds and under noise at machine precision, and finiteness of every
# returned value. The bitwise properties live in test-determinism.R; here
# closeness is the claim, so every comparison carries a tolerance.

recovery_fixture <- function(n = 120) {
  i <- seq_len(n) - 1L
  x <- cbind(i / (n - 1), ((i * 37) %% n) / n)
  list(x = x, f = 2 * (x[, 1] - 0.5)^2 + 0.5 * x[, 2])
}

recovery_control <- function() {
  thiessen_control(
    tessellations = 20,
    general_params = general_params(burn_in = 100, draws = 200)
  )
}

recovery_rmse <- function(fit, f) {
  sqrt(mean((fitted(fit) - f)^2))
}

test_that("the posterior mean recovers a known signal within tolerance", {
  data <- recovery_fixture()
  y <- data$f + 0.1 * sin(seq_along(data$f))

  fit <- thiessen(data$x, y, recovery_control(), seed = 1)

  expect_lt(recovery_rmse(fit, data$f), 0.1)
  expect_true(all(is.finite(fitted(fit))))
  expect_true(all(is.finite(posterior_epred(fit))))
  expect_true(all(is.finite(sigma(fit))))
})

test_that("different seeds agree within tolerance", {
  data <- recovery_fixture()
  y <- data$f + 0.1 * sin(seq_along(data$f))

  fits <- lapply(1:3, function(seed) {
    thiessen(data$x, y, recovery_control(), seed = seed)
  })

  means <- vapply(fits, fitted, numeric(nrow(data$x)))
  spread <- apply(means, 1L, function(row) diff(range(row)))
  expect_lt(max(spread), 0.15)
})

test_that("noise at machine precision does not move the posterior mean far", {
  data <- recovery_fixture()
  y <- data$f + 0.1 * sin(seq_along(data$f))
  jitter <- .Machine$double.eps * (1 + abs(data$x)) *
    rep(c(1, -1), length.out = length(data$x))

  plain <- thiessen(data$x, y, recovery_control(), seed = 1)
  moved <- thiessen(data$x + jitter, y, recovery_control(), seed = 1)

  expect_lt(max(abs(fitted(moved) - fitted(plain))), 0.15)
})

test_that("a response far from zero keeps its scale", {
  data <- recovery_fixture()
  y <- 1000 + 50 * data$f + 5 * sin(seq_along(data$f))

  fit <- thiessen(data$x, y, recovery_control(), seed = 1)

  expect_lt(recovery_rmse(fit, 1000 + 50 * data$f), 5)
  expect_equal(mean(fitted(fit)), mean(y), tolerance = 0.01)
})
