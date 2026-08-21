# The core's fixed-seed fixture, reproduced so the determinism test can
# compare against the chain the crate commits.

# The seed of the core's fixture.
CORE_SEED <- 7

core_fixture <- function() {
  n <- 48
  i <- seq_len(n) - 1L
  r0 <- i / (n - 1)
  r1 <- ((i * 37) %% n) / n
  # The multiplications associate as the core's fixture does; 3 * d^2 rounds
  # differently from (3 * d) * d and moves the chain.
  f <- 3 * (r0 - 0.4) * (r0 - 0.4) + 0.5 * r1
  list(
    x = unname(cbind(r0, r1)),
    y = f + 0.3 * (((i * 29) %% 17) / 16 - 0.5)
  )
}

# A small design and response for the tests that do not compare draws.
small_fixture <- function(n = 40) {
  x <- cbind(seq(0, 1, length.out = n), rep(c(0, 0.25, 0.5, 0.75), length.out = n))
  list(x = x, y = 2 * (x[, 1] - 0.5)^2 + 0.5 * x[, 2])
}

# The sweep schedule the tests share.
small_control <- function(...) {
  thiessen_control(m = 8, burn_in = 10, draws = 20, ...)
}
