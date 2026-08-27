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
  thiessen_control(
    tessellations = 8,
    general_params = general_params(burn_in = 10, draws = 20),
    ...
  )
}

# The response shapes of the experimental families over a numeric
# response `y`: a right-censored `Surv` with a third of the rows censored
# at their own times, an `interval2` `Surv` with one-sided censoring on
# every sixth and seventh row, and a three-level ordered factor.
right_censored_fixture <- function(y) {
  survival::Surv(exp(y), rep(c(TRUE, TRUE, FALSE), length.out = length(y)))
}

interval_fixture <- function(y) {
  n <- length(y)
  lower <- ifelse(seq_len(n) %% 7 == 0, NA, y - 0.1)
  upper <- ifelse(seq_len(n) %% 6 == 0, NA, y + 0.1)
  survival::Surv(lower, upper, type = "interval2")
}

ordered_fixture <- function(n) {
  factor(
    c("lo", "mid", "hi")[(seq_len(n) %% 3) + 1],
    levels = c("lo", "mid", "hi"), ordered = TRUE
  )
}

# The outcome families the core in use carries, at their defaults, as the
# core reports them.
core_catalogue <- function() {
  jsonlite::fromJSON(core_outcome_defaults(), simplifyVector = FALSE)
}

# The families of `core_catalogue()` by their stored names.
core_families <- function() {
  vapply(core_catalogue(), function(family) names(family)[[1L]], character(1))
}

# Replace the first occurrence of the name `from` in a raw MessagePack
# payload with `to`. A name of fewer than 32 bytes is a fixstr: one byte
# holding 0xa0 plus the length, then the bytes, so the prefix is rewritten
# with the name and the payload still frames.
swap_payload_name <- function(payload, from, to) {
  stopifnot(nchar(from) < 32L, nchar(to) < 32L)
  from <- c(as.raw(160L + nchar(from)), charToRaw(from))
  to <- c(as.raw(160L + nchar(to)), charToRaw(to))
  span <- length(from) - 1L
  for (start in which(payload == from[[1L]])) {
    if (start + span <= length(payload) &&
          identical(payload[start:(start + span)], from)) {
      return(c(payload[seq_len(start - 1L)], to,
               payload[-seq_len(start + span)]))
    }
  }
  stop("the name is not in the payload")
}
