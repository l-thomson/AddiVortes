# What every R adapter shares: the metric formulas and the output.
#
# The formulas are `adapters/common.py`'s, and the two must agree: a
# comparison where one method's log predictive density is computed
# differently from another's is a comparison of two formulas.
#
#   rmse      sqrt(mean((mean_d f_di - y_i)^2))
#   lpd       mean_i log(mean_d N(y_i; f_di, sigma_d))
#   coverage  share of y_i inside the 2.5 and 97.5 per cent quantiles of
#             one predictive sample per draw, f_di + sigma_d z_di
#   width     the mean width of those intervals
#
# The predictive sample comes from a stream keyed by the cell's seed and
# an offset, so it cannot share a stream with anything a method used.

PREDICTIVE_SEED_OFFSET <- 982451653L

# `f` is draws by rows; `sigma` one value per draw.
accuracy <- function(f, sigma, y, seed) {
  posterior_mean <- colMeans(f)
  rmse <- sqrt(mean((posterior_mean - y)^2))

  scale <- matrix(sigma, nrow(f), ncol(f))
  target <- matrix(y, nrow(f), ncol(f), byrow = TRUE)
  log_density <- dnorm(target, mean = f, sd = scale, log = TRUE)
  peak <- apply(log_density, 2, max)
  lpd <- mean(peak + log(colMeans(exp(sweep(log_density, 2, peak)))))

  state <- get0(".Random.seed", envir = globalenv(), inherits = FALSE)
  on.exit({
    if (!is.null(state)) .Random.seed <<- state
  })
  set.seed(seed + PREDICTIVE_SEED_OFFSET)
  predictive <- f + scale * matrix(rnorm(length(f)), nrow(f), ncol(f))
  bounds <- apply(predictive, 2, quantile, probs = c(0.025, 0.975), type = 7)

  list(
    rmse = rmse,
    lpd = lpd,
    coverage_95 = mean(y >= bounds[1, ] & y <= bounds[2, ]),
    width_95 = mean(bounds[2, ] - bounds[1, ])
  )
}

# `series` is a named list of chains-by-draws matrices.
write_draws <- function(path, series) {
  rows <- lapply(names(series), function(name) {
    values <- series[[name]]
    data.frame(
      chain = rep(seq_len(nrow(values)) - 1L, times = ncol(values)),
      draw = rep(seq_len(ncol(values)) - 1L, each = nrow(values)),
      quantity = name,
      value = sprintf("%.17e", as.vector(values))
    )
  })
  write.csv(do.call(rbind, rows), path, row.names = FALSE, quote = FALSE)
}

write_meta <- function(path, meta) {
  info <- Sys.info()
  meta$platform <- paste(info[["sysname"]], info[["release"]], info[["machine"]])
  meta$r <- as.character(getRversion())
  writeLines(jsonlite::toJSON(meta, auto_unbox = TRUE, pretty = TRUE), path)
}

read_csv_design <- function(path) {
  frame <- read.csv(path)
  list(x = as.matrix(frame[, -ncol(frame), drop = FALSE]), y = frame[[ncol(frame)]])
}
