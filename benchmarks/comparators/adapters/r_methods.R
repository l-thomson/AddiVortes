# The three R comparators: CRAN AddiVortes, dbarts and BART.
#
#   Rscript adapters/r_methods.R <method> <train.csv> <test.csv> <out-dir> \
#     <seed> <burn_in> <draws> <chains> <ensemble> <declared_rows>
#
# Each writes `draws.csv` and `meta.json` in the shape every adapter
# writes. None of the three takes a chain count, so a chain here is an
# independent fit at its own seed, which is what the comparison needs: the
# potential scale reduction is computed over runs that share nothing but
# the data.

script_dir <- function() {
  called <- commandArgs(trailingOnly = FALSE)
  file <- sub("^--file=", "", called[grep("^--file=", called)])
  if (length(file) == 1L) dirname(normalizePath(file)) else "."
}
source(file.path(script_dir(), "common.R"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 10) {
  stop("usage: r_methods.R <method> <train> <test> <out> <seed> <burn_in> ",
       "<draws> <chains> <ensemble> <declared_rows>")
}
method <- args[[1]]
train_path <- args[[2]]
test_path <- args[[3]]
out_dir <- args[[4]]
seed <- as.integer(args[[5]])
burn_in <- as.integer(args[[6]])
draws <- as.integer(args[[7]])
chains <- as.integer(args[[8]])
ensemble <- as.integer(args[[9]])
declared_rows <- as.integer(args[[10]])

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
train <- read_csv_design(train_path)
test <- read_csv_design(test_path)

# Per-draw f at `x_new` on the caller's scale: the loop of
# predict.AddiVortes without the summarising, through the exported
# cellIndices. Non-Euclidean columns stay on the caller's scale, as
# predict.AddiVortes leaves them.
addivortes_f <- function(fit, x_new) {
  x_new <- as.matrix(x_new)
  x_scaled <- sweep(sweep(x_new, 2, fit$xCentres), 2, fit$xRanges, "/")
  metric_aug <- rep(fit$metric_red, fit$member_red)
  x_scaled[, metric_aug != 0] <- x_new[, metric_aug != 0]
  kept <- length(fit$posteriorTess)
  out <- matrix(0, kept, nrow(x_new))
  for (s in seq_len(kept)) {
    total <- numeric(nrow(x_new))
    for (j in seq_along(fit$posteriorTess[[s]])) {
      idx <- AddiVortes::cellIndices(
        x_scaled, fit$posteriorTess[[s]][[j]], fit$posteriorDim[[s]][[j]],
        fit$metric_red, fit$member_red
      )
      total <- total + fit$posteriorPred[[s]][[j]][idx]
    }
    out[s, ] <- total * fit$yRange + fit$yCentre
  }
  out
}

run_chain <- function(chain_seed) {
  if (method == "addivortes") {
    set.seed(chain_seed)
    started <- proc.time()[["elapsed"]]
    fit <- AddiVortes::AddiVortes(
      train$y, train$x, m = ensemble,
      totalMCMCIter = burn_in + draws, mcmcBurnIn = burn_in,
      showProgress = FALSE
    )
    fit_seconds <- proc.time()[["elapsed"]] - started
    started <- proc.time()[["elapsed"]]
    f <- addivortes_f(fit, test$x)
    predict_seconds <- proc.time()[["elapsed"]] - started
    sigma <- sqrt(as.numeric(fit$posteriorSigma)) * fit$yRange
    cells <- mean(vapply(
      fit$posteriorTess,
      function(draw) mean(vapply(draw, nrow, numeric(1))),
      numeric(1)
    ))
  } else if (method == "dbarts") {
    set.seed(chain_seed)
    started <- proc.time()[["elapsed"]]
    fit <- dbarts::bart(
      x.train = train$x, y.train = train$y, x.test = test$x,
      ntree = ensemble, ndpost = draws, nskip = burn_in,
      keeptrees = FALSE, verbose = FALSE
    )
    fit_seconds <- proc.time()[["elapsed"]] - started
    # The test predictions come out of the fit, so there is no separate
    # predict pass to time.
    predict_seconds <- NA_real_
    f <- fit$yhat.test
    sigma <- tail(as.numeric(fit$sigma), draws)
    cells <- NA_real_
  } else if (method == "bart") {
    set.seed(chain_seed)
    started <- proc.time()[["elapsed"]]
    fit <- BART::wbart(
      x.train = train$x, y.train = train$y, x.test = test$x,
      ntree = ensemble, ndpost = draws, nskip = burn_in, printevery = draws + 1L
    )
    fit_seconds <- proc.time()[["elapsed"]] - started
    predict_seconds <- NA_real_
    f <- fit$yhat.test
    sigma <- tail(as.numeric(fit$sigma), draws)
    cells <- NA_real_
  } else {
    stop("unknown method ", method)
  }
  list(f = f, sigma = sigma, fit_seconds = fit_seconds,
       predict_seconds = predict_seconds, cells = cells)
}

runs <- lapply(seq_len(chains), function(k) run_chain(seed + k - 1L))

f <- do.call(rbind, lapply(runs, function(r) r$f))
sigma <- unlist(lapply(runs, function(r) r$sigma))

declared <- min(declared_rows, ncol(f))
series <- list()
for (i in seq_len(declared)) {
  series[[sprintf("f[%d]", i - 1L)]] <-
    do.call(rbind, lapply(runs, function(r) r$f[, i]))
}
series[["sigma"]] <- do.call(rbind, lapply(runs, function(r) r$sigma))
write_draws(file.path(out_dir, "draws.csv"), series)

version <- switch(
  method,
  addivortes = packageVersion("AddiVortes"),
  dbarts = packageVersion("dbarts"),
  bart = packageVersion("BART")
)
metrics <- accuracy(f, sigma, test$y, seed)
meta <- c(
  list(
    method = method,
    version = as.character(version),
    chains = chains,
    draws = draws,
    burn_in = burn_in,
    ensemble = ensemble,
    fit_seconds = sum(vapply(runs, function(r) r$fit_seconds, numeric(1))),
    predict_seconds = mean(vapply(runs, function(r) r$predict_seconds, numeric(1))),
    # None of the three separates warm-up from sampling in its own timing,
    # so the post-warm-up share is apportioned by sweep count. Stated
    # rather than hidden: it assumes a sweep costs the same in both
    # phases, which is true of these samplers and not of every sampler.
    warmup_seconds = sum(vapply(runs, function(r) r$fit_seconds, numeric(1))) *
      burn_in / (burn_in + draws),
    post_warmup_seconds = sum(vapply(runs, function(r) r$fit_seconds, numeric(1))) *
      draws / (burn_in + draws),
    cells_per_tessellation = mean(vapply(runs, function(r) r$cells, numeric(1)))
  ),
  metrics
)
write_meta(file.path(out_dir, "meta.json"), meta)
