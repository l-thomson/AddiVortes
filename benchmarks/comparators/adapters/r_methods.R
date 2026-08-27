# The three R comparators: upstream AddiVortes, dbarts and BART.
#
#   Rscript adapters/r_methods.R <method> <train.csv> <test.csv> <out-dir> \
#     <seed> <burn_in> <draws> <chains> <ensemble> <declared_rows> <threads>
#
# Each writes `draws.csv` and `meta.json` in the shape every adapter
# writes. A chain is an independent fit at its own seed, which is what
# the comparison needs: the potential scale reduction is computed over
# runs that share nothing but the data. At one thread the chains run in
# turn. At more than one, each method takes the thread count the way its
# own documentation says, never from `OMP_NUM_THREADS`, which none of
# them reads: AddiVortes has no thread option, so its chains fork through
# `parallel::mclapply`; dbarts runs its chains in one call on `nthread`
# threads; BART's `mc.wbart` forks and divides the draws of a chain over
# the cores. `mclapply` and `mc.wbart` fork, so the cores grid is
# Unix-only.
#
# `fit_seconds` is the wall-clock of fitting every chain in each case:
# the sum over the chains run in turn, the time around the `mclapply`
# call (collecting the fits from the workers included, since a user of
# upstream pays that too), the one dbarts call, or the sum of the
# `mc.wbart` calls.

script_dir <- function() {
  called <- commandArgs(trailingOnly = FALSE)
  file <- sub("^--file=", "", called[grep("^--file=", called)])
  if (length(file) == 1L) dirname(normalizePath(file)) else "."
}
source(file.path(script_dir(), "common.R"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 11) {
  stop("usage: r_methods.R <method> <train> <test> <out> <seed> <burn_in> ",
       "<draws> <chains> <ensemble> <declared_rows> <threads>")
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
threads <- as.integer(args[[11]])

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
train <- read_csv_design(train_path)
test <- read_csv_design(test_path)

elapsed <- function() proc.time()[["elapsed"]]

# Per-draw f at `x_new` on the caller's scale: the loop of
# predict.AddiVortes without the summarising, through the exported
# cellIndices. Non-Euclidean columns stay on the caller's scale, as
# predict.AddiVortes leaves them. Not timed: upstream's predict() is one
# C++ call, and timing this loop would misstate it.
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

chain_seeds <- seed + seq_len(chains) - 1L

# One AddiVortes chain: the fit and its own wall-clock. Seeded inside, so
# a forked worker draws the chain the serial run draws.
fit_addivortes <- function(chain_seed) {
  set.seed(chain_seed)
  started <- elapsed()
  fit <- AddiVortes::AddiVortes(
    train$y, train$x, m = ensemble,
    totalMCMCIter = burn_in + draws, mcmcBurnIn = burn_in,
    showProgress = FALSE
  )
  list(fit = fit, fit_seconds = elapsed() - started)
}

# The held-out summaries of one AddiVortes fit. `predict_seconds` is the
# time of upstream's own predict() on the held-out rows; the draws come
# from the loop above.
summarise_addivortes <- function(run) {
  # Forced before the clock starts: an unevaluated argument would run
  # the fit inside the predict timing.
  force(run)
  started <- elapsed()
  invisible(predict(run$fit, newdata = test$x, showProgress = FALSE))
  predict_seconds <- elapsed() - started
  list(
    f = addivortes_f(run$fit, test$x),
    sigma = sqrt(as.numeric(run$fit$posteriorSigma)) * run$fit$yRange,
    fit_seconds = run$fit_seconds,
    predict_seconds = predict_seconds,
    cells = mean(vapply(
      run$fit$posteriorTess,
      function(draw) mean(vapply(draw, nrow, numeric(1))),
      numeric(1)
    ))
  )
}

run_addivortes <- function() {
  # Loaded here, in the parent: a namespace a forked worker loads is not
  # loaded in the parent, and predict() dispatches on the S3 method the
  # namespace registers.
  loadNamespace("AddiVortes")
  if (threads > 1L) {
    started <- elapsed()
    fits <- parallel::mclapply(
      chain_seeds, fit_addivortes, mc.cores = min(threads, chains)
    )
    wall <- elapsed() - started
    runs <- lapply(fits, summarise_addivortes)
    fit_seconds <- wall
  } else {
    runs <- lapply(chain_seeds, function(s) summarise_addivortes(fit_addivortes(s)))
    fit_seconds <- sum(vapply(runs, function(r) r$fit_seconds, numeric(1)))
  }
  list(runs = runs, fit_seconds = fit_seconds)
}

# dbarts: `nthread` pinned explicitly on every grid, since its default is
# the only thing that keeps it single-threaded. At more than one thread
# the chains are one call, each chain on its own thread, kept apart by
# `combinechains = FALSE`. The test predictions come out of the fit, so
# there is no separate predict pass to time.
run_dbarts <- function() {
  if (threads > 1L) {
    set.seed(seed)
    started <- elapsed()
    fit <- dbarts::bart(
      x.train = train$x, y.train = train$y, x.test = test$x,
      ntree = ensemble, ndpost = draws, nskip = burn_in,
      keeptrees = FALSE, verbose = FALSE,
      nchain = chains, nthread = threads, combinechains = FALSE
    )
    fit_seconds <- elapsed() - started
    runs <- lapply(seq_len(chains), function(k) {
      sigma <- fit$sigma
      sigma <- if (is.matrix(sigma)) sigma[k, ] else sigma
      list(
        f = fit$yhat.test[k, , ],
        sigma = tail(as.numeric(sigma), draws),
        predict_seconds = NA_real_, cells = NA_real_
      )
    })
    return(list(runs = runs, fit_seconds = fit_seconds))
  }
  runs <- lapply(chain_seeds, function(chain_seed) {
    set.seed(chain_seed)
    started <- elapsed()
    fit <- dbarts::bart(
      x.train = train$x, y.train = train$y, x.test = test$x,
      ntree = ensemble, ndpost = draws, nskip = burn_in,
      keeptrees = FALSE, verbose = FALSE, nthread = 1L
    )
    list(
      f = fit$yhat.test, sigma = tail(as.numeric(fit$sigma), draws),
      fit_seconds = elapsed() - started,
      predict_seconds = NA_real_, cells = NA_real_
    )
  })
  list(runs = runs, fit_seconds = sum(vapply(runs, function(r) r$fit_seconds, numeric(1))))
}

# BART: `wbart` in turn, or `mc.wbart` per chain at more than one thread,
# which forks `mc.cores` workers and divides the chain's draws between
# them, each worker after the same burn-in. It rounds the draws up to a
# multiple of the cores and stacks the workers' draws in order; the first
# `draws` rows are kept so every chain has the schedule's length.
run_bart <- function() {
  runs <- lapply(chain_seeds, function(chain_seed) {
    started <- elapsed()
    if (threads > 1L) {
      fit <- BART::mc.wbart(
        x.train = train$x, y.train = train$y, x.test = test$x,
        ntree = ensemble, ndpost = draws, nskip = burn_in,
        printevery = draws + 1L, mc.cores = threads, seed = chain_seed
      )
      fit_seconds <- elapsed() - started
      # `sigma` is one column per worker, burn-in included; flattening
      # column-wise puts the workers' draws in the order of `yhat.test`.
      sigma <- as.numeric(fit$sigma[-seq_len(burn_in), , drop = FALSE])
      f <- fit$yhat.test[seq_len(draws), , drop = FALSE]
      sigma <- sigma[seq_len(draws)]
    } else {
      set.seed(chain_seed)
      fit <- BART::wbart(
        x.train = train$x, y.train = train$y, x.test = test$x,
        ntree = ensemble, ndpost = draws, nskip = burn_in,
        printevery = draws + 1L
      )
      fit_seconds <- elapsed() - started
      f <- fit$yhat.test
      sigma <- tail(as.numeric(fit$sigma), draws)
    }
    list(f = f, sigma = sigma, fit_seconds = fit_seconds,
         predict_seconds = NA_real_, cells = NA_real_)
  })
  list(runs = runs, fit_seconds = sum(vapply(runs, function(r) r$fit_seconds, numeric(1))))
}

result <- switch(
  method,
  addivortes = run_addivortes(),
  dbarts = run_dbarts(),
  bart = run_bart(),
  stop("unknown method ", method)
)
runs <- result$runs
fit_seconds <- result$fit_seconds

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
    threads = threads,
    fit_seconds = fit_seconds,
    predict_seconds = mean(vapply(runs, function(r) r$predict_seconds, numeric(1))),
    # None of the three separates warm-up from sampling in its own timing,
    # so the post-warm-up share is apportioned by sweep count. Stated
    # rather than hidden: it assumes a sweep costs the same in both
    # phases, which is true of these samplers and not of every sampler.
    warmup_seconds = fit_seconds * burn_in / (burn_in + draws),
    post_warmup_seconds = fit_seconds * draws / (burn_in + draws),
    cells_per_tessellation = mean(vapply(runs, function(r) r$cells, numeric(1)))
  ),
  metrics
)
write_meta(file.path(out_dir, "meta.json"), meta)
