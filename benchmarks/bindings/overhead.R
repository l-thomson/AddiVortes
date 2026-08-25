# The R binding's four cases, against the core's time on the same work.
#
# Usage, from the repository root:
#
#   cargo run --release --manifest-path bench/Cargo.toml --bin overhead -- \
#     designs target/bindings
#   cargo run --release --manifest-path bench/Cargo.toml --bin overhead -- \
#     run > target/bindings/core.json
#   Rscript benchmarks/bindings/overhead.R target/bindings
#
# The designs come from the core, so the two sides run on the same numbers
# rather than on two generators that happen to share a name. Run it on one
# machine, old revision against new, and put the table in the pull request:
# that is the record, and there is nothing stored to go stale.
#
# `bench::mark` verifies that the compared expressions return equal values
# before timing them, which is why the per-call and batched sampler cases
# are marked `check = FALSE`: they return NULL and differ only in how many
# times the boundary was crossed.

options(width = 120)

library(bench)
library(jsonlite)
library(progressr)
library(thiessen)

args <- commandArgs(trailingOnly = TRUE)
dir <- if (length(args) >= 1) args[[1]] else "target/bindings"

core <- fromJSON(file.path(dir, "core.json"))
core_case <- function(case, n, p) {
  row <- core$cases[core$cases$case == case & core$cases$n == n &
                      core$cases$p == p, ]
  if (nrow(row) != 1L) stop("no core time for ", case, " at n=", n, " p=", p)
  row$seconds
}

# The schedule and the ensemble of the core's registry workload; the
# comparison is void if these drift apart.
control <- thiessen_control(
  mean_params = term_params(tessellations = 200),
  general_params = general_params(burn_in = 20, draws = 50)
)
sizes <- unique(core$cases[, c("n", "p")])

read_design <- function(path) {
  frame <- read.csv(path)
  as.matrix(frame)
}

rows <- list()
for (i in seq_len(nrow(sizes))) {
  n <- sizes$n[[i]]
  p <- sizes$p[[i]]
  train <- read_design(file.path(dir, sprintf("train-n%d-p%d.csv", n, p)))
  x <- train[, seq_len(p), drop = FALSE]
  y <- train[, p + 1L]
  new_x <- read_design(file.path(dir, sprintf("predict-p%d.csv", p)))

  fit <- thiessen(x, y, control, seed = 1)

  timed <- bench::mark(
    fit = thiessen(x, y, control, seed = 1),
    check = FALSE, min_iterations = 3, filter_gc = FALSE
  )
  rows[[length(rows) + 1L]] <- data.frame(
    case = "fit", n = n, p = p,
    seconds = as.numeric(timed$median),
    allocated = as.numeric(timed$mem_alloc),
    gc = timed$n_gc[[1]]
  )

  timed <- bench::mark(
    predict = predict(fit, new_x),
    check = FALSE, min_iterations = 3, filter_gc = FALSE
  )
  rows[[length(rows) + 1L]] <- data.frame(
    case = "predict", n = n, p = p,
    seconds = as.numeric(timed$median),
    allocated = as.numeric(timed$mem_alloc),
    gc = timed$n_gc[[1]]
  )

  sweeps <- core$sweeps
  timed <- bench::mark(
    per_call = {
      sampler <- thiessen_sampler(x, y, control, seed = 1)
      for (s in seq_len(sweeps)) sampler$step(1)
    },
    batched = {
      sampler <- thiessen_sampler(x, y, control, seed = 1)
      sampler$step(sweeps)
    },
    check = FALSE, min_iterations = 3, filter_gc = FALSE
  )
  # `bench::mark` keeps the rows in the order the expressions were given.
  names <- c("sweeps_per_call", "sweeps_batched")
  for (k in seq_len(nrow(timed))) {
    rows[[length(rows) + 1L]] <- data.frame(
      case = names[[k]],
      n = n, p = p,
      seconds = as.numeric(timed$median[[k]]),
      allocated = as.numeric(timed$mem_alloc[[k]]),
      gc = timed$n_gc[[k]]
    )
  }

  # `handlers("void")` still raises one progression per report, so this
  # measures the reporting machinery rather than a terminal write.
  timed <- bench::mark(
    progress = with_progress(thiessen(x, y, control, seed = 1),
                             handlers = handlers("void", append = FALSE)),
    check = FALSE, min_iterations = 3, filter_gc = FALSE
  )
  rows[[length(rows) + 1L]] <- data.frame(
    case = "fit_progress", n = n, p = p,
    seconds = as.numeric(timed$median),
    allocated = as.numeric(timed$mem_alloc),
    gc = timed$n_gc[[1]]
  )
}

table <- do.call(rbind, rows)

# The core has no per-call case: crossing the boundary once per sweep and
# once for all of them is the same loop to it, so both R rows are read
# against the core's `sweeps`.
core_name <- c(
  fit = "fit", predict = "predict", fit_progress = "fit_progress",
  sweeps_per_call = "sweeps", sweeps_batched = "sweeps"
)
table$core_seconds <- mapply(
  function(case, n, p) core_case(core_name[[case]], n, p),
  table$case, table$n, table$p
)
# The absolute difference beside the ratio: a ratio on a call of a few
# hundred microseconds carries no information.
table$overhead_seconds <- table$seconds - table$core_seconds
table$ratio <- table$seconds / table$core_seconds

table$seconds <- signif(table$seconds, 4)
table$core_seconds <- signif(table$core_seconds, 4)
table$overhead_seconds <- signif(table$overhead_seconds, 4)
table$ratio <- round(table$ratio, 3)
table$allocated_mb <- round(table$allocated / 1e6, 1)
table$allocated <- NULL

cat("R binding against the core, one machine, one session\n")
cat("core", core$core_version, "; sweeps", core$sweeps,
    "; predict rows", core$predict_rows, "\n\n")
print(table, row.names = FALSE)
