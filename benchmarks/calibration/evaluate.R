# Nightly evaluation of the calibration output: rank ECDF difference
# plots with simultaneous confidence bands (Saeilynoja, Buerkner and
# Vehtari 2022) through the SBC package, and Geweke sample comparison
# plots. Reads every `sbc_ranks*.csv` and `geweke_samples*.csv` the
# full-size Rust tests write (one pair per model) and puts the plots next
# to them.
#
# Usage: Rscript evaluate.R <calibration-dir>

library(SBC)
library(ggplot2)

args <- commandArgs(trailingOnly = TRUE)
dir <- if (length(args) >= 1) args[[1]] else "target/calibration"

for (file in list.files(dir, pattern = "^sbc_ranks.*\\.csv$", full.names = TRUE)) {
  stem <- sub("\\.csv$", "", basename(file))
  ranks <- read.csv(file)
  max_rank <- max(ranks$max_rank)
  stats <- data.frame(
    sim_id = ave(seq_along(ranks$rank), ranks$quantity, FUN = seq_along),
    variable = ranks$quantity,
    rank = ranks$rank,
    max_rank = max_rank
  )
  p <- plot_ecdf_diff(stats)
  out <- file.path(dir, sub("sbc_ranks", "sbc_ecdf_diff", stem))
  ggsave(paste0(out, ".png"), p, width = 9, height = 6, dpi = 150)
  cat("wrote", paste0(out, ".png"), "\n")
  p <- plot_rank_hist(stats)
  out <- file.path(dir, sub("sbc_ranks", "sbc_rank_hist", stem))
  ggsave(paste0(out, ".png"), p, width = 9, height = 6, dpi = 150)
  cat("wrote", paste0(out, ".png"), "\n")
}

for (file in list.files(dir, pattern = "^geweke_samples.*\\.csv$", full.names = TRUE)) {
  stem <- sub("\\.csv$", "", basename(file))
  geweke <- read.csv(file)
  p <- ggplot(geweke, aes(value, colour = simulator)) +
    stat_ecdf(pad = FALSE) +
    facet_wrap(~quantity, scales = "free_x") +
    labs(
      x = "value",
      y = "empirical CDF",
      title = "Geweke marginal-conditional against successive-conditional"
    )
  out <- file.path(dir, sub("geweke_samples", "geweke_ecdf", stem))
  ggsave(paste0(out, ".png"), p, width = 9, height = 6, dpi = 150)
  cat("wrote", paste0(out, ".png"), "\n")
}
