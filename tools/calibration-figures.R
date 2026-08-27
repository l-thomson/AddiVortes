# Copies the calibration figures of one stem from a downloaded nightly
# `calibration-report` artefact into an article's figure directory and
# writes `battery.md` beside them: the run id and the battery parameters
# read from the artefact's CSVs, for the article's caption.
#
# Usage: Rscript tools/calibration-figures.R <artefact-dir> <stem> <article> <run-id>
#
# `stem` is the model's suffix in the artefact (`soft`, `linear`,
# `dart`, `student_t`, `laplace`, `aft`, `tobit`, `interval_censored`,
# `ordinal`), `article` the article's file name without extension, and
# `run-id` the id of the nightly run the artefact was downloaded from.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("usage: Rscript tools/calibration-figures.R <artefact-dir> <stem> <article> <run-id>")
}
artefact <- args[[1]]
stem <- args[[2]]
article <- args[[3]]
run_id <- args[[4]]

root <- normalizePath(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), ".."))
target <- file.path(root, "r", "vignettes", "articles", "figures", article)
dir.create(target, recursive = TRUE, showWarnings = FALSE)

figures <- c(
  paste0("sbc_ecdf_diff_", stem, ".png"),
  paste0("geweke_ecdf_", stem, ".png")
)
for (figure in figures) {
  source <- file.path(artefact, figure)
  if (!file.exists(source)) {
    stop("no ", figure, " in ", artefact)
  }
  file.copy(source, file.path(target, figure), overwrite = TRUE)
}

ranks <- read.csv(file.path(artefact, paste0("sbc_ranks_", stem, ".csv")))
geweke <- read.csv(file.path(artefact, paste0("geweke_samples_", stem, ".csv")))
simulations <- max(table(ranks$quantity))
lines <- c(
  paste0("Nightly run ", run_id, ", stem `", stem, "`."),
  paste0(
    "SBC: ", simulations, " simulations, ", max(ranks$max_rank),
    " posterior draws each, quantities ",
    paste(unique(ranks$quantity), collapse = ", "), "."
  ),
  paste0(
    "Geweke: ", max(table(geweke$simulator, geweke$quantity)),
    " samples per simulator, quantities ",
    paste(unique(geweke$quantity), collapse = ", "), "."
  )
)
writeLines(lines, file.path(target, "battery.md"))
cat(lines, sep = "\n")
