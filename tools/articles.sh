#!/bin/sh
# Knits the precomputed articles under r/vignettes/articles. The landing
# page is knitted from a default build and the feature articles from an
# opt-in build, each installed into a library of its own, so every page
# shows the behaviour of the build it names. Each `<name>.Rmd.orig` is
# knitted to `<name>.Rmd` with its figures under `figures/<name>/`; the
# `.Rmd` executes nothing, so the site build needs neither build.
#
# Usage: tools/articles.sh [landing|features|all]
set -eu

root=$(cd "$(dirname "$0")/.." && pwd)
articles="$root/r/vignettes/articles"
what=${1:-all}

knit() {
  lib=$1
  shift
  for orig in "$@"; do
    name=$(basename "$orig" .Rmd.orig)
    R_LIBS="$lib" Rscript -e "
      setwd('$articles')
      knitr::opts_chunk[['set']](fig.path = 'figures/$name/')
      knitr::knit('$name.Rmd.orig', '$name.Rmd', quiet = TRUE)
    "
  done
}

if [ "$what" = landing ] || [ "$what" = all ]; then
  lib=$(mktemp -d)
  R CMD INSTALL --no-multiarch -l "$lib" "$root/r"
  knit "$lib" "$articles/experimental.Rmd.orig"
fi

if [ "$what" = features ] || [ "$what" = all ]; then
  features=$(ls "$articles"/*.Rmd.orig | grep -v '/experimental\.Rmd\.orig$' || true)
  if [ -n "$features" ]; then
    lib=$(mktemp -d)
    THIESSEN_EXPERIMENTAL=1 R CMD INSTALL --no-multiarch -l "$lib" "$root/r"
    knit "$lib" $features
  fi
fi
