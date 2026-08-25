# The parameter groups of a configuration: nested control constructors,
# the `rpart.control()` pattern, so each group documents and validates its
# own arguments. Every name is a name in the core's stored configuration.

#' One ensemble of tessellations
#'
#' The size, priors and covariate space of one ensemble.
#' `thiessen_control()` takes one as `mean_params` and, for the
#' heteroscedastic model, one as `variance_params`.
#'
#' @param tessellations Number of tessellations in the ensemble. `NULL`,
#'   the default, resolves at fit to 200 as `mean_params` and to 0 as
#'   `variance_params`; a positive count as `variance_params` selects the
#'   heteroscedastic model (the paper's count is 40).
#' @param k Cell-value prior spread k: sigma_mu = w / (k sqrt(m)) with the
#'   half-width w the outcome family owns (Chipman, George and McCulloch
#'   2010, s. 4). Default 3. The variance ensemble's inverse-gamma cells
#'   do not use it.
#' @param lambda_c Cell-count prior rate lambda_c:
#'   b - 1 ~ Poisson(lambda_c). Default 5, following AddiVortes 0.6.8 and
#'   later; the paper reports 25.
#' @param geometry The covariate space, from [geometry_params()]. `NULL`,
#'   the default, takes the core's defaults. The ensembles share one
#'   covariate space: set it on `mean_params` and it applies to
#'   `variance_params` as well.
#' @param structure The covariate-inclusion prior, from
#'   [structure_params()]. `NULL` takes the core's defaults. Shared
#'   between the ensembles like `geometry`.
#'
#' @return An object of class `"term_params"`.
#'
#' @seealso [thiessen_control()], [geometry_params()], [structure_params()]
#' @examples
#' term_params(tessellations = 200, lambda_c = 25)
#' @export
term_params <- function(tessellations = NULL, k = 3, lambda_c = 5,
                        geometry = NULL, structure = NULL) {
  check_whole_number(tessellations, min = 0, allow_null = TRUE)
  check_number(k)
  check_number(lambda_c)
  check_group(geometry, "geometry", "geometry_params", null_ok = TRUE)
  check_group(structure, "structure", "structure_params", null_ok = TRUE)
  # The `structure` argument shadows `base::structure` in this body.
  group <- list(
    tessellations = tessellations, k = k, lambda_c = lambda_c,
    geometry = geometry, structure = structure
  )
  class(group) <- "term_params"
  group
}

#' The covariate space of the ensembles
#'
#' @param metric The metric of each covariate column, in column order: a
#'   list whose entries are `"euclidean"`, `"categorical"`, or
#'   `list(spherical = list(sphere = i))` for one coordinate of the sphere
#'   labelled `i`, its latitudes first and its longitude last, in radians.
#'   `NULL`, the default, is Euclidean on every column. Entries are
#'   matched case-sensitively, so `"Euclidean"` is rejected. Non-Euclidean
#'   columns are not scaled.
#' @param sigma_c Prior and proposal standard deviation sigma_c of a
#'   centre coordinate. A Euclidean column is min-max scaled to
#'   \[-0.5, 0.5\] over its training range inside the sampler and `sigma_c`
#'   is on that scale, so 1 is the full range of a column. Default 0.8.
#'
#' @return An object of class `"geometry_params"`.
#'
#' @seealso [term_params()]
#' @examples
#' geometry_params(metric = list("euclidean", "categorical"))
#' @export
geometry_params <- function(metric = NULL, sigma_c = 0.8) {
  check_number(sigma_c)
  if (!is.null(metric)) metric <- as.list(metric)
  structure(list(metric = metric, sigma_c = sigma_c),
            class = "geometry_params")
}

#' The covariate-inclusion prior of the ensembles
#'
#' @param omega Dimension-count prior parameter omega; omega / p is the
#'   prior probability of including a covariate. `NULL`, the default,
#'   resolves to min(3, p) at fit. Must satisfy 0 < omega <= p.
#'
#' @return An object of class `"structure_params"`.
#'
#' @seealso [term_params()]
#' @examples
#' structure_params(omega = 2)
#' @export
structure_params <- function(omega = NULL) {
  check_number(omega, allow_null = TRUE)
  structure(list(omega = omega), class = "structure_params")
}

#' The sweep schedule of a fit
#'
#' @param burn_in Sweeps discarded before the kept draws. Default 200.
#' @param draws Posterior draws kept. Default 1000.
#' @param thinning Keep every `thinning`-th sweep after burn-in. Default 1.
#' @param prior_only Switch off the likelihood, so the chain draws from
#'   the prior and `predict()` gives prior predictive draws. Default
#'   `FALSE`.
#'
#' @return An object of class `"general_params"`.
#'
#' @seealso [thiessen_control()]
#' @examples
#' general_params(burn_in = 100, draws = 500)
#' @export
general_params <- function(burn_in = 200, draws = 1000, thinning = 1,
                           prior_only = FALSE) {
  check_whole_number(burn_in, min = 0)
  check_whole_number(draws, min = 0)
  check_whole_number(thinning, min = 0)
  check_flag(prior_only)
  structure(
    list(burn_in = burn_in, draws = draws, thinning = thinning,
         prior_only = prior_only),
    class = "general_params"
  )
}

#' @export
format.term_params <- function(x, ...) {
  constructor_call("term_params", compact(unclass(x)))
}

#' @export
format.geometry_params <- function(x, ...) {
  constructor_call("geometry_params", compact(unclass(x)))
}

#' @export
format.structure_params <- function(x, ...) {
  constructor_call("structure_params", compact(unclass(x)))
}

#' @export
format.general_params <- function(x, ...) {
  constructor_call("general_params", compact(unclass(x)))
}

#' Print a parameter group
#'
#' @param x A parameter group from [term_params()], [geometry_params()],
#'   [structure_params()] or [general_params()].
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @examples
#' print(term_params(tessellations = 40))
#' @name print.params
NULL

#' @rdname print.params
#' @export
print.term_params <- function(x, ...) {
  cat(format(x), "\n", sep = "")
  invisible(x)
}

#' @rdname print.params
#' @export
print.geometry_params <- print.term_params

#' @rdname print.params
#' @export
print.structure_params <- print.term_params

#' @rdname print.params
#' @export
print.general_params <- print.term_params

#' Drop the `NULL` entries of a list
#'
#' @param fields A named list.
#' @return The list without its `NULL` entries.
#' @noRd
compact <- function(fields) {
  fields[!vapply(fields, is.null, logical(1))]
}

#' Render a constructor call from a group's set fields
#'
#' A nested group renders as its own call; every other value is deparsed,
#' so the string parses back to the object.
#'
#' @param name The constructor's name.
#' @param fields The non-`NULL` fields.
#' @return A character string, `name(field = value, ...)`.
#' @noRd
constructor_call <- function(name, fields) {
  shown <- vapply(
    fields,
    function(value) {
      if (inherits(value, "term_params") ||
            inherits(value, "geometry_params") ||
            inherits(value, "structure_params")) {
        format(value)
      } else {
        deparse1(value)
      }
    },
    character(1)
  )
  arguments <- paste(names(fields), "=", shown, collapse = ", ")
  if (length(fields) == 0L) arguments <- ""
  paste0(name, "(", arguments, ")")
}
