# Whether the core was built with its experimental feature

The outcome families and options the core crate keeps behind its
`experimental` Cargo feature are absent from a build without it, so a
configuration or a saved fit naming one is rejected with the core's
message naming the feature. This package enables the feature in no
build, so the answer is `FALSE` in every released version; report it
with
[`core_version()`](https://l-thomson.github.io/thiessen/r/reference/core_version.md)
in a bug report, since a fit rejected for naming a gated option looks
the same either way.

## Usage

``` r
core_experimental()
```

## Value

A logical of length one: whether the core in use was built with the
`experimental` feature.

## Examples

``` r
core_experimental()
#> [1] FALSE
```
