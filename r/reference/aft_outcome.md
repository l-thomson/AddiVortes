# The accelerated failure time outcome family

**\[experimental\]**

## Usage

``` r
aft_outcome(nu = 6, q = 0.85)
```

## Arguments

- nu:

  Degrees of freedom nu of the sigma^2 prior, sigma^2 ~ nu lambda /
  chi^2_nu. Default 6. A variance ensemble requires nu \> 2.

- q:

  Calibration quantile q of the sigma^2 prior, Pr(sigma \< sigma_hat)
  = q. Default 0.85.

## Value

An object of class `c("thiessen_aft", "thiessen_outcome")`.

## Details

The lognormal accelerated failure time model (Wei 1992) for a
right-censored time to event, the model of the BART package's `abart`:
ln T = f(x) + e with e ~ N(0, sigma^2), the log time of a censored row
drawn from its truncated conditional before each sweep.

The times and the event indicator are data, not parameters. The fit
entry points of this package take a plain response, so a fit under this
family is rejected until one taking a censored time is added.

## Experimental

This family is compiled only into a core built with its `experimental`
Cargo feature. The constructor exists in every build, so a script naming
the family is portable, but a fit or a validated configuration is
rejected with the condition class `thiessen_requires_feature` unless the
package was installed from source with `THIESSEN_EXPERIMENTAL=1` in the
environment;
[`core_experimental()`](https://l-thomson.github.io/thiessen/r/reference/core_experimental.md)
reports the setting of the build in use. An experimental family sits
outside semantic versioning: its configuration and the values it draws
may change in any release. The table of experimental items and their
status is
[`docs/experimental.md`](https://github.com/l-thomson/thiessen/blob/dev/docs/experimental.md).

## References

Wei, L. J. (1992). The accelerated failure time model: a useful
alternative to the Cox regression model in survival analysis.
*Statistics in Medicine* 11(14-15), 1871-1879.
[doi:10.1002/sim.4780111409](https://doi.org/10.1002/sim.4780111409)

## See also

[`thiessen_control()`](https://l-thomson.github.io/thiessen/r/reference/thiessen_control.md),
[`core_experimental()`](https://l-thomson.github.io/thiessen/r/reference/core_experimental.md)

## Examples

``` r
if (FALSE) { # core_experimental()
aft_outcome()
}
```
