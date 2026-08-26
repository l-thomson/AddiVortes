"""The outcome families.

An outcome family is made by its constructor function, the idiom of
``glm(family = binomial())`` in R and of ``family=`` in statsmodels:
`gaussian` for the Gaussian and heteroscedastic models and `probit` for
binary classification. The returned object carries its parameters and
serialises as the configuration's ``outcome`` group, so the constructor
arguments and the stored form share one set of names.

`gaussian` and `probit` are the published models of Stone and Gosling
(2025). The families below them are experimental: the core compiles them
only with its ``experimental`` Cargo feature, so their constructors exist
in every build but a configuration naming one is rejected with
`RequiresFeatureError` unless the extension was built with
``--features experimental``. An experimental family sits outside semantic
versioning, its configuration and the values it draws may change in any
release, and ``docs/experimental.md`` in the repository is the table of
gated items.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._params import Outcome, _dense

__all__ = [
    "Aft",
    "Gaussian",
    "IntervalCensored",
    "Laplace",
    "Ordinal",
    "Outcome",
    "Probit",
    "StudentT",
    "Tobit",
    "aft",
    "gaussian",
    "interval_censored",
    "laplace",
    "ordinal",
    "probit",
    "student_t",
    "tobit",
]


class Gaussian(Outcome):
    """The Gaussian outcome, returned by `gaussian`.

    Parameters
    ----------
    nu : float, default=6.0
        sigma^2 prior degrees of freedom nu. A variance ensemble requires
        nu > 2.
    q : float, default=0.85
        sigma^2 prior calibration quantile q, Pr(sigma < sigma_hat) = q.
    """

    def __init__(self, nu: float = 6.0, q: float = 0.85) -> None:
        self.nu = nu
        self.q = q

    def _display_name(self) -> str:
        return "gaussian"

    def _core(self) -> dict[str, Any]:
        return {"gaussian": _dense({"nu": self.nu, "q": self.q})}


class Probit(Outcome):
    """The binary probit outcome, returned by `probit`.

    Parameters
    ----------
    offset : float, optional
        The offset c in P(y = 1 | x) = Phi(c + f(x)). `None` resolves to
        Phi^-1(ybar) at fit, the BART ``binaryOffset`` default.
    """

    def __init__(self, offset: float | None = None) -> None:
        self.offset = offset

    def _display_name(self) -> str:
        return "probit"

    def _core(self) -> dict[str, Any]:
        return {"probit": _dense({"offset": self.offset})}


class Tobit(Outcome):
    """The tobit outcome, returned by `tobit`. Experimental.

    Parameters
    ----------
    lower : float, optional
        The lower censoring limit. `None` is no lower limit.
    upper : float, optional
        The upper censoring limit. `None` is no upper limit.
    nu : float, default=6.0
        sigma^2 prior degrees of freedom nu.
    q : float, default=0.85
        sigma^2 prior calibration quantile q.
    """

    def __init__(
        self,
        lower: float | None = None,
        upper: float | None = None,
        nu: float = 6.0,
        q: float = 0.85,
    ) -> None:
        self.lower = lower
        self.upper = upper
        self.nu = nu
        self.q = q

    def _display_name(self) -> str:
        return "tobit"

    def _core(self) -> dict[str, Any]:
        return {
            "tobit": _dense(
                {
                    "lower": self.lower,
                    "upper": self.upper,
                    "nu": self.nu,
                    "q": self.q,
                }
            )
        }


class Aft(Outcome):
    """The accelerated failure time outcome, returned by `aft`. Experimental.

    Parameters
    ----------
    nu : float, default=6.0
        sigma^2 prior degrees of freedom nu on the log-time scale.
    q : float, default=0.85
        sigma^2 prior calibration quantile q.
    """

    def __init__(self, nu: float = 6.0, q: float = 0.85) -> None:
        self.nu = nu
        self.q = q

    def _display_name(self) -> str:
        return "aft"

    def _core(self) -> dict[str, Any]:
        return {"aft": _dense({"nu": self.nu, "q": self.q})}


class IntervalCensored(Outcome):
    """The interval-censored outcome, returned by `interval_censored`.

    Experimental.

    Parameters
    ----------
    nu : float, default=6.0
        sigma^2 prior degrees of freedom nu.
    q : float, default=0.85
        sigma^2 prior calibration quantile q.
    """

    def __init__(self, nu: float = 6.0, q: float = 0.85) -> None:
        self.nu = nu
        self.q = q

    def _display_name(self) -> str:
        return "interval_censored"

    def _core(self) -> dict[str, Any]:
        return {"interval_censored": _dense({"nu": self.nu, "q": self.q})}


class Ordinal(Outcome):
    """The ordinal outcome, returned by `ordinal`. Experimental.

    Parameters
    ----------
    categories : int, default=2
        Number of ordered categories K, at least 2; the response holds
        integer codes 0 to K - 1.
    offset : float, optional
        The offset c. `None` resolves at fit to Phi^-1 of the share of
        rows above the first category.
    cutpoint_sd : float, default=1.0
        Standard deviation of the prior on the log-gaps between interior
        cutpoints.
    """

    def __init__(
        self,
        categories: int = 2,
        offset: float | None = None,
        cutpoint_sd: float = 1.0,
    ) -> None:
        self.categories = categories
        self.offset = offset
        self.cutpoint_sd = cutpoint_sd

    def _display_name(self) -> str:
        return "ordinal"

    def _core(self) -> dict[str, Any]:
        return {
            "ordinal": _dense(
                {
                    "categories": self.categories,
                    "offset": self.offset,
                    "cutpoint_sd": self.cutpoint_sd,
                }
            )
        }


class StudentT(Outcome):
    """The Student-t outcome, returned by `student_t`. Experimental.

    Parameters
    ----------
    df : float or sequence of float, default=4.0
        The error degrees of freedom: one value, or a grid of at least
        two strictly increasing values drawn over.
    nu : float, default=6.0
        sigma^2 prior degrees of freedom nu.
    q : float, default=0.85
        sigma^2 prior calibration quantile q.
    """

    def __init__(
        self,
        df: float | Sequence[float] = 4.0,
        nu: float = 6.0,
        q: float = 0.85,
    ) -> None:
        self.df = df
        self.nu = nu
        self.q = q

    def _display_name(self) -> str:
        return "student_t"

    def _core(self) -> dict[str, Any]:
        return {"student_t": _dense({"df": self.df, "nu": self.nu, "q": self.q})}


class Laplace(Outcome):
    """The Laplace outcome, returned by `laplace`. Experimental.

    Parameters
    ----------
    nu : float, default=6.0
        sigma^2 prior degrees of freedom nu.
    q : float, default=0.85
        sigma^2 prior calibration quantile q.
    """

    def __init__(self, nu: float = 6.0, q: float = 0.85) -> None:
        self.nu = nu
        self.q = q

    def _display_name(self) -> str:
        return "laplace"

    def _core(self) -> dict[str, Any]:
        return {"laplace": _dense({"nu": self.nu, "q": self.q})}


def gaussian(nu: float = 6.0, q: float = 0.85) -> Gaussian:
    """Return the Gaussian outcome family.

    One sigma^2 is drawn per sweep. Attaching a variance ensemble
    (``variance_params`` with a positive tessellation count) makes the
    model heteroscedastic, so the residual variance varies with x.

    Parameters
    ----------
    nu : float, default=6.0
        sigma^2 prior degrees of freedom nu. A variance ensemble requires
        nu > 2.
    q : float, default=0.85
        sigma^2 prior calibration quantile q, Pr(sigma < sigma_hat) = q.

    Returns
    -------
    Gaussian
        The family object.

    Examples
    --------
    >>> from thiessen import gaussian
    >>> gaussian(nu=3.0)
    gaussian(nu=3.0)
    """
    return Gaussian(nu=nu, q=q)


def probit(offset: float | None = None) -> Probit:
    """Return the binary probit outcome family.

    P(y = 1 | x) = Phi(c + f(x)) with Albert and Chib (1993) augmentation.
    The latent scale is fixed at 1 for identification, so a variance
    ensemble is not available under this family.

    Parameters
    ----------
    offset : float, optional
        The offset c. `None` resolves to Phi^-1(ybar) at fit, the BART
        ``binaryOffset`` default.

    Returns
    -------
    Probit
        The family object.

    Examples
    --------
    >>> from thiessen import probit
    >>> probit()
    probit()
    """
    return Probit(offset=offset)


def tobit(
    lower: float | None = None,
    upper: float | None = None,
    nu: float = 6.0,
    q: float = 0.85,
) -> Tobit:
    """Return the tobit outcome family. Experimental.

    The type-I tobit model (Tobin 1958) for a response censored at known
    limits: a response value equal to a limit is read as censored on that
    side, and the latent value behind it is drawn by the augmentation of
    Chib (1992). At least one limit is required, and a response value
    beyond a limit is rejected at fit.

    Parameters
    ----------
    lower : float, optional
        The lower censoring limit. `None` is no lower limit.
    upper : float, optional
        The upper censoring limit. `None` is no upper limit.
    nu : float, default=6.0
        sigma^2 prior degrees of freedom nu.
    q : float, default=0.85
        sigma^2 prior calibration quantile q.

    Returns
    -------
    Tobit
        The family object.

    Examples
    --------
    >>> from thiessen import tobit
    >>> tobit(lower=0.0)
    tobit(lower=0.0)
    """
    return Tobit(lower=lower, upper=upper, nu=nu, q=q)


def aft(nu: float = 6.0, q: float = 0.85) -> Aft:
    """Return the accelerated failure time outcome family. Experimental.

    The lognormal accelerated failure time model (Wei 1992) for a
    right-censored time to event, the model of the BART package's
    ``abart``. The times and the event indicator are data, not
    parameters, and the fit entry points of this package take a plain
    response, so a fit under this family is rejected until one taking a
    censored time is added.

    Parameters
    ----------
    nu : float, default=6.0
        sigma^2 prior degrees of freedom nu on the log-time scale.
    q : float, default=0.85
        sigma^2 prior calibration quantile q.

    Returns
    -------
    Aft
        The family object.

    Examples
    --------
    >>> from thiessen import aft
    >>> aft()
    aft()
    """
    return Aft(nu=nu, q=q)


def interval_censored(nu: float = 6.0, q: float = 0.85) -> IntervalCensored:
    """Return the interval-censored outcome family. Experimental.

    The interval-censoring observation scheme (Sun 2006) for a response
    known only to lie between two row-specific bounds. The bounds are
    data, not parameters, and the fit entry points of this package take a
    plain response, so a fit under this family is rejected until one
    taking a pair of bounds is added.

    Parameters
    ----------
    nu : float, default=6.0
        sigma^2 prior degrees of freedom nu.
    q : float, default=0.85
        sigma^2 prior calibration quantile q.

    Returns
    -------
    IntervalCensored
        The family object.

    Examples
    --------
    >>> from thiessen import interval_censored
    >>> interval_censored()
    interval_censored()
    """
    return IntervalCensored(nu=nu, q=q)


def ordinal(
    categories: int = 2,
    offset: float | None = None,
    cutpoint_sd: float = 1.0,
) -> Ordinal:
    """Return the ordinal outcome family. Experimental.

    The ordered probit model of Albert and Chib (1993),
    P(y <= k | x) = Phi(gamma_(k+1) - c - f(x)), for a response holding
    integer codes 0 to K - 1. The latent variance is fixed at 1 and the
    first cutpoint at 0 for identification, and the interior cutpoints
    are drawn on the log-gap scale of Albert and Chib (2001). At K = 2
    the model is `probit`.

    Parameters
    ----------
    categories : int, default=2
        Number of ordered categories K, at least 2.
    offset : float, optional
        The offset c. `None` resolves at fit to Phi^-1 of the share of
        rows above the first category.
    cutpoint_sd : float, default=1.0
        Standard deviation of the prior on the log-gaps between interior
        cutpoints.

    Returns
    -------
    Ordinal
        The family object.

    Examples
    --------
    >>> from thiessen import ordinal
    >>> ordinal(categories=4)
    ordinal(categories=4)
    """
    return Ordinal(categories=categories, offset=offset, cutpoint_sd=cutpoint_sd)


def student_t(
    df: float | Sequence[float] = 4.0,
    nu: float = 6.0,
    q: float = 0.85,
) -> StudentT:
    """Return the Student-t outcome family. Experimental.

    The independent Student-t model of Geweke (1993) for a continuous
    response with outliers, drawn through its scale-mixture
    representation. The degrees of freedom are fixed at a value, or drawn
    each sweep over a grid carrying a uniform prior; no continuous
    sampler over them exists, df being weakly identified.

    Parameters
    ----------
    df : float or sequence of float, default=4.0
        The error degrees of freedom: one value, or a grid of at least
        two strictly increasing values drawn over.
    nu : float, default=6.0
        sigma^2 prior degrees of freedom nu.
    q : float, default=0.85
        sigma^2 prior calibration quantile q.

    Returns
    -------
    StudentT
        The family object.

    Examples
    --------
    >>> from thiessen import student_t
    >>> student_t(df=[3.0, 6.0, 12.0])
    student_t(df=[3.0, 6.0, 12.0])
    """
    return StudentT(df=df, nu=nu, q=q)


def laplace(nu: float = 6.0, q: float = 0.85) -> Laplace:
    """Return the Laplace outcome family. Experimental.

    The Laplace model for a continuous response with outliers, drawn
    through the normal-exponential mixture of Park and Casella (2008).
    The errors have exponential tails, so a wild observation is
    discounted at rate 1/|r| against the Student-t model's 1/r^2.

    Parameters
    ----------
    nu : float, default=6.0
        sigma^2 prior degrees of freedom nu.
    q : float, default=0.85
        sigma^2 prior calibration quantile q.

    Returns
    -------
    Laplace
        The family object.

    Examples
    --------
    >>> from thiessen import laplace
    >>> laplace()
    laplace()
    """
    return Laplace(nu=nu, q=q)
