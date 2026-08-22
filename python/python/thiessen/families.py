"""The outcome families.

An outcome family is made by its constructor function, the idiom of
``glm(family = binomial())`` in R and of ``family=`` in statsmodels:
`gaussian` for the Gaussian and heteroscedastic models and `probit` for
binary classification. The returned object carries its parameters and
serialises as the configuration's ``outcome`` group, so the constructor
arguments and the stored form share one set of names.

The families here are the published models of Stone and Gosling (2025).
The core's experimental catalogue has no constructor in this package.
"""

from __future__ import annotations

from typing import Any

from ._params import Params, _dense

__all__ = ["Gaussian", "Probit", "gaussian", "probit"]


class Gaussian(Params):
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


class Probit(Params):
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
