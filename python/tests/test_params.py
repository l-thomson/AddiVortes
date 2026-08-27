"""The families and parameter groups."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from thiessen import (
    CellParams,
    GeometryParams,
    Model,
    StructureParams,
    TermParams,
    dart_inclusion,
    gaussian,
    probit,
    soft_membership,
    weighted_inclusion,
)
from thiessen.families import Gaussian, Probit
from thiessen.params import DartInclusion, SoftMembership, WeightedInclusion


def test_the_constructors_return_their_family_objects():
    assert isinstance(gaussian(), Gaussian)
    assert isinstance(probit(), Probit)


def test_the_family_defaults_are_the_core_defaults():
    family = gaussian()
    assert family.nu == 6.0
    assert family.q == 0.85
    assert probit().offset is None


def test_the_family_serialises_as_the_outcome_group():
    assert gaussian()._core() == {"gaussian": {"nu": 6.0, "q": 0.85}}
    assert probit()._core() == {"probit": {}}
    assert probit(offset=0.5)._core() == {"probit": {"offset": 0.5}}


def test_repr_reads_as_the_constructor():
    assert repr(gaussian()) == "gaussian()"
    assert repr(gaussian(nu=3.0)) == "gaussian(nu=3.0)"
    assert repr(probit(offset=0.5)) == "probit(offset=0.5)"
    assert repr(TermParams(tessellations=40)) == "TermParams(tessellations=40)"
    nested = TermParams(geometry=GeometryParams(sigma_c=0.5))
    assert repr(nested) == "TermParams(geometry=GeometryParams(sigma_c=0.5))"


def test_equality_compares_the_parameters():
    assert gaussian() == gaussian()
    assert gaussian(nu=3.0) != gaussian()
    assert gaussian() != probit()
    assert TermParams(tessellations=40) == TermParams(tessellations=40)
    assert TermParams(geometry=GeometryParams()) == TermParams(
        geometry=GeometryParams()
    )


def test_equality_compares_an_array_parameter_element_wise():
    assert GeometryParams(precision=np.eye(2)) == GeometryParams(precision=np.eye(2))
    assert GeometryParams(precision=np.eye(2)) != GeometryParams(
        precision=2.0 * np.eye(2)
    )
    assert GeometryParams(precision=np.eye(2)) != GeometryParams(precision=np.eye(3))
    assert weighted_inclusion([1.0, 2.0]) == weighted_inclusion([1.0, 2.0])
    assert weighted_inclusion([1.0, 2.0]) != weighted_inclusion([2.0, 1.0])


def test_get_params_routes_nested_objects():
    group = TermParams(tessellations=40, geometry=GeometryParams(sigma_c=0.5))
    deep = group.get_params(deep=True)
    assert deep["tessellations"] == 40
    assert deep["geometry__sigma_c"] == 0.5

    shallow = group.get_params(deep=False)
    assert "geometry__sigma_c" not in shallow


def test_set_params_routes_nested_objects():
    group = TermParams(geometry=GeometryParams())
    group.set_params(tessellations=40, geometry__sigma_c=0.5)
    assert group.tessellations == 40
    assert group.geometry is not None
    assert group.geometry.sigma_c == 0.5


def test_set_params_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="Invalid parameter 'zeta'"):
        StructureParams().set_params(zeta=1.0)


def test_a_copy_from_get_params_is_the_object():
    group = TermParams(tessellations=40, k=2.0, geometry=GeometryParams(sigma_c=0.5))
    copy = TermParams(**group.get_params(deep=False))
    assert copy == group


def test_the_objects_pickle():
    group = TermParams(tessellations=40, geometry=GeometryParams(sigma_c=0.5))
    assert pickle.loads(pickle.dumps(group)) == group
    assert pickle.loads(pickle.dumps(gaussian(nu=3.0))) == gaussian(nu=3.0)


def test_term_params_serialises_only_the_set_groups():
    assert TermParams()._core() == {"k": 3.0, "lambda_c": 5.0}
    assert TermParams(structure=StructureParams())._core() == {
        "k": 3.0,
        "lambda_c": 5.0,
    }
    with_omega = TermParams(structure=StructureParams(omega=2.0))
    assert with_omega._core()["structure"] == {"omega": 2.0}


def test_a_nested_group_of_the_wrong_type_is_rejected():
    with pytest.raises(TypeError, match="geometry"):
        bad = TermParams(geometry={"sigma_c": 0.5})  # type: ignore[arg-type]
        Model(mean_params=bad).validate()
    with pytest.raises(TypeError, match="structure"):
        bad = TermParams(structure={"omega": 2.0})  # type: ignore[arg-type]
        Model(mean_params=bad).validate()


def test_model_routes_like_the_groups():
    model = Model(mean_params=TermParams(tessellations=8))
    model.set_params(mean_params__tessellations=16, draws=50)
    assert model.mean_params is not None
    assert model.mean_params.tessellations == 16
    assert model.draws == 50


def test_the_option_constructors_return_their_objects():
    assert isinstance(soft_membership(), SoftMembership)
    assert isinstance(weighted_inclusion([1.0]), WeightedInclusion)
    assert isinstance(dart_inclusion(), DartInclusion)


def test_the_options_serialise_as_the_tagged_form():
    assert soft_membership()._core() == {"soft": {"rate": 10.0}}
    assert weighted_inclusion([2.0, 1.0])._core() == {
        "weighted": {"weights": [2.0, 1.0]}
    }
    assert dart_inclusion()._core() == {"dart": {"a": 0.5, "b": 1.0}}
    assert dart_inclusion(rho=3.0)._core() == {"dart": {"a": 0.5, "b": 1.0, "rho": 3.0}}


def test_the_option_repr_reads_as_the_constructor():
    assert repr(soft_membership()) == "soft_membership()"
    assert repr(dart_inclusion(rho=3.0)) == "dart_inclusion(rho=3.0)"
    assert (
        repr(weighted_inclusion([2.0, 1.0])) == "weighted_inclusion(weights=[2.0, 1.0])"
    )
    assert repr(CellParams(basis="linear")) == "CellParams(basis='linear')"


def test_the_groups_carry_the_options():
    geometry = GeometryParams(
        metric=["manhattan", "cosine"], membership=soft_membership()
    )
    assert geometry._core() == {
        "metric": [{"manhattan": {}}, {"cosine": {}}],
        "sigma_c": 0.8,
        "membership": {"soft": {"rate": 10.0}},
    }
    structure = StructureParams(inclusion=dart_inclusion())
    assert structure._core() == {"inclusion": {"dart": {"a": 0.5, "b": 1.0}}}
    assert StructureParams(inclusion="uniform")._core() == {"inclusion": "uniform"}
    term = TermParams(cell=CellParams(basis="linear"))
    assert term._core()["cell"] == {"basis": "linear"}
    assert "cell" not in TermParams(cell=CellParams())._core()


def test_the_precision_matrix_crosses_row_major():
    geometry = GeometryParams(
        metric=["mahalanobis"] * 2, precision=[[1.0, 0.5], [0.5, 2.0]]
    )

    assert list(geometry._core()["precision"]) == [1.0, 0.5, 0.5, 2.0]


def test_an_option_of_the_wrong_kind_is_rejected():
    with pytest.raises(TypeError, match="membership"):
        GeometryParams(membership=dart_inclusion())._core()  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="inclusion"):
        StructureParams(inclusion=soft_membership())._core()  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="cell"):
        TermParams(cell=GeometryParams())._core()  # type: ignore[arg-type]


def test_the_options_route_and_pickle():
    term = TermParams(
        geometry=GeometryParams(membership=soft_membership()),
        cell=CellParams(basis="linear"),
    )
    term.set_params(geometry__membership__rate=5.0, cell__basis="constant")

    routed = term.get_params()
    assert routed["geometry__membership__rate"] == 5.0
    assert routed["cell__basis"] == "constant"
    assert pickle.loads(pickle.dumps(term)) == term
