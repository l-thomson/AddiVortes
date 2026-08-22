"""The families and parameter groups."""

from __future__ import annotations

import pickle

import pytest
from thiessen import (
    GeometryParams,
    Model,
    StructureParams,
    TermParams,
    gaussian,
    probit,
)
from thiessen.families import Gaussian, Probit


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
