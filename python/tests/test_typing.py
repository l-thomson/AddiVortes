"""The stub of the compiled extension against the extension itself.

The stub is hand-written, so nothing but a test keeps it in step with the
module. Every name the extension exposes must appear in the stub and every
name the stub declares must exist, including the methods of `Fitted`.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Iterable
from pathlib import Path

import thiessen
from thiessen import _native

STUB = Path(_native.__file__).with_name("_native.pyi")


def _stub_tree() -> ast.Module:
    return ast.parse(STUB.read_text())


def _module_level(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _class_members(tree: ast.Module, name: str) -> set[str]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return {
                member.name
                for member in node.body
                if isinstance(member, ast.FunctionDef)
            }
    raise AssertionError(f"the stub declares no class {name}")


def _public(names: Iterable[str]) -> set[str]:
    return {name for name in names if not name.startswith("_")}


def test_the_stub_exists():
    assert STUB.is_file()


def test_every_extension_name_is_in_the_stub():
    declared = _module_level(_stub_tree())
    assert _public(dir(_native)) <= declared


def test_every_stub_name_exists_in_the_extension():
    for name in _module_level(_stub_tree()):
        assert hasattr(_native, name), name


def test_the_fitted_methods_match():
    declared = _class_members(_stub_tree(), "Fitted")
    actual = {name for name in dir(_native.Fitted) if not name.startswith("__")}
    # Getters are properties on the class and `def` in the stub.
    assert actual <= declared
    for name in declared:
        assert hasattr(_native.Fitted, name), name


def test_every_public_package_name_is_annotated():
    for name in thiessen.__all__:
        member = getattr(thiessen, name)
        if not callable(member) or isinstance(member, type):
            continue
        assert inspect.signature(member) is not None


def test_the_package_declares_all():
    from thiessen import (
        _arrays,
        _config,
        _encoding,
        _params,
        _seed,
        estimators,
        families,
        model,
        params,
    )

    modules = (
        _arrays,
        _config,
        _encoding,
        _params,
        _seed,
        estimators,
        families,
        model,
        params,
        thiessen,
    )
    for module in modules:
        assert module.__all__, module.__name__


def test_every_public_callable_has_a_docstring():
    from thiessen import estimators, families, model, params

    for module in (model, estimators, families, params):
        for name in module.__all__:
            member = getattr(module, name)
            assert member.__doc__, f"{module.__name__}.{name}"
            for attribute in vars(member):
                if attribute.startswith("_"):
                    continue
                value = getattr(member, attribute)
                if callable(value) or isinstance(value, property):
                    documented = (
                        value.__doc__
                        if not isinstance(value, property)
                        else value.fget.__doc__
                    )
                    assert documented, f"{module.__name__}.{name}.{attribute}"
