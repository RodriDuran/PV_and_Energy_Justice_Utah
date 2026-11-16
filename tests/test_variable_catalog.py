"""Consistency checks of the variable catalog."""

from importlib import resources

import pandas as pd
import pytest

REQUIRED = [
    "variable",
    "description",
    "source",
    "scale",
    "period",
    "block",
    "justice_dimension",
    "universe",
    "verification",
    "evidence_url",
    "notes",
]


@pytest.fixture(scope="module")
def catalog():
    path = resources.files("utah_ejpi") / "data" / "variable_catalog.csv"
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def test_required_columns(catalog):
    assert set(REQUIRED) <= set(catalog.columns)


def test_verification_states(catalog):
    assert set(catalog.verification) <= {"verified", "to_confirm", "not_verified"}


def test_verified_entries_have_evidence_url(catalog):
    verified = catalog[catalog.verification == "verified"]
    assert (verified.evidence_url != "").all()


def test_single_justice_dimension_per_variable(catalog):
    assert not catalog.variable.duplicated().any()
    assert set(catalog.justice_dimension) <= {
        "none",
        "distributive",
        "recognition",
        "procedural",
    }
