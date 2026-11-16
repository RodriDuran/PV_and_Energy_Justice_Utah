"""Tests of the tract master table helpers."""

import pandas as pd

from utah_ejpi.geography import resstock_gisjoin_to_geoid


def test_gisjoin_to_geoid():
    s = pd.Series(["G4900010100100", "G0100010020100"])
    assert resstock_gisjoin_to_geoid(s).tolist() == ["49001100100", "01001020100"]
