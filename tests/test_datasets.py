from inferlab.datasets import load_emergencyroom, load_iq, load_reactiontime, load_rutherford

import numpy as np

from pytest import approx
import pytest

def test_load_emergencyroom():
    data = load_emergencyroom()
    assert np.mean(data) == approx(32.9)
    assert np.median(data) == approx(38.5)
    assert np.size(data) == 10

def test_load_rutherford():
    raw_data = load_rutherford("raw")
    table_data = load_rutherford("table")
    assert np.size(raw_data) == 2608
    assert np.mean(raw_data) == approx(3.87, rel=1e-3)
    assert table_data[0] == 57
    assert table_data[5] == 408
    assert table_data[11] == 6
    assert sum(table_data.values()) == 2608
    with pytest.raises(ValueError):
        load_rutherford("abc")

def test_load_iq():
    iq = load_iq()
    assert iq.mu == 100
    assert iq.sigma == 15

def test_load_reactiontime():
    group1, group2 = load_reactiontime()
    assert group1.group_size == 32
    assert group1.mu == 585.2
    assert group1.sigma == 89.6
    assert group2.group_size == 32
    assert group2.mu == 533.7
    assert group2.sigma == 65.3


