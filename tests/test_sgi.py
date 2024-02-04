import numpy as np
import pandas as pd
import pytest

from buckley24drought.sgi import SGI


all_wells = [
    32,
    820,
    824,
    5418,
    9771,
    9858,
    50808,
    55463,
    56528,
    57128,
    57525,
    58096,
    60137,
    91230,
    91244,
    96132,
    96826,
    99215,
    123132,
    126793,
    129491,
    129952,
    130860,
    132260,
    133162,
    133165,
    133167,
    133172,
    133174,
    133176,
    135680,
    135689,
    135720,
    135722,
    135734,
    135735,
    136050,
    136486,
    136964,
    136969,
    136970,
    139989,
    140366,
    148531,
]


@pytest.mark.parametrize("gwicid", all_wells)
def test_SGI_create(gwicid):
    sgi = SGI(gwicid=gwicid)

    # check data type
    assert isinstance(sgi.data, pd.DataFrame)

    # check column names
    assert np.all(sgi.data.columns == ["gwicid", "date", "month", "monthly_average"])

    # check column types
    assert sgi.data.dtypes["gwicid"].kind == "i"
    assert sgi.data.dtypes["date"].kind == "M"
    assert sgi.data.dtypes["month"].kind == "i"
    assert sgi.data.dtypes["monthly_average"].kind == "f"


@pytest.mark.intg
def test_SGI_generate_series():
    sgi = SGI(gwicid=32)
    res = sgi.generate_series()

    # check column names
    assert np.all(res.columns == ["gwicid", "date", "month", "SGI"])

    # check column types
    assert res.dtypes["gwicid"].kind == "i"
    assert res.dtypes["date"].kind == "M"
    assert res.dtypes["month"].kind == "i"
    assert res.dtypes["SGI"].kind == "f"


@pytest.mark.intg
def test_SGI_check_fit():
    sgi = SGI(gwicid=32)
    fit = sgi.check_fit()
    # goodness-of-fit test results should all be non-significant
    # (i.e., SGI distribution matches the standard normal distribution)
    assert np.all(fit > 0.05)
