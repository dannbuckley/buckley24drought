import numpy as np
import pandas as pd
import pytest

from buckley24drought.sgi import SGI


@pytest.mark.parametrize("gwicid", SGI.all_wells)
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
