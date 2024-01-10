import numpy as np
import pandas as pd
import pytest

from buckley24drought.spei import loglogistic_cdf, SPEI


def test_loglogistic_invalid_value_gamma():
    with pytest.raises(ValueError):
        loglogistic_cdf(0, 1, 1, 1)


def test_loglogistic_invalid_alpha():
    with pytest.raises(ValueError):
        loglogistic_cdf(1, 0, 1, 0)


def test_loglogistic_invalid_beta():
    with pytest.raises(ValueError):
        loglogistic_cdf(1, 1, 0, 0)


def test_SPEI_create():
    spei = SPEI()

    # check data type
    assert isinstance(spei.data, pd.DataFrame)

    # check original columns
    assert "area" in spei.data.columns
    assert "end_date" in spei.data.columns
    assert "value" in spei.data.columns
    assert spei.data.dtypes["area"].kind == "O"
    assert spei.data.dtypes["end_date"].kind == "M"
    assert spei.data.dtypes["value"].kind == "f"

    # check created column
    assert "month" in spei.data.columns
    assert spei.data.dtypes["month"].kind == "i"


@pytest.mark.parametrize("window", [0, 49])
def test_SPEI_generate_invalid_window(window):
    with pytest.raises(ValueError):
        spei = SPEI()
        spei.generate_series(window)


@pytest.mark.parametrize("window", range(1, 49))
def test_SPEI_generate_series(window):
    spei = SPEI()
    res = spei.generate_series(window)

    # check column names
    assert np.all(res.columns == ["area", "end_date", "month", "SPEI"])

    # check column types
    assert res.dtypes["area"].kind == "O"
    assert res.dtypes["end_date"].kind == "M"
    assert res.dtypes["month"].kind == "i"
    assert res.dtypes["SPEI"].kind == "f"


@pytest.mark.parametrize("window", [0, 49])
def test_SPEI_check_invalid_window(window):
    with pytest.raises(ValueError):
        spei = SPEI()
        spei.check_fit(window)


@pytest.mark.parametrize("window", range(1, 49))
def test_SPEI_check_fit(window):
    spei = SPEI()
    fit = spei.check_fit(window)
    # goodness-of-fit test results should all be non-significant
    # (i.e., SPEI distribution matches the standard normal distribution)
    assert np.all(fit > 0.05)
