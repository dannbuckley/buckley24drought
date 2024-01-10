import numpy as np
import pandas as pd
import pytest

from buckley24drought.spi import SPI


def test_SPI_create():
    spi = SPI()

    # check data type
    assert isinstance(spi.data, pd.DataFrame)

    # check original columns
    assert "area" in spi.data.columns
    assert "end_date" in spi.data.columns
    assert "value" in spi.data.columns
    assert spi.data.dtypes["area"].kind == "O"
    assert spi.data.dtypes["end_date"].kind == "M"
    assert spi.data.dtypes["value"].kind == "f"

    # check created column
    assert "month" in spi.data.columns
    assert spi.data.dtypes["month"].kind == "i"


@pytest.mark.parametrize("window", [0, 49])
def test_SPI_generate_invalid_window(window):
    with pytest.raises(ValueError):
        spi = SPI()
        spi.generate_series(window)


@pytest.mark.parametrize("window", range(1, 49))
def test_SPI_generate_series(window):
    spi = SPI()
    res = spi.generate_series(window)

    # check column names
    assert np.all(res.columns == ["area", "end_date", "month", "SPI"])

    # check column types
    assert res.dtypes["area"].kind == "O"
    assert res.dtypes["end_date"].kind == "M"
    assert res.dtypes["month"].kind == "i"
    assert res.dtypes["SPI"].kind == "f"


@pytest.mark.parametrize("window", [0, 49])
def test_SPI_check_invalid_window(window):
    with pytest.raises(ValueError):
        spi = SPI()
        spi.check_fit(window)


@pytest.mark.parametrize("window", range(1, 49))
def test_SPI_check_fit(window):
    spi = SPI()
    fit = spi.check_fit(window)
    # goodness-of-fit test results should all be non-significant
    # (i.e., SPI distribution matches the standard normal distribution)
    assert np.all(fit > 0.05)
