import pytest

from buckley24drought.util import inv_norm


def test_inv_norm():
    with pytest.raises(ValueError):
        inv_norm(2.0)
