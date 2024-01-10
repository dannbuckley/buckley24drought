import pytest

from buckley24drought.util import inv_norm


@pytest.mark.parametrize("prob", [0.0, 2.0])
def test_inv_norm_invalid_prob(prob):
    with pytest.raises(ValueError):
        inv_norm(prob)
