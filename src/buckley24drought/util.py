"""Module for utility functions"""
import numpy as np


def inv_norm(prob: float) -> float:
    """Inverse normal CDF approximation

    Parameters
    ----------
    prob : float
        Non-exceedance probability from the original distribution

    Returns
    -------
    float
        Standard normal variate

    Raises
    ------
    ValueError
        If `prob` is outside of the interval (0, 1).
    """
    sign: float = None
    p_t: float = None

    # check whether original value was below or above the median
    if 0 < prob <= 0.5:
        # below the median, N(0, 1) value will be negative
        sign = -1.0
        p_t = np.power(-2 * np.log(prob), 0.5)
    elif 0.5 < prob < 1.0:
        # above the median, N(0, 1) value will be positive
        sign = 1.0
        p_t = np.power(-2 * np.log(1.0 - prob), 0.5)
    else:
        raise ValueError("Argument must be between 0 and 1 (exclusive).")

    # numerator equivalent to: 2.515517 + 0.802853(p_t) + 0.010328(p_t)^2
    numer: float = 2.515517 + (p_t * (0.802853 + (p_t * 0.010328)))
    # denominator equivalent to: 1 + 1.432788(p_t) + 0.189269(p_t)^2 + 0.001308(p_t)^3
    denom: float = 1 + (p_t * (1.432788 + (p_t * (0.189269 + (p_t * 0.001308)))))
    return sign * (p_t - (numer / denom))
