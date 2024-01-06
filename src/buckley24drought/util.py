"""Module for utility functions"""
import numpy as np


def inv_norm(x: float):
    """Inverse normal CDF approximation

    Parameters
    ----------
    x : float
        CDF value
    
    Returns
    -------
    value : float
        Normal variate
    
    Raises
    ------
    ValueError
        If the input is outside of the interval (0, 1).
    """
    s = None
    u = None

    if 0 < x <= 0.5:
        s = -1
        u = np.power(-2 * np.log(x), 0.5)
    elif 0.5 < x < 1.0:
        s = 1
        u = np.power(-2 * np.log(1.0 - x), 0.5)
    else:
        raise ValueError('Argument must be between 0 and 1 (exclusive).')

    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308
    numer = c0 + (u * (c1 + (u * c2)))
    denom = 1 + (u * (d1 + (u * (d2 + (u * d3)))))
    return s * (u - (numer / denom))
