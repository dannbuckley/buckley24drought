"""Module for utility functions"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize, Bounds
from scipy.stats import describe, iqr, norm
from sklearn.model_selection import LeaveOneOut


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


class GaussianKDE1D:
    """Gaussian kernel density estimator (1D)

    Attributes
    ----------
    data : numpy.ndarray
    bandwidth : float
    """

    def __init__(self, data: np.ndarray):
        self.data = data
        self.bandwidth: float = self.find_bandwidth()

    def pdf(self, value: float, bandwidth: float) -> float:
        """Probability density function (PDF)

        Parameters
        ----------
        value : float
        bandwidth : float
            Must be positive.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If `bandwidth` is not positive.
        """
        if bandwidth <= 0:
            raise ValueError("Bandwidth must be positive.")
        return np.sum(
            [norm.pdf(value, loc=x_i, scale=bandwidth) for x_i in self.data]
        ) / len(self.data)

    def cdf(self, value: float, bandwidth: float) -> float:
        """Cumulative distribution function (CDF)

        Returns non-exceedance probabilities `P(X <= x)`.

        Parameters
        ----------
        value : float
        bandwidth : float
            Must be positive.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If `bandwidth` is not positive.
        """
        if bandwidth <= 0:
            raise ValueError("Bandwidth must be positive.")
        return np.sum(
            [norm.cdf(value, loc=x_i, scale=bandwidth) for x_i in self.data]
        ) / len(self.data)

    def opt_pdf(self, value: float) -> float:
        """Probability density function (PDF) using optimized bandwidth

        Parameters
        ----------
        value : float

        Returns
        -------
        float

        See Also
        --------
        GaussianKDE1D.pdf
        """
        return self.pdf(value, bandwidth=self.bandwidth)

    def opt_cdf(self, value: float) -> float:
        """Cumulative distribution function (CDF) using optimized bandwidth

        Parameters
        ----------
        value : float

        Returns
        -------
        float

        See Also
        --------
        GaussianKDE1D.cdf
        """
        return self.cdf(value, bandwidth=self.bandwidth)

    def ucv(self, bandwidth: float) -> float:
        """Unbiased cross-validation criterion for estimated bandwidth
        (see Scott & Sain, 2005, pg. 236).

        Parameters
        ----------
        bandwidth : float
            Must be positive.

        Returns
        -------
        float
            Error associated with given bandwidth value

        Raises
        ------
        ValueError
            If `bandwidth` is not positive.
        """
        if bandwidth <= 0:
            raise ValueError("Bandwidth must be positive.")

        # compute integrated squared error component
        er_sq = quad(
            func=lambda x: self.pdf(x, bandwidth=bandwidth) ** 2, a=-np.inf, b=np.inf
        )[0]

        # compute leave-one-out error component
        loo = LeaveOneOut()
        er_lo = (2 / len(self.data)) * np.sum(
            [
                np.sum(
                    [
                        norm.pdf(self.data[test], loc=x_i, scale=bandwidth)
                        for x_i in self.data[train]
                    ]
                )
                / len(train)
                for train, test in loo.split(self.data)
            ]
        )
        return er_sq - er_lo

    def find_bandwidth(self):
        """Find optimal bandwidth for the data.

        Returns
        -------
        float
            Bandwidth that minimizes the error of the kernel density estimate
        """
        # start with suboptimal bandwidth
        bnd0 = [
            0.9
            * np.minimum(np.sqrt(describe(self.data).variance), iqr(self.data) / 1.34)
            * np.power(len(self.data), -1 / 5)
        ]
        # bandwidth constrained to be positive
        bounds = Bounds([1e-8], [np.inf])
        # find optimal bandwidth by minimizing error
        return minimize(fun=self.ucv, x0=bnd0, method="trust-constr", bounds=bounds).x[
            0
        ]
