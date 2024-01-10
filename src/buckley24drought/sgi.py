"""SGI module"""

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize, Bounds
from scipy.stats import describe, iqr, norm, kstest
from sklearn.model_selection import LeaveOneOut

from .data.gw import swl
from .util import inv_norm


class GaussianKDE1D:
    """Gaussian kernel density estimator (1D)

    Attributes
    ----------
    data : numpy.ndarray
    bandwidth : float
    """

    def __init__(self, data: np.ndarray):
        self.data = data
        self.bandwidth: float = None

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
        (see Scott & Sain, 2005, pg. 236)

        Parameters
        ----------
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
        """Find optimal bandwidth for the data

        Returns
        -------
        float
        """
        # start with suboptimal bandwidth
        b0 = [
            0.9
            * np.minimum(np.sqrt(describe(self.data).variance), iqr(self.data) / 1.34)
            * np.power(len(self.data), -1 / 5)
        ]
        # bandwidth constrained to be positive
        bounds = Bounds([1e-8], [np.inf])
        # find optimal bandwidth by minimizing error
        return minimize(fun=self.ucv, x0=b0, method="trust-constr", bounds=bounds).x[0]


class SGI:
    """SGI class"""

    def __init__(self):
        pass

    def check_fit(self) -> pd.DataFrame:
        """Check the goodness-of-fit of the SGI.

        Returns
        -------
        data_meta : pandas.DataFrame
            A pivot table of p-values from the Kolmogorov-Smirnov test.
        """
        pass

    def generate_series(self) -> pd.DataFrame:
        """Generate the SGI.

        Returns
        -------
        data_gen : pandas.DataFrame
        """
        pass
