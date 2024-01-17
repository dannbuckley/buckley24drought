"""SGI module"""
# pylint: disable=R0801
from functools import partial
from importlib.resources import files

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


class SGI:
    """SGI class

    Parameters
    ----------
    gwicid : int
        GWIC ID of the well that will be used to compute the SGI.

    Raises
    ------
    ValueError
        If there is no data for the given GWIC ID within this project.

    Attributes
    ----------
    data : pandas.DataFrame
    """

    _all_wells = [
        32,
        820,
        824,
        5418,
        9771,
        9858,
        21567,
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
        99787,
        99837,
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

    def __init__(self, gwicid: int):
        if not gwicid in self._all_wells:
            raise ValueError(
                "There is no data associated with the given GWIC ID (for this project)."
            )

        # get well data from package resources
        rsrc = files(swl) / f"{gwicid}.csv"
        self.data = pd.read_csv(
            rsrc,
            index_col="date",
            usecols=["gwicid", "date", "daily_average"],
            # enforce date type for date column
            parse_dates=["date"],
        )
        # interpret water levels as heads
        self.data["daily_average"] = -self.data["daily_average"]

        # resample to monthly average
        # NOTE: this will introduce NaNs if record is incomplete
        temp_df = (
            self.data["daily_average"]
            .resample("1M")
            .mean()
            .reset_index()
            .rename(columns={"daily_average": "monthly_average"})
        )
        temp_df["gwicid"] = self.data["gwicid"].drop_duplicates().values[0]
        # add integer column for the month number
        temp_df["month"] = temp_df["date"].apply(lambda x: x.month)
        self.data = (
            temp_df.reindex(columns=["gwicid", "date", "month", "monthly_average"])
            # constrain record to study period
            .query("date >= '1991-01-01' and date <= '2020-12-31'").copy(deep=True)
        )

    def check_fit(self) -> pd.DataFrame:
        """Check the goodness-of-fit of the SGI.

        Returns
        -------
        data_meta : pandas.DataFrame
            A pivot table of p-values from the Kolmogorov-Smirnov test.
        """
        # generate SGI series
        sgi_df = self.generate_series()

        def _apply_test(month: int, sgi_df: pd.DataFrame):
            return kstest(
                rvs=sgi_df.query(f"month == {month}")["SGI"].dropna(),
                cdf=norm.cdf,
                alternative="two-sided",
            ).pvalue

        # check goodness-of-fit for each monthly distribution
        data_meta = pd.DataFrame(
            {
                "gwicid": sgi_df["gwicid"].drop_duplicates().values[0],
                "month": range(1, 13),
                "pvalue": map(partial(_apply_test, sgi_df=sgi_df), range(1, 13)),
            }
        )

        # transform metadata into a pivot table
        return data_meta.pivot_table(values="pvalue", index="gwicid", columns="month")

    def generate_series(self) -> pd.DataFrame:
        """Generate the SGI.

        Returns
        -------
        data_gen : pandas.DataFrame
        """

        def _inv_norm_wrapper(value: float, kde: GaussianKDE1D) -> float:
            """Wrapper function to combine inv_norm and GaussianKDE1D.opt_cdf.
            This is a fix for pylint W0640."""
            return inv_norm(kde.opt_cdf(value))

        transformed_dict = {"index": [], "SGI": []}
        for month in range(1, 13):
            month_data = self.data.query(f"month == {month}")[
                "monthly_average"
            ].dropna()
            transformed_dict["index"].extend(month_data.index)
            month_kde = GaussianKDE1D(month_data.values)
            month_sgi = list(
                map(partial(_inv_norm_wrapper, kde=month_kde), month_data.values)
            )
            transformed_dict["SGI"].extend(month_sgi)
        data_gen = self.data.copy(deep=True).drop(columns="monthly_average")
        data_gen["SGI"] = pd.Series(
            data=transformed_dict["SGI"], index=transformed_dict["index"]
        )
        return data_gen
