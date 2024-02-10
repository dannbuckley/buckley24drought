"""SGI module"""

# pylint: disable=R0801
from functools import partial
from importlib.resources import files

import pandas as pd
from scipy.stats import norm, kstest

from .data.gw import swl
from .util import inv_norm, GaussianKDE1D


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

    all_wells = [
        32,
        820,
        5418,
        9771,
        9858,
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
        148531,
    ]

    def __init__(self, gwicid: int):
        if not gwicid in self.all_wells:
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
            """Apply the two-sided Kolomogorov-Smirnov test to the SGI data.

            Parameters
            ----------
            month : int
            sgi_df : pandas.DataFrame

            Returns
            -------
            pvalue : float
            """
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
            This is a fix for pylint W0640.

            Parameters
            ----------
            value : float
            kde : GaussianKDE1D

            Returns
            -------
            float
                Result of inverse normal CDF
            """
            return inv_norm(kde.opt_cdf(value))

        transformed_dict = {"index": [], "SGI": []}
        for month in range(1, 13):
            month_data = self.data.query(f"month == {month}")[
                "monthly_average"
            ].dropna()
            transformed_dict["index"].extend(month_data.index)
            month_kde = GaussianKDE1D(month_data.values)
            transformed_dict["SGI"].extend(
                map(partial(_inv_norm_wrapper, kde=month_kde), month_data.values)
            )
        data_gen = self.data.copy(deep=True).drop(columns="monthly_average")
        data_gen["SGI"] = pd.Series(
            data=transformed_dict["SGI"], index=transformed_dict["index"]
        )
        return data_gen
