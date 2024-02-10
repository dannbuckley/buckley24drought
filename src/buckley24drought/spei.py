"""SPEI module"""

# pylint: disable=R0801
from functools import partial
from importlib.resources import files

import pandas as pd
from scipy.stats import kstest, norm

from .data import gridmet
from .util import inv_norm, GaussianKDE1D


class SPEI:
    """SPEI class

    Attributes
    ----------
    data : pandas.DataFrame
        Climatic water balance data from which to generate the SPEI.
    """

    def __init__(self):
        # get climatic water balance data from package resources
        rsrc = files(gridmet) / "balance_eto_gridmet_1981_2020.csv"
        self.data = pd.read_csv(
            rsrc,
            # start_date not needed for analysis
            usecols=["area", "end_date", "value"],
            # enforce date type for end_date column
            parse_dates=["end_date"],
        )
        # add integer column for the month number
        self.data["month"] = self.data["end_date"].apply(lambda x: x.month)

    def check_fit(self, window: int = 1) -> pd.DataFrame:
        """Check the goodness-of-fit of the SPEI for a given window length.

        Uses the two-sided Kolmogorov-Smirnov goodness-of-fit test to compare
        the SPEI against the standard normal distribution.

        Parameters
        ----------
        window : int, default=1
            Length of the moving average window.
            Must be between 1 and 48 months (inclusive).

        Returns
        -------
        data_meta : pandas.DataFrame
            A pivot table of p-values from the Kolmogorov-Smirnov test.

        Raises
        ------
        ValueError
            If `window` is not between 1 and 48 (inclusive).
        """
        if window < 1 or window > 48:
            raise ValueError(
                "Moving average window length must be between 1 and 48 months (inclusive)."
            )

        # generate SPEI series
        spei_df = self.generate_series(window=window)

        # get metadata for distributions
        data_meta = self.data[["area", "month"]].drop_duplicates().copy(deep=True)

        def _apply_test(row: pd.Series, spei_df: pd.DataFrame):
            """Apply the two-sided Kolmogorov-Smirnov test to the SPEI data.

            Parameters
            ----------
            row : pandas.Series
            spei_df : pandas.DataFrame

            Returns
            -------
            pvalue : float
            """
            return kstest(
                rvs=spei_df.query(
                    f"area == '{row['area']}' and month == {row['month']}"
                )["SPEI"],
                cdf=norm.cdf,
                alternative="two-sided",
            ).pvalue

        # check goodness-of-fit for each distribution
        data_meta["pvalue"] = data_meta.apply(
            # apply test to SPEI from generated dataframe
            partial(_apply_test, spei_df=spei_df),
            axis="columns",
            # pass each row as a pandas.Series object
            raw=False,
        )

        # transform metadata into a pivot table
        return data_meta.pivot_table(values="pvalue", index="area", columns="month")

    def generate_series(self, window: int = 1) -> pd.DataFrame:
        """Generate the SPEI for a given window length.

        Parameters
        ----------
        window : int, default=1
            Length of the moving average window.
            Must be between 1 and 48 months (inclusive).

        Returns
        -------
        data_gen : pandas.DataFrame
            A DataFrame containing the following columns:
            `area`, `end_date`, `month`, `SPEI`

        Raises
        ------
        ValueError
            If `window` is not between 1 and 48 (inclusive).
        """
        if window < 1 or window > 48:
            raise ValueError(
                "Moving average window length must be between 1 and 48 months (inclusive)."
            )

        data_gen = self.data.copy(deep=True)
        # apply moving average to climatic water balance data
        data_gen["value"] = (
            data_gen.groupby(by="area")
            .rolling(window=window, center=False, on="end_date")
            .mean()
            .reset_index()
            .set_index("level_1")["value"]
        )
        # restrict data to study period (1991-2020)
        data_gen = data_gen.query("end_date >= '1991-01-01'").copy(deep=True)

        def _inv_norm_wrapper(value: float, kde: GaussianKDE1D) -> float:
            return inv_norm(kde.opt_cdf(value))

        transformed_dict = {"index": [], "SPEI": []}
        for area in ["Bitterroot", "Gallatin"]:
            for month in range(1, 13):
                month_data = data_gen.query(f"area == '{area}' and month == {month}")[
                    "value"
                ]
                transformed_dict["index"].extend(month_data.index)
                month_kde = GaussianKDE1D(month_data.values)
                transformed_dict["SPEI"].extend(
                    map(partial(_inv_norm_wrapper, kde=month_kde), month_data.values)
                )
        data_gen["SPEI"] = pd.Series(
            data=transformed_dict["SPEI"], index=transformed_dict["index"]
        )

        # ignore computation columns for return value
        return data_gen[["area", "end_date", "month", "SPEI"]].sort_values(
            by=["area", "end_date"], ascending=True
        )
