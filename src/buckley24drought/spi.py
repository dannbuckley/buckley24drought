"""SPI module"""
# pylint: disable=R0801
from functools import partial
from importlib.resources import files, as_file

import numpy as np
import pandas as pd
from scipy.stats import gamma, kstest, norm

from .data import gridmet
from .util import inv_norm


class SPI:
    """SPI class

    Attributes
    ----------
    data : pandas.DataFrame
        Precipitation data from which to generate the SPI.
    """

    def __init__(self):
        # get precipitation data from package resources
        rsrc = files(gridmet).joinpath("precip_gridmet_1981_2020.csv")
        with as_file(rsrc) as csv:
            self.data = pd.read_csv(
                csv,
                # start_date not needed for analysis
                usecols=["area", "end_date", "value"],
                # enforce date type for end_date column
                parse_dates=[1],
            )
        # add integer column for the month number
        self.data["month"] = self.data["end_date"].apply(lambda x: x.month)

    def check_fit(self, window: int = 1) -> pd.DataFrame:
        """Check the goodness-of-fit of the SPI for a given window length.

        Uses the two-sided Kolmogorov-Smirnov goodness-of-fit test to compare
        the SPI against the standard normal distribution.

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

        # generate SPI series
        spi_df = self.generate_series(window=window)

        # get metadata for distributions
        data_meta = self.data[["area", "month"]].drop_duplicates().copy(deep=True)

        def _apply_test(row: pd.Series, spi_df: pd.DataFrame):
            return kstest(
                rvs=spi_df.query(
                    f"area == '{row['area']}' and month == {row['month']}"
                )["SPI"],
                cdf=norm.cdf,
                alternative="two-sided",
            ).pvalue

        # check goodness-of-fit for each distribution
        data_meta["pvalue"] = data_meta.apply(
            # apply test to SPI from generated dataframe
            partial(_apply_test, spi_df=spi_df),
            axis="columns",
            # pass each row as a pandas.Series object
            raw=False,
        )

        # transform metadata into a pivot table
        return data_meta.pivot_table(values="pvalue", index="area", columns="month")

    def generate_series(self, window: int = 1) -> pd.DataFrame:
        """Generate the SPI for a given window length.

        Parameters
        ----------
        window : int, default=1
            Length of the moving average window.
            Must be between 1 and 48 months (inclusive).

        Returns
        -------
        data_gen : pandas.DataFrame
            A DataFrame containing the following columns:
            `area`, `end_date`, `month`, `SPI`

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
        # apply moving average to precipitation data
        data_gen["value"] = (
            data_gen.groupby(by="area")
            .rolling(window=window, center=False, on="end_date")
            .mean()
            .reset_index()
            .set_index("level_1")["value"]
        )
        # restrict data to study period (1991-2020)
        data_gen = data_gen.query("end_date >= '1991-01-01'").copy(deep=True)

        # prepare for l-moments (1/2): get rank i of each value
        data_gen["rank"] = (
            data_gen[["area", "month", "value"]]
            .groupby(by=["area", "month"])
            .rank(ascending=True, pct=False)
            .astype(int)
        )

        # prepare for l-moments (2/2): get number of entries n for each month
        data_gen = data_gen.merge(
            right=data_gen.groupby(by=["area", "month"])["value"]
            .count()
            .reset_index()
            .rename(columns={"value": "count"}),
            left_on=["area", "month"],
            right_on=["area", "month"],
        )

        # compute l-moments in separate dataframe
        lmom = (
            pd.DataFrame(
                {
                    "area": data_gen["area"],
                    "month": data_gen["month"],
                    # bring fractions inside the summations for l-moment calculations
                    "l1": data_gen.eval("value / count"),
                    "l2": data_gen.eval(
                        "(value * ((2 * rank) - count - 1)) / (count * (count - 1))"
                    ),
                }
            )
            .groupby(by=["area", "month"])
            .sum()
            .reset_index()
        )

        # compute L-CV t
        lmom["t"] = lmom.eval("l2 / l1")

        # for easier computation of piecewise functions:
        # express "t in interval" comparison as integer
        lmom["t_under_half"] = lmom.eval("0 < t < 0.5").astype(int)
        lmom["t_over_half"] = lmom.eval("0.5 <= t < 1").astype(int)
        # setup evaluation string for piecewise computations
        # using addition instead of a conditional here
        pc_eval_str = "(t_under_half * ({under})) + (t_over_half * ({over}))"

        # z is evaluated with piecewise function
        z_under_half = "@pi * t * t"
        z_over_half = "1 - t"
        lmom["z"] = lmom.eval(
            pc_eval_str.format(under=z_under_half, over=z_over_half),
            # interpret numpy pi constant as local variable
            local_dict={"pi": np.pi},
        )

        # alpha is evaluated with piecewise function
        # under-half equivalent to: (1 - 0.308z) / (z - 0.05812z^2 + 0.01765z^3)
        alpha_under_half = (
            "(1 + (-0.308 * z)) / (z * (1 + (z * (-0.05812 + (0.01765 * z)))))"
        )
        # over-half equivalent to: (0.7213z - 0.5947z^2) / (1 - 2.1817z + 1.2113z^2)
        alpha_over_half = (
            "(z * (0.7213 + (-0.5947 * z))) / (1 + (z * (-2.1817 + (1.2113 * z))))"
        )
        lmom["alpha"] = lmom.eval(
            pc_eval_str.format(under=alpha_under_half, over=alpha_over_half)
        )

        # since l1 is identical to the sample mean,
        # it can be used to estimate the population mean E[X] = alpha * beta
        lmom["beta"] = lmom.eval("l1 / alpha")

        # drop intermediate steps, keep final parameter estimates
        lmom = lmom.drop(columns=["l1", "l2", "t", "t_under_half", "t_over_half", "z"])

        # compute SPI using parameter estimates
        data_gen["SPI"] = data_gen.merge(
            right=lmom, left_on=["area", "month"], right_on=["area", "month"]
        )[["value", "alpha", "beta"]].apply(
            # apply inverse normal transform on each row
            lambda row: inv_norm(
                gamma.cdf(row["value"], row["alpha"], scale=row["beta"])
            ),
            axis="columns",
            # pass each row as a pandas.Series object
            raw=False,
        )

        # ignore computation columns for return value
        return data_gen[["area", "end_date", "month", "SPI"]].sort_values(
            by=["area", "end_date"], ascending=True
        )
