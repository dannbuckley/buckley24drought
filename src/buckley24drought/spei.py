"""SPEI module"""

from functools import partial
from importlib.resources import files, as_file

import numpy as np
import pandas as pd
from scipy.special import gamma
from scipy.stats import kstest, norm

from .data import gridmet
from .util import inv_norm


def loglogistic_cdf(value: float, alpha: float, beta: float, gamma: float):
    """CDF of the 3-parameter log-logistic distribution (see Singh, Guo, & Yu, 1993)

    Parameters
    ----------
    value : float
        Must be greater than `gamma`.
    alpha : float
        Must be greater than 0 (zero).
    beta : float
        Must be greater than or equal to 1 (one).
    gamma : float
        Must be less than `value`.

    Returns
    -------
    float
        The CDF evaluated at `value`.

    Raises
    ------
    ValueError
        If `alpha` <= 0, `value` <= `gamma`, or `beta` < 1.
    """
    if alpha <= 0:
        raise ValueError("Parameter alpha must be greater than 0 (zero).")
    elif value <= gamma:
        raise ValueError("Value must be greater than the gamma parameter.")
    elif beta < 1:
        raise ValueError("Parameter beta must be greater than or equal to 1 (one).")
    return np.power((value - gamma) / alpha, beta) / (
        1 + np.power((value - gamma) / alpha, beta)
    )


class SPEI:
    """SPEI class

    Attributes
    ----------
    data : pandas.DataFrame
        Climatic water balance data from which to generate the SPEI.
    """

    def __init__(self):
        # get climatic water balance data from package resources
        rsrc = files(gridmet).joinpath("balance_eto_gridmet_1981_2020.csv")
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

    def check_fit(self, window: int = 1):
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

    def generate_series(self, window: int = 1):
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

        # prepare for probability weighted moments (1/2):
        # get rank i of each value
        data_gen["rank"] = (
            data_gen[["area", "month", "value"]]
            .groupby(by=["area", "month"])
            .rank(ascending=True, pct=False)
            .astype(int)
        )

        # prepare for probability weighted moments (2/2):
        # get number of entries n for each month
        data_gen = data_gen.merge(
            right=data_gen.groupby(by=["area", "month"])["value"]
            .count()
            .reset_index()
            .rename(columns={"value": "count"}),
            left_on=["area", "month"],
            right_on=["area", "month"],
        )

        # compute probability weighted moments in separate dataframe
        pwm = (
            pd.DataFrame(
                {
                    "area": data_gen["area"],
                    "month": data_gen["month"],
                    "w0": data_gen.eval("value / count"),
                    "w1": data_gen.eval(
                        "value / count * (1 - ((rank - 0.35) / count))"
                    ),
                    "w2": data_gen.eval(
                        "value / count * ((1 - ((rank - 0.35) / count)) ** 2)"
                    ),
                }
            )
            .groupby(by=["area", "month"])
            .sum()
            .reset_index()
        )

        # compute beta estimate
        pwm["beta"] = pwm.eval("((2 * w1) - w0) / ((6 * w1) - w0 - (6 * w2))")

        # compute alpha estimate
        pwm["opo_beta"] = pwm.eval("1 + (1 / beta)").apply(gamma)
        pwm["omo_beta"] = pwm.eval("1 - (1 / beta)").apply(gamma)
        pwm["alpha"] = pwm.eval("((w0 - (2 * w1)) * beta) / (opo_beta * omo_beta)")

        # compute gamma estimate
        pwm["gamma"] = pwm.eval("w0 - ((w0 - (2 * w1)) * beta)")

        # drop intermediate steps, keep final parameter estimates
        pwm = pwm.drop(columns=["w0", "w1", "w2", "opo_beta", "omo_beta"])

        # compute SPEI using parameter estimates
        data_gen["SPEI"] = data_gen.merge(
            right=pwm, left_on=["area", "month"], right_on=["area", "month"]
        )[["value", "alpha", "beta", "gamma"]].apply(
            # apply inverse normal transform on each row
            lambda row: inv_norm(
                loglogistic_cdf(row["value"], row["alpha"], row["beta"], row["gamma"])
            ),
            axis="columns",
            # pass each row as a pandas.Series object
            raw=False,
        )

        # ignore computation columns for return value
        return data_gen[["area", "end_date", "month", "SPEI"]].sort_values(
            by=["area", "end_date"], ascending=True
        )
