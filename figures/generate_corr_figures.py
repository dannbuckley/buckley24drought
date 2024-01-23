from argparse import ArgumentParser
from os.path import exists

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def generate_crosscorr_plots():
    def _get_meta_df(area: str) -> pd.DataFrame:
        """Retrieve the metadata dataframe for the given area.

        Parameters
        ----------
        area : str
            Area name as present in the filename

        Returns
        -------
        df_meta : pandas.DataFrame

        Raises
        ------
        AssertionError
            If the `gwicid` column is not present in the loaded dataframe.
        """
        df_meta = pd.read_csv(
            f"../src/buckley24drought/data/gw/{area}_gw_well_meta.csv"
        )
        assert "gwicid" in df_meta.columns, "Did not find gwicid column in dataframe!"
        # filter dataframe to records that have an available SGI data file
        return df_meta[
            df_meta["gwicid"].apply(
                lambda gwicid: exists(f"../index_data/sgi/sgi_{gwicid}.csv")
            )
        ].sort_values(by=["aquifer", "total_depth"], ascending=True)

    # load metadata for study areas
    bv_meta = _get_meta_df("Bitterroot")
    gv_meta = _get_meta_df("Gallatin")
    pv_meta = _get_meta_df("Paradise")

    # load all SPI data into one dataframe
    spi_df = pd.read_csv(
        "../index_data/spi/spi_1.csv",
        usecols=["area", "end_date", "SPI"],
        parse_dates=["end_date"],
    ).rename(columns={"SPI": "SPI_1"})
    # merge in remaining SPI records
    for window in range(2, 49):
        spi_df = spi_df.merge(
            right=pd.read_csv(
                f"../index_data/spi/spi_{window}.csv",
                usecols=["area", "end_date", "SPI"],
                parse_dates=["end_date"],
            ).rename(columns={"SPI": f"SPI_{window}"}),
            left_on=["area", "end_date"],
            right_on=["area", "end_date"],
        )

    def _get_cross_corr_df(
        area: str, meta_df: pd.DataFrame, spi_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Produce an SPI vs. SGI cross-correlation dataframe for the given area.

        Parameters
        ----------
        area : str
            Area name as used in the SPI dataframe
        meta_df : pandas.DataFrame
        spi_df : pandas.DataFrame

        Returns
        -------
        spi_sgi_df : pandas.DataFrame
            Dataframe in which the rows are the SGI records,
            the columns are the SPI records,
            and the values are the Spearman correlations.
        """
        spi_sgi_df = (
            spi_df.query(f"area == '{area}'")
            .sort_values(by="end_date", ascending=True)
            .copy(deep=True)
        )
        # merge in SGI data for area
        for gwicid in meta_df["gwicid"]:
            spi_sgi_df = spi_sgi_df.merge(
                right=pd.read_csv(
                    f"../index_data/sgi/sgi_{gwicid}.csv",
                    usecols=["date", "SGI"],
                    parse_dates=["date"],
                ).rename(
                    columns={
                        # avoid multiple date columns in merged dataframe
                        "date": "end_date",
                        "SGI": f"SGI_{gwicid}",
                    }
                ),
                # use SPI date array just in case some dates are missing from SGI record
                how="left",
                left_on="end_date",
                right_on="end_date",
            )
        # perform SPI vs. SGI cross-correlation
        return (
            spi_sgi_df[spi_sgi_df.columns[2:]].corr(method="spearman")
            # rows are SGI records
            .loc[spi_sgi_df.columns[50:]][
                # columns are SPI periods
                spi_sgi_df.columns[2:50]
            ]
        )

    def _generate_heatmap(
        title: str, filename: str, corr_df: pd.DataFrame, meta_df: pd.DataFrame
    ):
        """Generate a cross-correlation heatmap for the given area.

        Parameters
        ----------
        title : str
        filename : str
        corr_df : pandas.DataFrame
        meta_df : pandas.DataFrame
        """
        fig, ax = plt.subplots(figsize=(corr_df.shape[1] // 2, corr_df.shape[0] // 2))
        sns.heatmap(data=corr_df, vmax=1, cmap="RdGy_r", center=0, ax=ax)
        ax.set_xlabel("SPI Accumulation Period", fontweight="bold")
        # use upright integer tick labels for x-axis
        ax.set_xticks(
            ax.get_xticks(),
            [str(i + 1) for i, _ in enumerate(ax.get_xticks())],
            rotation=0,
        )
        # display GWIC ID, aquifer code, and total depth on y-axis
        ax.set_yticks(
            ax.get_yticks(),
            meta_df[["gwicid", "aquifer", "total_depth"]].apply(
                lambda row: f"{row['gwicid']}: {row['aquifer']}, {row['total_depth']} ft.",
                axis="columns",
                raw=False,
            ),
        )
        ax.set_title(
            f"SPI vs. SGI Cross-Correlation (Spearman) for the {title}",
            fontsize=18,
            fontweight="bold",
        )
        fig.savefig(fname=f"corr/cross/heatmap/{filename}.svg", dpi=300, format="svg")
        print(f"Heatmap figure generated at corr/cross/heatmap/{filename}.svg")
        plt.close(fig)

    def _generate_series_plots(corr_df: pd.DataFrame, meta_df: pd.DataFrame):
        """Generate individual cross-correlation plots for each SGI record."""
        xticks = np.array([1, 3, 6, 12, 24, 48])
        for gwicid in meta_df["gwicid"]:
            corr_val = corr_df.loc[f"SGI_{gwicid}"]

            fig, ax = plt.subplots()
            ax.plot(np.arange(1, 49), corr_val, color="tab:gray")
            # plot maximum cross-correlation as point
            ax.plot([np.argmax(corr_val) + 1], [corr_val.max()], "ko")
            ax.set_xlim(0, 49)
            ax.set_xticks(xticks, xticks.astype(str))
            ax.set_xlabel("SPI Accumulation Period")
            ax.set_ylabel("Cross-correlation (Spearman)")
            ax.set_title(
                f"SPI vs. SGI for GWIC ID {gwicid}", loc="left", fontweight="bold"
            )
            sns.despine(ax=ax)
            fig.savefig(
                fname=f"corr/cross/series/spi_sgi_{gwicid}.svg", dpi=300, format="svg"
            )
            print(
                f"Cross-correlation series figure generated at corr/cross/series/spi_sgi_{gwicid}.svg"
            )
            plt.close(fig)

    bv_corr = _get_cross_corr_df("Bitterroot", bv_meta, spi_df)
    gv_corr = _get_cross_corr_df("Gallatin", gv_meta, spi_df)
    pv_corr = _get_cross_corr_df("Paradise Valley", pv_meta, spi_df)
    _generate_heatmap("Bitterroot Valley", "bitterroot", bv_corr, bv_meta)
    _generate_heatmap("Gallatin Valley", "gallatin", gv_corr, gv_meta)
    _generate_series_plots(bv_corr, bv_meta)
    _generate_series_plots(gv_corr, gv_meta)
    _generate_series_plots(pv_corr, pv_meta)


parser = ArgumentParser()
parser.add_argument("figtype")
args = parser.parse_args()

if __name__ == "__main__":
    match args.figtype:
        case "cross":
            generate_crosscorr_plots()
