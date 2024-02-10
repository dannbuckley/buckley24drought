from argparse import ArgumentParser
from os.path import exists

from matplotlib.colors import to_rgba
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

    # load all SPEI data into one dataframe
    spei_df = pd.read_csv(
        "../index_data/spei/spei_1.csv",
        usecols=["area", "end_date", "SPEI"],
        parse_dates=["end_date"],
    ).rename(columns={"SPEI": "SPEI_1"})
    # merge in remaining SPEI records
    for window in range(2, 49):
        spei_df = spei_df.merge(
            right=pd.read_csv(
                f"../index_data/spei/spei_{window}.csv",
                usecols=["area", "end_date", "SPEI"],
                parse_dates=["end_date"],
            ).rename(columns={"SPEI": f"SPEI_{window}"}),
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
        title: str,
        filename: str,
        corr_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        spei: bool = False,
    ):
        """Generate a cross-correlation heatmap for the given area.

        Parameters
        ----------
        title : str
        filename : str
        corr_df : pandas.DataFrame
        meta_df : pandas.DataFrame
        """
        index_label = "SPEI" if spei else "SPI"
        fig, ax = plt.subplots(figsize=(corr_df.shape[1] // 2, corr_df.shape[0] // 2))
        sns.heatmap(data=corr_df, vmax=1, cmap="RdGy_r", center=0, ax=ax)
        ax.set_xlabel(f"{index_label} Accumulation Period", fontweight="bold")
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
            f"{index_label} vs. SGI Cross-Correlation (Spearman) for the {title}",
            fontsize=18,
            fontweight="bold",
        )
        fig.savefig(
            fname=f"corr/cross/heatmap/{filename}_{index_label.lower()}.svg",
            dpi=300,
            format="svg",
        )
        print(
            f"Heatmap figure generated at corr/cross/heatmap/{filename}_{index_label.lower()}.svg"
        )
        plt.close(fig)

    def _generate_series_plots(
        corr_df: pd.DataFrame, meta_df: pd.DataFrame, spei: bool = False
    ):
        """Generate individual cross-correlation plots for each SGI record.

        Parameters
        ----------
        corr_df : pandas.DataFrame
        meta_df : pandas.DataFrame
        """
        index_label = "SPEI" if spei else "SPI"
        xticks = np.array([1, 3, 6, 12, 24, 48])
        for gwicid in meta_df["gwicid"]:
            corr_val = corr_df.loc[f"SGI_{gwicid}"]

            fig, ax = plt.subplots()
            ax.plot(np.arange(1, 49), corr_val, color=to_rgba("tab:gray", 0.6))
            # plot maximum cross-correlation as point
            ax.plot(
                [np.argmax(corr_val) + 1], [corr_val.max()], "ko", label="Max. Corr."
            )
            ax.legend()
            corr_ymin, corr_ymax = ax.get_ylim()
            ax.axvline(
                x=np.argmax(corr_val) + 1,
                ymax=(corr_val.max() - corr_ymin) / (corr_ymax - corr_ymin),
                color="black",
                ls="--",
            )
            ax.set_xlim(0, 49)
            ax.set_xticks(xticks, xticks.astype(str))
            ax.set_xlabel(f"{index_label} Accumulation Period")
            ax.set_ylabel("Cross-correlation (Spearman)")
            ax.set_title(
                f"{index_label} vs. SGI for GWIC ID {gwicid}",
                loc="left",
                fontweight="bold",
            )
            sns.despine(ax=ax)
            fig.savefig(
                fname=f"corr/cross/series/{index_label.lower()}_sgi_{gwicid}.svg",
                dpi=300,
                format="svg",
            )
            print(
                f"Cross-correlation series figure generated at corr/cross/series/{index_label.lower()}_sgi_{gwicid}.svg"
            )
            plt.close(fig)

    def _generate_met_sgi_mosaic_plots(
        area: str, corr_df: pd.DataFrame, meta_df: pd.DataFrame, spei: bool = False
    ):
        """Combine SPI/SPEI series, SGI series, and cross-correlation series into one figure.

        Parameters
        ----------
        area : str
            Area name as used in the SPI dataframe
        corr_df : pandas.DataFrame
        meta_df : pandas.DataFrame
        """
        corr_xticks = np.array([1, 3, 6, 12, 24, 48])
        index_label = "SPEI" if spei else "SPI"
        for gwicid in meta_df["gwicid"]:
            corr_val = corr_df.loc[f"SGI_{gwicid}"]

            met_max_per = np.argmax(corr_val) + 1
            met_sgi_df = pd.merge(
                left=pd.read_csv(
                    f"../index_data/{index_label.lower()}/{index_label.lower()}_{met_max_per}.csv",
                    usecols=["area", "end_date", index_label],
                    parse_dates=["end_date"],
                )
                .query(f"area == '{area}'")
                .drop(columns="area")
                .sort_values(by="end_date", ascending=True),
                right=pd.read_csv(
                    f"../index_data/sgi/sgi_{gwicid}.csv",
                    usecols=["date", "SGI"],
                    parse_dates=["date"],
                ).rename(
                    columns={
                        "date": "end_date",
                    }
                ),
                how="left",
                left_on=["end_date"],
                right_on=["end_date"],
            )

            fig = plt.figure(figsize=(10, 4), layout="constrained")
            ax_dict = fig.subplot_mosaic(
                mosaic=[[index_label, "corr"], ["sgi", "corr"]], width_ratios=[1.5, 1]
            )

            for _, ax in ax_dict.items():
                sns.despine(ax=ax)

            # === SPI/SPEI PLOT ===
            ax_dict[index_label].axhline(y=0, color="black")
            ax_dict[index_label].fill_between(
                x=met_sgi_df["end_date"],
                y1=met_sgi_df[index_label],
                y2=0,
                where=met_sgi_df[index_label] > 0,
                step="mid",
                edgecolor=to_rgba("tab:blue", 0.8),
                facecolor=to_rgba("tab:cyan", 0.6),
                lw=1.5,
            )
            ax_dict[index_label].fill_between(
                x=met_sgi_df["end_date"],
                y1=met_sgi_df[index_label],
                y2=0,
                where=met_sgi_df[index_label] <= 0,
                step="mid",
                edgecolor=to_rgba("firebrick", 0.8),
                facecolor=to_rgba("tab:red", 0.6),
                lw=1.5,
            )
            ax_dict[index_label].set_xticks(
                ax_dict[index_label].get_xticks(),
                ["" for _ in ax_dict[index_label].get_xticks()],
            )
            ax_dict[index_label].set_xlim(
                met_sgi_df["end_date"].iloc[0], met_sgi_df["end_date"].iloc[-1]
            )
            ax_dict[index_label].set_ylim(
                np.min([np.min(met_sgi_df[index_label]), -3]),
                np.max([np.max(met_sgi_df[index_label]), 3]),
            )
            ax_dict[index_label].set_ylabel(f"{index_label}-{met_max_per}")
            ax_dict[index_label].set_title("(a)", loc="left", fontweight="bold")

            # === SGI PLOT ===
            ax_dict["sgi"].axhline(y=0, color="black")
            sns.lineplot(
                data=met_sgi_df,
                x="end_date",
                y="SGI",
                color=to_rgba("tab:gray", 0.6),
                ax=ax_dict["sgi"],
            )
            sns.scatterplot(
                data=met_sgi_df,
                x="end_date",
                y="SGI",
                color="black",
                ax=ax_dict["sgi"],
                zorder=10,
                s=15,
            )
            ax_dict["sgi"].set_xlim(
                met_sgi_df["end_date"].iloc[0], met_sgi_df["end_date"].iloc[-1]
            )
            ax_dict["sgi"].set_ylim(
                np.min([np.min(met_sgi_df["SGI"]), -3]),
                np.max([np.max(met_sgi_df["SGI"]), 3]),
            )
            ax_dict["sgi"].set_xlabel("Date")
            ax_dict["sgi"].set_ylabel(f"SGI (GWIC {gwicid})")
            ax_dict["sgi"].set_title("(b)", loc="left", fontweight="bold")

            # === CROSS-CORRELATION PLOT ===
            ax_dict["corr"].plot(
                np.arange(1, 49), corr_val, color=to_rgba("tab:gray", 0.6)
            )
            ax_dict["corr"].plot(
                [met_max_per], [corr_val.max()], "ko", label="Max. Corr."
            )
            ax_dict["corr"].legend()
            corr_ymin, corr_ymax = ax_dict["corr"].get_ylim()
            ax_dict["corr"].axvline(
                x=met_max_per,
                ymax=(corr_val.max() - corr_ymin) / (corr_ymax - corr_ymin),
                color="black",
                ls="--",
            )
            ax_dict["corr"].set_xlim(0, 49)
            ax_dict["corr"].set_xticks(corr_xticks, corr_xticks.astype(str))
            ax_dict["corr"].set_xlabel(f"{index_label} Accumulation Period")
            ax_dict["corr"].set_title("(c)", loc="left", fontweight="bold")

            fig.savefig(
                fname=f"corr/cross/mosaic/max{index_label.lower()}_sgi/max{index_label.lower()}_sgi_{gwicid}.svg",
                dpi=300,
                format="svg",
            )
            print(
                f"Mosaic {index_label}-SGI figure generated at corr/cross/mosaic/max{index_label.lower()}_sgi/max{index_label.lower()}_sgi_{gwicid}.svg"
            )
            plt.close(fig)

    bv_corr_spi = _get_cross_corr_df("Bitterroot", bv_meta, spi_df)
    gv_corr_spi = _get_cross_corr_df("Gallatin", gv_meta, spi_df)
    bv_corr_spei = _get_cross_corr_df("Bitterroot", bv_meta, spei_df)
    gv_corr_spei = _get_cross_corr_df("Gallatin", gv_meta, spei_df)
    _generate_heatmap("Bitterroot Valley", "bitterroot", bv_corr_spi, bv_meta)
    _generate_heatmap("Gallatin Valley", "gallatin", gv_corr_spi, gv_meta)
    _generate_heatmap(
        "Bitterroot Valley", "bitterroot", bv_corr_spei, bv_meta, spei=True
    )
    _generate_heatmap("Gallatin Valley", "gallatin", gv_corr_spei, gv_meta, spei=True)
    _generate_series_plots(bv_corr_spi, bv_meta)
    _generate_series_plots(gv_corr_spi, gv_meta)
    _generate_series_plots(bv_corr_spei, bv_meta, spei=True)
    _generate_series_plots(gv_corr_spei, gv_meta, spei=True)
    _generate_met_sgi_mosaic_plots("Bitterroot", bv_corr_spi, bv_meta)
    _generate_met_sgi_mosaic_plots("Gallatin", gv_corr_spi, gv_meta)
    _generate_met_sgi_mosaic_plots("Bitterroot", bv_corr_spei, bv_meta, spei=True)
    _generate_met_sgi_mosaic_plots("Gallatin", gv_corr_spei, gv_meta, spei=True)


parser = ArgumentParser()
parser.add_argument("figtype")
args = parser.parse_args()

if __name__ == "__main__":
    match args.figtype:
        case "cross":
            generate_crosscorr_plots()
