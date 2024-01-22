from argparse import ArgumentParser
from os.path import exists

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def generate_crosscorr_plots():
    # load in Bitterroot Valley metadata
    bv_meta = pd.read_csv("../src/buckley24drought/data/gw/Bitterroot_gw_well_meta.csv")
    bv_meta = bv_meta[
        bv_meta["gwicid"].apply(lambda x: exists(f"../index_data/sgi/sgi_{x}.csv"))
    ]

    # load in Gallatin Valley metadata
    gv_meta = pd.read_csv("../src/buckley24drought/data/gw/Gallatin_gw_well_meta.csv")
    gv_meta = gv_meta[
        gv_meta["gwicid"].apply(lambda x: exists(f"../index_data/sgi/sgi_{x}.csv"))
    ]

    # load in Paradise Valley metadata
    pv_meta = pd.read_csv("../src/buckley24drought/data/gw/Paradise_gw_well_meta.csv")
    pv_meta = pv_meta[
        pv_meta["gwicid"].apply(lambda x: exists(f"../index_data/sgi/sgi_{x}.csv"))
    ]

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

    # setup dataframe for Bitterroot Valley cross-correlation
    bv_spi_sgi = (
        spi_df.query("area == 'Bitterroot'")
        .sort_values(by="end_date", ascending=True)
        .copy()
    )
    # merge in SGI data for Bitterroot Valley
    for gwicid in bv_meta["gwicid"]:
        bv_spi_sgi = bv_spi_sgi.merge(
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

    # generate SPI vs. SGI cross-correlation data for Bitterroot Valley
    bv_corr = (
        bv_spi_sgi[bv_spi_sgi.columns[2:]]
        .corr(method="spearman")
        .loc[bv_spi_sgi.columns[50:]][bv_spi_sgi.columns[2:50]]
    )

    # generate cross-correlation heatmap for Bitterroot Valley
    fig, ax = plt.subplots(figsize=(bv_corr.shape[1] // 2, bv_corr.shape[0] // 2))
    sns.heatmap(data=bv_corr, vmax=1, cmap="RdGy_r", center=0, ax=ax)
    fig.savefig(fname="corr/cross/heatmap/bitterroot.svg", dpi=300, format="svg")
    plt.close(fig)

    # setup dataframe for Gallatin Valley cross-correlation
    gv_spi_sgi = (
        spi_df.query("area == 'Gallatin'")
        .sort_values(by="end_date", ascending=True)
        .copy()
    )
    # merge in SGI data for Gallatin Valley
    for gwicid in gv_meta["gwicid"]:
        gv_spi_sgi = gv_spi_sgi.merge(
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

    # setup dataframe for Paradise Valley cross-correlation
    pv_spi_sgi = (
        spi_df.query("area == 'Paradise Valley'")
        .sort_values(by="end_date", ascending=True)
        .copy()
    )
    # merge in SGI data for Paradise Valley
    for gwicid in pv_meta["gwicid"]:
        pv_spi_sgi = pv_spi_sgi.merge(
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


parser = ArgumentParser()
parser.add_argument("figtype")
args = parser.parse_args()

if __name__ == "__main__":
    match args.figtype:
        case "cross":
            generate_crosscorr_plots()
