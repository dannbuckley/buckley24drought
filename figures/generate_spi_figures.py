from argparse import ArgumentParser

from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from buckley24drought.spi import SPI


def generate_fit_heatmap(window: int = 1):
    if window < 1 or window > 48:
        raise ValueError("SPI window must be between 1 and 48 (inclusive).")

    spi = SPI()
    fit_check = spi.check_fit(window=window)
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(
        data=fit_check,
        vmin=0,
        vmax=1,
        cmap="Reds_r",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "P-Value"},
        ax=ax,
    )
    ax.set_title(
        f"SPI-{window} Goodness-of-Fit\nP-Values from Two-Sided Kolmogorov-Smirnov Test",
        fontweight="bold",
    )
    ax.set_xlabel("Month", fontweight="bold")
    ax.set_ylabel("Area", fontweight="bold")
    fig.savefig(fname=f"spi/fit/spi_fit_{window}.svg", dpi=300, format="svg")
    plt.close(fig)
    print(f"Figure spi/fit/spi_fit_{window}.svg generated.")


def generate_series_plot(window: int = 1):
    if window < 1 or window > 48:
        raise ValueError("SPI window must be between 1 and 48 (inclusive).")

    spi_df = pd.read_csv(f"../index_data/spi/spi_{window}.csv", parse_dates=[1])
    spi_df["year"] = spi_df["end_date"].apply(lambda x: x.year)

    fig, ax = plt.subplots(nrows=2, ncols=1, squeeze=True, figsize=(8, 5))
    for i, area in enumerate(["Bitterroot", "Gallatin"]):
        temp_df = spi_df.query(f"area == '{area}'")
        ax[i].set_xlim(temp_df.index[0], temp_df.index[-1])
        ax[i].set_ylim(
            np.min([np.min(temp_df["SPI"]), -3]), np.max([np.max(temp_df["SPI"]), 3])
        )

        ax[i].axhline(y=0, color="black")
        ax[i].fill_between(
            x=temp_df.index,
            y1=temp_df["SPI"],
            y2=0,
            where=temp_df["SPI"] > 0,
            step="mid",
            edgecolor=to_rgba("tab:blue", 0.8),
            facecolor=to_rgba("tab:cyan", 0.6),
            lw=1.5,
        )
        ax[i].fill_between(
            x=temp_df.index,
            y1=temp_df["SPI"],
            y2=0,
            where=temp_df["SPI"] <= 0,
            step="mid",
            edgecolor=to_rgba("firebrick", 0.8),
            facecolor=to_rgba("tab:red", 0.6),
            lw=1.5,
        )

        xpos = temp_df.query("year % 5 == 0 and month == 1")["year"]
        ax[i].set_xticks(xpos.index, xpos.astype(str).values)
        ax[i].set_ylabel(area.split(" ")[0])

        sns.despine(ax=ax[i])

    fig.suptitle(f"SPI-{window}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"spi/series/spi_{window}.svg", dpi=300, format="svg")
    plt.close(fig)
    print(f"Figure spi/series/spi_{window}.svg generated.")


parser = ArgumentParser()
parser.add_argument("figtype")
args = parser.parse_args()

if __name__ == "__main__":
    match args.figtype:
        case "fit_heatmap":
            for w in range(1, 49):
                generate_fit_heatmap(window=w)
        case "series_plot":
            for w in range(1, 49):
                generate_series_plot(window=w)
