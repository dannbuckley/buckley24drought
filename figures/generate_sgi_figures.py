from argparse import ArgumentParser
from glob import glob
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generate_series_plots():
    sgi_files = list(map(Path, glob("../index_data/sgi/sgi_*.csv")))
    for file in sgi_files:
        sgi_df = pd.read_csv(file, usecols=["date", "SGI"], parse_dates=["date"])
        gwicid = re.match(r"sgi_(\d+).csv", file.name).group(1)

        fig, ax = plt.subplots(figsize=(15, 5))
        sns.lineplot(data=sgi_df, x="date", y="SGI", color="tab:gray", ax=ax)
        sns.scatterplot(data=sgi_df, x="date", y="SGI", color="black", ax=ax, zorder=10)
        ax.axhline(y=0, color="black")
        sns.despine(ax=ax)
        ax.set_xlabel("")
        ax.set_xlim(sgi_df["date"].iloc[0], sgi_df["date"].iloc[-1])
        ax.set_title(f"SGI for GWIC ID {gwicid}", loc="left", fontweight="bold")
        fig.savefig(f"sgi/series/sgi_{gwicid}.svg", dpi=300, format="svg")
        print(f"Series plot generated at sgi/series/sgi_{gwicid}.svg")
        plt.close(fig)


parser = ArgumentParser()
parser.add_argument("figtype")
args = parser.parse_args()

if __name__ == "__main__":
    match args.figtype:
        case "series_plot":
            generate_series_plots()
