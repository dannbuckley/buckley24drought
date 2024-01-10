import matplotlib.pyplot as plt
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
    ax.set_xlabel("Month")
    ax.set_ylabel("Area")
    fig.savefig(fname=f"spi/fit/spi_fit_{window}.svg", dpi=300, format="svg")
    plt.close(fig)
    print(f"Figure spi/fit/spi_fit_{window}.svg generated.")


if __name__ == "__main__":
    for w in range(1, 49):
        generate_fit_heatmap(window=w)
