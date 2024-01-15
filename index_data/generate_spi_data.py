from buckley24drought.spi import SPI


def generate_spi_data(window: int = 1):
    if window < 1 or window > 48:
        raise ValueError("SPI window must be between 1 and 48 (inclusive).")

    spi = SPI()
    df = spi.generate_series(window=window)
    df.to_csv(f"spi/spi_{window}.csv", index=False, header=True)
    print(f"SPI-{window} series saved at spi/spi_{window}.csv")


if __name__ == "__main__":
    for w in range(1, 49):
        generate_spi_data(window=w)
