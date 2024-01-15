from buckley24drought.spei import SPEI


def generate_spei_data(window: int = 1):
    if window < 1 or window > 48:
        raise ValueError("SPEI window must be between 1 and 48 (inclusive).")

    try:
        spei = SPEI()
        df = spei.generate_series(window=window)
        df.to_csv(f"spei/spei_{window}.csv", index=False, header=True)
        print(f"SPEI-{window} series saved at spei/spei_{window}.csv")
    except ValueError:
        print(f"SPEI-{window} skipped.")


if __name__ == "__main__":
    for w in range(1, 49):
        generate_spei_data(window=w)
