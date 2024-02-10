from time import time
from buckley24drought.spei import SPEI


def generate_spei_data(window: int = 1):
    if window < 1 or window > 48:
        raise ValueError("SPEI window must be between 1 and 48 (inclusive).")

    try:
        spei = SPEI()
        time_start = time()
        df = spei.generate_series(window=window)
        time_end = time()
        df.to_csv(f"spei/spei_{window}.csv", index=False, header=True)
        print(
            f"SPEI-{window} series saved at spei/spei_{window}.csv ({time_end - time_start:.1f})"
        )
    except ValueError:
        print(f"SPEI-{window} skipped.")


if __name__ == "__main__":
    for w in range(1, 49):
        generate_spei_data(window=w)
