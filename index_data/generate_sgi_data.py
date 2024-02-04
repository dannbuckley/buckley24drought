from os.path import exists
from time import time

from buckley24drought.sgi import SGI


all_wells = [
    32,
    820,
    824,
    5418,
    9771,
    9858,
    50808,
    55463,
    56528,
    57128,
    57525,
    58096,
    60137,
    91230,
    91244,
    96132,
    96826,
    99215,
    123132,
    126793,
    129491,
    129952,
    130860,
    132260,
    133162,
    133165,
    133167,
    133172,
    133174,
    133176,
    135680,
    135689,
    135720,
    135722,
    135734,
    135735,
    136050,
    136486,
    136964,
    136969,
    136970,
    139989,
    140366,
    148531,
]


def generate_sgi_data(gwicid: int):
    if not gwicid in all_wells:
        raise ValueError(
            "GWIC ID does not have any associated data (for this project)."
        )

    if exists(f"sgi/sgi_{gwicid}.csv"):
        print(f"SGI for GWIC ID {gwicid} already computed.")
        return

    try:
        sgi = SGI(gwicid=gwicid)
        time_start = time()
        df = sgi.generate_series()
        time_end = time()
        df.to_csv(f"sgi/sgi_{gwicid}.csv", index=False, header=True)
        print(
            f"SGI for GWIC ID {gwicid} saved at sgi/sgi_{gwicid}.csv ({time_end - time_start:.1f})"
        )
    except ValueError:
        print(f"SGI for GWIC ID {gwicid} skipped.")


if __name__ == "__main__":
    for g in all_wells:
        print(f"Starting GWIC ID {g}...")
        generate_sgi_data(gwicid=g)
