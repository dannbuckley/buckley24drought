from os.path import exists
from time import time

from buckley24drought.sgi import SGI


def generate_sgi_data(gwicid: int):
    if not gwicid in SGI.all_wells:
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
    for g in SGI.all_wells:
        print(f"Starting GWIC ID {g}...")
        generate_sgi_data(gwicid=g)
