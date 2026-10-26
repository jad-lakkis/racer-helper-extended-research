"""
Build the file that tells the RL agent which driver to drive like: his style ranges per corner, the corner definitions and his
fastest laps as reference lines / speeds. Run after scripts/extract_style_vectors.py.

    python scripts/build_driver_package.py ../style_vectors/Hockolicious --driver tastysnack2 lundstrm
    python scripts/build_driver_package.py ../style_vectors/Hockolicious --top-drivers 5

writes ../style_vectors/Hockolicious/packages/<driver>.npz (about 80 kB each).
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trackmania_rl.driver_context import DriverPackage  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("style_dir", type=Path, help="style_vectors/<track> written by extract_style_vectors.py")
    ap.add_argument("--driver", nargs="+", default=[], help="driver login(s)")
    ap.add_argument("--top-drivers", type=int, default=0, help="also the N drivers with the fastest lap on this track")
    ap.add_argument("--top-k", type=int, default=5, help="reference laps per driver; must equal the --top-k of the extraction")
    ap.add_argument("--min-laps", type=int, default=2, help="skip drivers with fewer laps than this")
    ap.add_argument("--out-dir", type=Path, help="default: <style_dir>/packages")
    args = ap.parse_args()

    drivers = list(args.driver)
    if args.top_drivers:
        laps = pd.read_csv(args.style_dir / "laps.csv")
        counts = laps.groupby("driver").size()
        fastest = laps.groupby("driver")["lap_time_ms"].min().sort_values()
        drivers += [d for d in fastest.index if counts[d] >= args.min_laps and d not in drivers][: args.top_drivers]
    if not drivers:
        sys.exit("give --driver or --top-drivers")

    out_dir = args.out_dir or args.style_dir / "packages"
    for d in drivers:
        pkg = DriverPackage.build(args.style_dir, d, k=args.top_k)
        path = out_dir / (re.sub(r"[^A-Za-z0-9._-]+", "-", d) + ".npz")
        pkg.save(path)
        print(f"{path}  ({path.stat().st_size / 1024:.0f} kB)  {pkg.describe()}")


if __name__ == "__main__":
    main()
