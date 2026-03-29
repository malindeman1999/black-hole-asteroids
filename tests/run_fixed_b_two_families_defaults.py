from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAVER = PROJECT_ROOT / "sky_projections" / "save_sky_candidates_fixed_b_two_families_b10_a21.py"
PLOTTER = PROJECT_ROOT / "tests" / "plot_sky_candidates_fixed_b_two_families_b10_a21.py"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run fixed-B two-family solver and then plotter using each program's defaults."
    )
    p.add_argument("--no-show", action="store_true", help="Disable interactive plotting window.")
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def main() -> None:
    args = _parse_args()
    py = sys.executable

    _run([py, str(SAVER)])

    plot_cmd = [py, str(PLOTTER)]
    if args.no_show:
        plot_cmd.append("--no-show")
    _run(plot_cmd)

    print("Done: defaults solver + plotter completed.", flush=True)


if __name__ == "__main__":
    main()

