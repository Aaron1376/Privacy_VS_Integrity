"""Run all three probes at a chosen scale and write CSVs (and figures) to --out."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("PVI_THREADS", "1")))
from pvi.probes import backdoor, canary, tension  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scale", default="quick", choices=["smoke", "quick", "full"])
ap.add_argument("--out", default="results/preliminary")
ap.add_argument("--only", default="", help="comma list: backdoor,canary,tension")
a = ap.parse_args()
only = set(filter(None, a.only.split(",")))

if not only or "backdoor" in only:
    backdoor.run(a.scale, f"{a.out}/backdoor_sweep.csv")
if not only or "canary" in only:
    canary.run(a.scale, f"{a.out}/canary_exposure.csv")
if not only or "tension" in only:
    tension.run(a.scale, f"{a.out}/tension.csv")

import subprocess  # noqa: E402

subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "make_figures.py"), "--in", a.out, "--out", a.out], check=False)
