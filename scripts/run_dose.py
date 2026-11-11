"""Dose-response probe (attacker count). Seeds can be split across processes: --seeds 0,1."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("PVI_THREADS", "1")))
from pvi.probes import backdoor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scale", default="full")
ap.add_argument("--seeds", default="")
ap.add_argument("--out", default="results/preliminary/backdoor_dose.csv")
a = ap.parse_args()
seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else None
backdoor.run_dose(a.scale, a.out, seeds)
