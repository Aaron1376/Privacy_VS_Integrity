"""Validate docs/risk_register.csv, write a Markdown table and the before/after heat map."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pvi import risk_register as rr  # noqa: E402

rows = rr.priority_order(rr.load("docs/risk_register.csv"))
lines = ["| ID | Risk | L | I | Raw | Band | Control | L' | I' | Residual | Band |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['id']} | {r['risk']} | {r['L']} | {r['I']} | {r['raw']} | {r['band_raw']} | {r['control']} | {r['L_res']} | {r['I_res']} | {r['residual']} | {r['band_res']} |")
open("docs/risk_register.md", "w").write("# Risk register (sorted by residual risk)\n\n" + "\n".join(lines) + "\n\nJustifications and evidence per cell are in `risk_register.csv`.\n")
rr.heatmap(rows, "results/preliminary/fig_risk_heatmap.png")
print("\n".join(lines))
