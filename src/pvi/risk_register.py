"""Load the risk register, validate it, compute scores and bands, and draw the before/after heat map."""

from __future__ import annotations

import csv

REQUIRED = ["id", "risk", "L", "I", "control", "L_res", "I_res", "justification_raw", "justification_residual", "evidence"]


def band(score: int) -> str:
    return "Low" if score <= 4 else "Medium" if score <= 9 else "High" if score <= 15 else "Critical"


def load(path: str) -> list:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        missing = [k for k in REQUIRED if not r.get(k, "").strip()]
        if missing:
            raise ValueError(f"{r.get('id', '?')}: missing {missing}")
        for k in ("L", "I", "L_res", "I_res"):
            r[k] = int(r[k])
            if not 1 <= r[k] <= 5:
                raise ValueError(f"{r['id']}: {k} must be 1-5")
        r["raw"], r["residual"] = r["L"] * r["I"], r["L_res"] * r["I_res"]
        r["band_raw"], r["band_res"] = band(r["raw"]), band(r["residual"])
        r["reduction"] = 1 - r["residual"] / r["raw"]
        out.append(r)
    return out


def priority_order(rows: list) -> list:
    """Rank by residual risk first: what is still dangerous after the controls."""
    return sorted(rows, key=lambda r: (-r["residual"], -r["raw"]))


def heatmap(rows: list, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    cols = {"Low": "#d9ead3", "Medium": "#fff2cc", "High": "#fce5cd", "Critical": "#f4cccc"}
    for l in range(1, 6):
        for i in range(1, 6):
            ax.add_patch(plt.Rectangle((i - 0.5, l - 0.5), 1, 1, color=cols[band(l * i)], ec="white"))
    for r in rows:
        ax.annotate("", xy=(r["I_res"], r["L_res"]), xytext=(r["I"], r["L"]),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1))
        ax.text(r["I"], r["L"], r["id"].replace("FL-", ""), ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="#333"))
        ax.text(r["I_res"], r["L_res"], r["id"].replace("FL-", ""), ha="center", va="center", fontsize=8,
                color="white", bbox=dict(boxstyle="circle,pad=0.2", fc="#333", ec="#333"))
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)
    ax.set_xlabel("Impact")
    ax.set_ylabel("Likelihood")
    ax.set_title("Risk register: raw (white) to residual (dark)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
