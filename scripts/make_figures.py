"""Turn probe CSVs into figures. Safe to run on partial results."""
import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pvi.metrics import mean_ci  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", default="results/preliminary")
ap.add_argument("--out", default="results/preliminary")
a = ap.parse_args()


def load(name):
    p = os.path.join(a.inp, name)
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def group(rows, keys, val):
    g = defaultdict(list)
    for r in rows:
        try:
            g[tuple(r[k] for k in keys)].append(float(r[val]))
        except (ValueError, KeyError):
            pass
    return g


rows = load("backdoor_sweep.csv")
if rows:
    attacks = [x for x in dict.fromkeys(r["attack"] for r in rows) if x != "none"]
    defs = list(dict.fromkeys(r["defense"] for r in rows))
    fig, axes = plt.subplots(1, len(attacks), figsize=(4.2 * len(attacks), 3.8), sharey=True, squeeze=False)
    for ax, atk in zip(axes[0], attacks):
        g = group([r for r in rows if r["attack"] == atk], ["defense"], "asr")
        xs = [d for d in defs if (d,) in g]
        stats = [mean_ci(g[(d,)]) for d in xs]
        ax.bar(range(len(xs)), [s[0] for s in stats], yerr=[s[1] for s in stats], color="#4C78A8", capsize=2)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, rotation=60, ha="right", fontsize=7)
        ax.set_title(atk, fontsize=9)
        ax.set_ylim(0, 1.05)
    axes[0][0].set_ylabel("Backdoor success rate")
    fig.suptitle("Backdoor success by control (mean, 95% CI over seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_backdoor_asr.png"), dpi=160)
    plt.close(fig)

rows = load("canary_exposure.csv")
if rows:
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for rep in sorted({int(r["repeats"]) for r in rows}):
        zs = sorted({float(r["noise_multiplier"]) for r in rows})
        g = group([r for r in rows if int(r["repeats"]) == rep], ["noise_multiplier"], "exposure")
        ys = [mean_ci(g.get((str(z),), []))[0] for z in zs]
        ax.plot(range(len(zs)), ys, marker="o", label=f"{rep} copies")
    ax.set_xticks(range(len(zs)))
    ax.set_xticklabels([str(z) for z in zs])
    ax.set_xlabel("DP noise multiplier z")
    ax.set_ylabel("Canary exposure (bits)")
    ax.legend(fontsize=8)
    ax.set_title("Memorization vs DP noise", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_canary_exposure.png"), dpi=160)
    plt.close(fig)

rows = load("tension.csv")
if rows:
    det = [r for r in rows if r["probe"] == "detect"]
    mem = [r for r in rows if r["probe"] == "membership"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    if det:
        combos = list(dict.fromkeys((r["setting"], r["attack"]) for r in det))
        for off, key, col in [(-0.2, "auc_norm", "#4C78A8"), (0.2, "auc_cosine", "#F58518")]:
            vals = [mean_ci([float(r[key]) for r in det if (r["setting"], r["attack"]) == c])[0] for c in combos]
            axes[0].bar([i + off for i in range(len(combos))], vals, width=0.4, label=key.replace("auc_", ""), color=col)
        axes[0].set_xticks(range(len(combos)))
        axes[0].set_xticklabels([f"{c[1].replace('_', chr(10))}\n({c[0]})" for c in combos], fontsize=6)
        axes[0].axhline(0.5, ls="--", c="gray", lw=0.8)
        axes[0].set_ylim(0, 1.05)
        axes[0].set_ylabel("AUC")
        axes[0].set_title("Integrity: can a server that sees updates spot attackers?", fontsize=8)
        axes[0].legend(fontsize=7)
    if mem:
        reps = sorted({int(r["canary_repeats"]) for r in mem})
        keys = [("auc_single_update", "single visible update", "#E45756"),
                ("auc_aggregate", "aggregate (secagg)", "#54A24B"),
                ("auc_aggregate_dp_z1", "aggregate + DP z=1", "#4C78A8")]
        for j, (k, lab, col) in enumerate(keys):
            vals = [mean_ci([float(r[k]) for r in mem if int(r["canary_repeats"]) == rep])[0] for rep in reps]
            axes[1].bar([i + (j - 1) * 0.27 for i in range(len(reps))], vals, width=0.27, label=lab, color=col)
        axes[1].set_xticks(range(len(reps)))
        axes[1].set_xticklabels([f"{r} canary copies" for r in reps], fontsize=7)
        axes[1].axhline(0.5, ls="--", c="gray", lw=0.8)
        axes[1].set_ylim(0, 1.05)
        axes[1].set_title("Privacy: membership-inference AUC", fontsize=8)
        axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_tension.png"), dpi=160)
    plt.close(fig)
rows = load("backdoor_dose.csv")
if rows:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)
    for ax, d in zip(axes, ["clip", "clip_dp"]):
        for atk, col in [("trigger_boost", "#E45756"), ("edgecase_normmatch", "#4C78A8")]:
            ks = sorted({int(r["n_malicious"]) for r in rows})
            ys, cs = [], []
            for k in ks:
                v = [float(r["asr"]) for r in rows if r["defense"] == d and r["attack"] == atk and int(r["n_malicious"]) == k]
                m, c = mean_ci(v)
                ys.append(m)
                cs.append(c)
            ax.errorbar(ks, ys, yerr=cs, marker="o", capsize=2, label=atk, color=col)
        ax.set_title(f"control: {d}", fontsize=9)
        ax.set_xlabel("malicious clients (of 30)")
        ax.set_xticks(ks)
        ax.set_ylim(-0.02, 1.05)
    axes[0].set_ylabel("Backdoor success rate")
    axes[0].legend(fontsize=7)
    fig.suptitle("Dose-response (k=0 is the clean baseline; seed 0 trigger baseline is inflated, see report)", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_dose_response.png"), dpi=160)
    plt.close(fig)
print("figures written to", a.out)
