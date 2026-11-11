"""Summarize probe CSVs into baseline-adjusted tables (CSV + Markdown) for the report."""
import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pvi.metrics import mean_ci  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results/preliminary")
a = ap.parse_args()


def load(name):
    p = os.path.join(a.dir, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


md = []
rows = load("backdoor_sweep.csv")
if rows:
    metric = lambda atk: "asr_trigger" if atk.startswith("trigger") else "asr_edge"
    clean = defaultdict(dict)  # seed -> defense -> {trigger, edge}
    for r in rows:
        if r["attack"] == "none":
            clean[r["seed"]][r["defense"]] = {"asr_trigger": f(r["asr_trigger"]), "asr_edge": f(r["asr_edge"]), "acc": f(r["acc"])}

    def baseline(seed, defense, key):
        d = clean[seed]
        if defense in d:
            return d[defense][key]
        vals = [v[key] for v in d.values()]
        return sum(vals) / len(vals)

    out, groups = [], defaultdict(list)
    for r in rows:
        if r["attack"] == "none":
            continue
        k = metric(r["attack"])
        base = baseline(r["seed"], r["defense"], k)
        groups[(r["attack"], r["defense"])].append((r, f(r[k]), max(0.0, f(r[k]) - base), baseline(r["seed"], r["defense"], "acc")))
    for (atk, d), items in groups.items():
        asr_m, asr_ci = mean_ci([i[1] for i in items])
        ex_m, ex_ci = mean_ci([i[2] for i in items])
        acc_m, _ = mean_ci([f(i[0]["acc"]) for i in items])
        clean_acc = mean_ci([i[3] for i in items])[0]
        rec = mean_ci([f(i[0]["recall"]) for i in items])[0]
        fpr = mean_ci([f(i[0]["fpr"]) for i in items])[0]
        eps = f(items[0][0]["epsilon"])
        out.append(dict(attack=atk, defense=d, n_seeds=len(items), asr=asr_m, asr_ci=asr_ci, excess_asr=ex_m,
                        excess_ci=ex_ci, acc=acc_m, clean_acc=clean_acc, recall=rec, fpr=fpr, epsilon=eps))
    with open(os.path.join(a.dir, "summary_backdoor.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    md.append("### Backdoor sweep (3 seeds; excess ASR = ASR minus the clean run's rate for the same seed and defense)\n")
    md.append("| Attack | Control | Excess ASR (95% CI) | Main acc | Detect recall | FPR |")
    md.append("|---|---|---|---|---|---|")
    for o in out:
        rec = "n/a" if o["recall"] != o["recall"] else f"{o['recall']:.2f}"
        fp = "n/a" if o["fpr"] != o["fpr"] else f"{o['fpr']:.2f}"
        md.append(f"| {o['attack']} | {o['defense']} | {o['excess_asr']:.2f} ({o['excess_ci']:.2f}) | {o['acc']:.3f} | {rec} | {fp} |")

rows = load("canary_exposure.csv")
if rows:
    g = defaultdict(list)
    for r in rows:
        g[(int(r["repeats"]), float(r["noise_multiplier"]))].append(r)
    md.append("\n### Canary exposure (bits; max about 9.97 for 1000 candidates)\n")
    md.append("| Copies | z | eps at delta=1e-10 (conservative) | Exposure (95% CI) | Main acc |")
    md.append("|---|---|---|---|---|")
    for (rep, z), rs in sorted(g.items()):
        e, ci = mean_ci([f(r["exposure"]) for r in rs])
        acc = mean_ci([f(r["acc"]) for r in rs])[0]
        eps = f(rs[0]["epsilon"])
        md.append(f"| {rep} | {z} | {'inf' if eps == float('inf') else f'{eps:.0f}'} | {e:.2f} ({ci:.2f}) | {acc:.3f} |")

rows = load("tension.csv")
if rows:
    det = [r for r in rows if r["probe"] == "detect"]
    mem = [r for r in rows if r["probe"] == "membership"]
    md.append("\n### Detection AUC when the server can see updates\n")
    md.append("| Setting | Attack | Norm AUC | Cosine AUC |")
    md.append("|---|---|---|---|")
    for key in dict.fromkeys((r["setting"], r["attack"]) for r in det):
        rs = [r for r in det if (r["setting"], r["attack"]) == key]
        md.append(f"| {key[0]} | {key[1]} | {mean_ci([f(r['auc_norm']) for r in rs])[0]:.2f} | {mean_ci([f(r['auc_cosine']) for r in rs])[0]:.2f} |")
    md.append("\n### Membership-inference AUC for the canary client\n")
    md.append("| Canary copies | Single visible update | Aggregate (secagg) | Aggregate + DP (z=1) |")
    md.append("|---|---|---|---|")
    for rep in sorted({int(r["canary_repeats"]) for r in mem}):
        rs = [r for r in mem if int(r["canary_repeats"]) == rep]
        vals = [mean_ci([f(r[k]) for r in rs]) for k in ("auc_single_update", "auc_aggregate", "auc_aggregate_dp_z1")]
        md.append(f"| {rep} | " + " | ".join(f"{m:.2f} ({c:.2f})" for m, c in vals) + " |")

rows = load("backdoor_dose.csv")
if rows:
    g = defaultdict(list)
    for r in rows:
        g[(r["defense"], r["attack"], int(r["n_malicious"]))].append(r)
    md.append("\n### Dose-response: ASR versus number of malicious clients (population 30)\n")
    md.append("| Control | Attack | k=0 (clean) | k=1 | k=2 | k=3 |")
    md.append("|---|---|---|---|---|---|")
    for d, atk in dict.fromkeys((k[0], k[1]) for k in g):
        cells = []
        for k in (0, 1, 2, 3):
            rs = g.get((d, atk, k), [])
            m, ci = mean_ci([f(r["asr"]) for r in rs])
            cells.append(f"{m:.2f} ({ci:.2f})" if rs else "-")
        md.append(f"| {d} | {atk} | " + " | ".join(cells) + " |")

open(os.path.join(a.dir, "summary.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
