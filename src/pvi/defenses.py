"""Server-side aggregation rules and update inspection.

The central tension of the audit lives here. Every rule below except plain clipping needs to
*see individual updates*. Secure aggregation removes exactly that visibility, so under secagg the
server can only run ``mean`` (optionally with client-side clipping that is enforced or not).
"""

from __future__ import annotations

import torch

INSPECTING = {"median", "trimmed_mean", "norm_filter", "cosine_filter", "krum_lite"}
BLIND_OK = {"mean"}


def clip_update(u: torch.Tensor, c: float) -> torch.Tensor:
    n = u.norm()
    return u * min(1.0, c / (float(n) + 1e-12))


def aggregate(updates: list, rule: str = "mean", **kw) -> tuple:
    """Return (aggregate_update, info). ``info['flagged']`` lists indices the rule discarded."""
    U = torch.stack(updates)
    k = len(updates)
    info = {"flagged": []}
    if rule == "mean":
        return U.mean(0), info
    if rule == "median":
        return U.median(0).values, info
    if rule == "trimmed_mean":
        trim = max(1, int(kw.get("trim_frac", 0.2) * k))
        if 2 * trim >= k:
            return U.mean(0), info
        s = U.sort(0).values
        return s[trim : k - trim].mean(0), info
    if rule == "norm_filter":
        norms = U.norm(dim=1)
        thr = kw.get("mult", 2.0) * norms.median()
        keep = norms <= thr
        info["flagged"] = [i for i in range(k) if not keep[i]]
        return (U[keep].mean(0) if keep.any() else U.mean(0)), info
    if rule == "cosine_filter":
        mean_dir = torch.nn.functional.normalize(U.median(0).values, dim=0)
        cos = torch.nn.functional.normalize(U, dim=1) @ mean_dir
        thr = kw.get("cos_thr", 0.0)
        keep = cos > thr
        info["flagged"] = [i for i in range(k) if not keep[i]]
        info["cos"] = cos
        return (U[keep].mean(0) if keep.any() else U.mean(0)), info
    if rule == "krum_lite":
        d = torch.cdist(U, U)
        m = max(1, k - int(kw.get("f", 2)) - 2)
        scores = d.sort(1).values[:, 1 : m + 1].sum(1)
        keep_n = max(1, k - int(kw.get("f", 2)))
        keep_idx = scores.argsort()[:keep_n]
        mask = torch.ones(k, dtype=torch.bool)
        mask[keep_idx] = False
        info["flagged"] = [i for i in range(k) if mask[i]]
        return U[keep_idx].mean(0), info
    raise ValueError(f"unknown rule {rule}")


def detection_scores(updates: list) -> dict:
    """Per-update anomaly signals an inspecting server could compute (norm, cosine to median)."""
    U = torch.stack(updates)
    med = U.median(0).values
    cos = torch.nn.functional.normalize(U, dim=1) @ torch.nn.functional.normalize(med, dim=0)
    return {"norm": U.norm(dim=1), "neg_cos": -cos}
