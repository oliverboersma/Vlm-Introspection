#!/usr/bin/env python3
"""Summarize a sweep directory into ``summary.md`` and ``sweep.png``.

    python scripts/analyze.py results/pilot
"""

import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

from vlm_introspection.analysis import (baseline_table, deltas, fmt_table, freeform_rates, load, plot_sweep,
                                        specificity_table)


def _get(rows, **kw):
    for r in rows:
        if all(r[k] == v for k, v in kw.items()):
            return r
    return None


def main(d):
    d = Path(d)
    recs = load(d / "sweep.jsonl")
    dl = deltas(recs)
    rows = specificity_table(dl)
    out = [f"# Sweep summary: `{d}`", "", f"{len(recs)} probe records.", ""]

    out += ["## Uninjected reference points (raw log-odds yes − no)", "",
            fmt_table(baseline_table(recs), ["condition", "probe", "on_target", "mean", "se", "n"]), ""]

    real = {p: _get(rows, condition="real", probe=p) for p in ("see", "think")}
    out += ["## Real images of the concept (Δ vs neutral image)", "",
            fmt_table([real["see"], real["think"]], ["probe", "on", "off", "spec", "spec_se", "n"]), ""]

    layers = sorted({r["layer"] for r in rows if r["condition"] == "concept"})
    alphas = sorted({r["alpha"] for r in rows if r["condition"] == "concept"})
    for a in alphas:
        table = []
        for L in layers:
            row = {"layer": L}
            for cond, site, tag in (("concept", "image", "img"), ("random", "image", "img_rand"),
                                    ("concept", "text", "txt"), ("random", "text", "txt_rand")):
                for p in ("see", "think"):
                    r = _get(rows, condition=cond, site=site, layer=L, alpha=a, probe=p)
                    row[f"{tag}_{p}"] = r["spec"] if r else float("nan")
            for cond, tag in (("text_only_concept", "txtonly"), ("text_only_random", "txtonly_rand")):
                r = _get(rows, condition=cond, layer=L, alpha=a, probe="think")
                row[f"{tag}_think"] = r["spec"] if r else float("nan")
            table.append(row)
        cols = ["layer", "img_see", "img_think", "img_rand_see", "img_rand_think", "txt_see", "txt_think",
                "txt_rand_see", "txt_rand_think", "txtonly_think", "txtonly_rand_think"]
        out += [f"## Specificity (on − off-target Δ log-odds), α = {a:g}", "", fmt_table(table, cols), ""]

    # Best image-site setting and its per-concept breakdown.
    cand = [r for r in rows if r["condition"] == "concept" and r["site"] == "image" and r["probe"] == "see"]
    best = max(cand, key=lambda r: r["spec"])
    L, a = best["layer"], best["alpha"]
    out += [f"## Per concept at the best image-site setting (layer {L}, α = {a:g})", ""]
    per = defaultdict(lambda: defaultdict(list))
    for r in dl:
        if (r["condition"], r["site"], r["layer"], r["alpha"]) == ("concept", "image", L, a):
            per[r["concept"]][(r["probe"], r["on_target"])].append(r["delta"])
    prow = []
    for c, g in per.items():
        prow.append(dict(concept=c, see_on=mean(g[("see", True)]), see_off=mean(g[("see", False)]),
                         think_on=mean(g[("think", True)]), think_off=mean(g[("think", False)])))
    out += [fmt_table(prow, ["concept", "see_on", "see_off", "think_on", "think_off"]), ""]

    ff = d / "freeform.jsonl"
    if ff.exists():
        out += ["## Free text: rate at which the injected concept is named", "",
                fmt_table(freeform_rates(ff), ["condition", "site", "layer", "alpha", "question", "rate", "n"]), ""]

    (d / "summary.md").write_text("\n".join(out))
    plot_sweep(rows, d / "sweep.png", title="Concept injection while viewing a neutral image (SmolVLM-500M)")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
