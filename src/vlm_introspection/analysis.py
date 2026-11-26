"""Aggregate sweep results into effect sizes and plots.

Core quantity: Δ = logodds(condition) − logodds(none), matched on
(concept, probe, probe_concept, target image). Positive Δ on SEE means the
model became more willing to say the concept is *in the image*; positive Δ on
THINK means more willing to say it is *on its mind*.

* on-target Δ        probe_concept == injected concept
* specificity        on-target Δ − mean off-target Δ (same family)
* seen-vs-thought    specificity(SEE) − specificity(THINK)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def load(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _key(r):
    return (r["concept"], r["probe"], r["probe_concept"], r["target"])


def deltas(records: list[dict]) -> list[dict]:
    """Attach ``delta`` (vs the matched no-injection record) to every injected record."""
    base = {}
    for r in records:
        if r["condition"] in ("none", "text_only_none"):
            base[(r["condition"].startswith("text_only"),) + _key(r)] = r["logodds"]
    out = []
    for r in records:
        if r["condition"] in ("none", "text_only_none"):
            continue
        b = base[(r["condition"].startswith("text_only"),) + _key(r)]
        out.append({**r, "delta": r["logodds"] - b})
    return out


def _se(xs):
    return stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else 0.0


def specificity_table(dl: list[dict]) -> list[dict]:
    """One row per (condition, site, layer, alpha, probe): on/off-target Δ and specificity.

    Specificity is computed per (concept, target) and then averaged over them,
    so its standard error reflects concept × image variability.
    """
    groups = defaultdict(lambda: defaultdict(lambda: {"on": [], "off": []}))
    for r in dl:
        g = (r["condition"], r["site"], r["layer"], r["alpha"], r["probe"])
        groups[g][(r["concept"], r["target"])]["on" if r["on_target"] else "off"].append(r["delta"])
    rows = []
    for (cond, site, L, a, probe), units in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        on = [mean(u["on"]) for u in units.values() if u["on"]]
        off = [mean(u["off"]) for u in units.values() if u["off"]]
        spec = [mean(u["on"]) - mean(u["off"]) for u in units.values() if u["on"] and u["off"]]
        rows.append(dict(condition=cond, site=site, layer=L, alpha=a, probe=probe, n=len(on),
                         on=mean(on), on_se=_se(on), off=mean(off) if off else float("nan"),
                         spec=mean(spec) if spec else float("nan"), spec_se=_se(spec) if spec else 0.0))
    return rows


def baseline_table(records: list[dict]) -> list[dict]:
    """Raw log-odds for the uninjected conditions (false-positive floor and real-image ceiling)."""
    groups = defaultdict(list)
    for r in records:
        if r["condition"] in ("none", "real", "text_only_none"):
            groups[(r["condition"], r["probe"], r["on_target"])].append(r["logodds"])
    return [dict(condition=c, probe=p, on_target=o, mean=mean(v), se=_se(v), n=len(v))
            for (c, p, o), v in sorted(groups.items())]


def freeform_rates(path) -> list[dict]:
    groups = defaultdict(list)
    for r in load(path):
        groups[(r["condition"], r["site"], r["layer"], r["alpha"], r["question"])].append(r["hit"])
    return [dict(condition=c, site=s, layer=L, alpha=a, question=q, rate=mean(h), n=len(h))
            for (c, s, L, a, q), h in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0])))]


# ------------------------------------------------------------------- plotting
def plot_sweep(rows: list[dict], out_png, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alphas = sorted({r["alpha"] for r in rows if r["condition"] == "concept"})
    panels = [("concept", "image"), ("concept", "text"), ("text_only_concept", "text")]
    titles = {("concept", "image"): "Injected at image tokens",
              ("concept", "text"): "Injected at text tokens",
              ("text_only_concept", "text"): "Text-only prompt (no image)"}
    fig, axes = plt.subplots(len(alphas), len(panels), figsize=(4.2 * len(panels), 2.6 * len(alphas)),
                             sharex=True, squeeze=False)
    colors = {"see": "#2b6cb0", "think": "#c05621"}
    for i, a in enumerate(alphas):
        for j, (cond, site) in enumerate(panels):
            ax = axes[i][j]
            ax.axhline(0, color="#999", lw=0.8)
            for probe in ("see", "think"):
                for c, ls, label in ((cond, "-", "concept"), (cond.replace("concept", "random"), ":", "random")):
                    pts = sorted((r["layer"], r["spec"], r["spec_se"]) for r in rows
                                 if r["condition"] == c and r["site"] == site and r["alpha"] == a
                                 and r["probe"] == probe)
                    if not pts:
                        continue
                    x, y, e = zip(*pts)
                    ax.errorbar(x, y, yerr=e, color=colors[probe], ls=ls, marker="o" if ls == "-" else None,
                                ms=3, lw=1.4 if ls == "-" else 1.0, capsize=2,
                                label=f"{probe.upper()} · {label}" if (i, j) == (0, 0) else None)
            if i == 0:
                ax.set_title(titles[(cond, site)], fontsize=10)
            if j == 0:
                ax.set_ylabel(f"α={a:g}\nspecificity Δ log-odds", fontsize=9)
            if i == len(alphas) - 1:
                ax.set_xlabel("layer")
    fig.legend(loc="upper center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.0))
    if title:
        fig.suptitle(title, y=1.03, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_headline(rows: list[dict], out_png, alpha: float = 1.0, title: str = "") -> None:
    """Image site vs text site on one shared scale, with real-image reference lines."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"see": "#2b6cb0", "think": "#c05621"}
    real = {p: next((r for r in rows if r["condition"] == "real" and r["probe"] == p), None) for p in colors}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    for ax, site, name in ((axes[0], "image", "Vector added at image tokens"),
                           (axes[1], "text", "Vector added at text tokens")):
        ax.axhline(0, color="#999", lw=0.8)
        for probe, col in colors.items():
            if real[probe]:
                ax.axhline(real[probe]["spec"], color=col, ls="--", lw=1, alpha=0.6)
            for cond, ls, mk in (("concept", "-", "o"), ("random", ":", None)):
                pts = sorted((r["layer"], r["spec"], r["spec_se"]) for r in rows if r["condition"] == cond
                             and r["site"] == site and r["alpha"] == alpha and r["probe"] == probe)
                if pts:
                    x, y, e = zip(*pts)
                    ax.errorbar(x, y, yerr=e, color=col, ls=ls, marker=mk, ms=4, capsize=2, lw=1.6 if mk else 1,
                                label=f"{probe.upper()} probe, {cond} vector" if site == "image" else None)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("injection layer")
    axes[0].set_ylabel("specificity (Δ log-odds,\non-target − off-target)")
    if real["see"]:
        axes[1].text(axes[1].get_xlim()[1], real["see"]["spec"], " real image, SEE", color=colors["see"],
                     va="center", fontsize=8)
        axes[1].text(axes[1].get_xlim()[1], real["think"]["spec"], " real image, THINK", color=colors["think"],
                     va="center", fontsize=8)
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.88))
    fig.suptitle(title or f"Concept injection while viewing a neutral image (α = {alpha:g})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fmt_table(rows: list[dict], cols: list[str]) -> str:
    def f(v):
        return f"{v:.2f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(f(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)
