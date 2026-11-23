#!/usr/bin/env python3
"""Figures and LaTeX tables for experiments E1-E9.

    python make_figures.py                     # reads results/, writes figures/
    python make_figures.py --results results_quick --out figures_quick

Every figure is written as vector PDF (for the paper) and 300-dpi PNG. Where
three variables are involved there are two versions: a 2-D heatmap
(recommended for print: every value is readable) and a 3-D surface
(`*_3d.pdf`, better suited to slides and posters).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Style: validated categorical palette (fixed order), one-hue sequential ramp,
# hairline solid grid, sans-serif text, no in-figure titles (captions do that).
# ---------------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE_RAMP = ["#f4f8fe", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#ec9a92", "#c22f2f"])
SINGLE, DOUBLE = 3.45, 7.1          # JISA column / text widths (inches)

DEF_STYLE = {   # defense -> (label, color, marker, linestyle) ; color follows the entity everywhere
    "fedavg": ("FedAvg", SERIES[1], "o", "-"),
    "median": ("Median", SERIES[3], "s", "-"),
    "trimmed": ("Trimmed mean", SERIES[4], "D", "-"),
    "krum": ("Multi-Krum", SERIES[6], "^", "-"),
    "fltrust": ("FLTrust", SERIES[2], "v", "-"),
    "foolsgold": ("FoolsGold", SERIES[5], "P", "-"),
    "sentinel": ("Sentinel-Flow", SERIES[0], "o", "-"),
    "dual": ("Dual attestation", SERIES[7], "*", "-"),
}


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.linewidth": 1.5, "lines.markersize": 4,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 150,
    })


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    plt.close(fig)
    print("  wrote", name)


def load(results: Path, exp: str) -> list[dict]:
    d = results / exp
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.exists() else []


def ci95(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    from scipy import stats
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def text_color(v, vmin, vmax):
    return "#ffffff" if (v - vmin) / max(vmax - vmin, 1e-9) > 0.55 else INK


def heatmap(ax, M, xt, yt, vmin=0, vmax=100, cmap=SEQ, fmt="{:.0f}", annotate=True, fs=6):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_xticks(range(len(xt))); ax.set_xticklabels(xt)
    ax.set_yticks(range(len(yt))); ax.set_yticklabels(yt)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    if annotate:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if np.isfinite(v):
                    ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fs,
                            color=text_color(v, vmin, vmax))
    return im


def bars3d(ax, M, xt, yt, zlabel, xlabel, ylabel, zlim=(0, 100), elev=28, azim=-135, zlab=True):
    """3-D bar chart of a small grid; low values in front, tall bars at the back."""
    xs, ys = np.meshgrid(np.arange(M.shape[1]), np.arange(M.shape[0]))
    x, y, z = xs.ravel(), ys.ravel(), np.nan_to_num(M.ravel())
    cols = SEQ(np.clip((z - zlim[0]) / (zlim[1] - zlim[0]), 0, 1))
    ax.bar3d(x - 0.35, y - 0.35, np.full_like(z, zlim[0]), 0.7, 0.7, z - zlim[0], color=cols,
             edgecolor="white", linewidth=0.3, shade=True)
    ax.set_xticks(range(len(xt))); ax.set_xticklabels(xt)
    ax.set_yticks(range(len(yt))); ax.set_yticklabels(yt)
    ax.set_zlim(*zlim)
    ax.set_xlabel(xlabel, labelpad=0); ax.set_ylabel(ylabel, labelpad=0)
    if zlab:
        ax.set_zlabel(zlabel, labelpad=0)
    ax.view_init(elev=elev, azim=azim)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((1, 1, 1, 0)); a.pane.set_edgecolor(GRID)
        a._axinfo["grid"].update(color=GRID, linewidth=0.4)
    ax.tick_params(pad=0, labelsize=6)


def outline(ax, mask, color=INK, lw=1.1):
    """Draw the outer border of the cells where mask is True (no line crosses a value)."""
    n, m = mask.shape
    for i in range(n):
        for j in range(m):
            if not mask[i, j]:
                continue
            if i == 0 or not mask[i - 1, j]:
                ax.plot([j - .5, j + .5], [i - .5, i - .5], color=color, lw=lw, solid_capstyle="projecting")
            if i == n - 1 or not mask[i + 1, j]:
                ax.plot([j - .5, j + .5], [i + .5, i + .5], color=color, lw=lw, solid_capstyle="projecting")
            if j == 0 or not mask[i, j - 1]:
                ax.plot([j - .5, j - .5], [i - .5, i + .5], color=color, lw=lw, solid_capstyle="projecting")
            if j == m - 1 or not mask[i, j + 1]:
                ax.plot([j + .5, j + .5], [i - .5, i + .5], color=color, lw=lw, solid_capstyle="projecting")


def ber_for_frr(frr, n_bits=256, tau=0.9):
    """Largest bit-error rate whose binomial FRR at tau stays below frr."""
    from scipy import stats
    k = int(np.floor(n_bits * (1 - tau) + 1e-9))
    lo, hi = 0.0, 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        if stats.binom.sf(k, n_bits, mid) <= frr:
            lo = mid
        else:
            hi = mid
    return lo


def surface3d(ax, X, Y, Z, zlabel, xlabel, ylabel, xticks=None, yticks=None, zlim=(0, 100), elev=24, azim=-128):
    ax.plot_surface(X, Y, Z, cmap=SEQ, vmin=zlim[0], vmax=zlim[1], edgecolor="#ffffff", linewidth=0.3,
                    antialiased=True, alpha=0.96, rstride=1, cstride=1)
    ax.contourf(X, Y, Z, zdir="z", offset=zlim[0], cmap=SEQ, vmin=zlim[0], vmax=zlim[1], alpha=0.55,
                levels=np.linspace(zlim[0], zlim[1], 11))
    ax.set_zlim(*zlim)
    ax.set_xlabel(xlabel, labelpad=2); ax.set_ylabel(ylabel, labelpad=2); ax.set_zlabel(zlabel, labelpad=2)
    if xticks is not None:
        ax.set_xticks(xticks[0]); ax.set_xticklabels(xticks[1])
    if yticks is not None:
        ax.set_yticks(yticks[0]); ax.set_yticklabels(yticks[1])
    ax.view_init(elev=elev, azim=azim)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((1, 1, 1, 0)); a.pane.set_edgecolor(GRID)
        a._axinfo["grid"].update(color=GRID, linewidth=0.4)
    ax.tick_params(pad=0, labelsize=6)


def tag_of(r):
    return r["config"]["tag"]


# ---------------------------------------------------------------------------
# E1: replication over seeds
# ---------------------------------------------------------------------------
SCEN_LABEL = {"S1_baseline": "1  Baseline (no attack)", "S2_attack": "2  Poisoning, no defense",
              "S3_sentinel": "3  Sentinel-Flow", "S4_dual": "4  Sentinel-Flow + PUF",
              "S5A_sybil_sentinel": "5A Smart Sybil vs. Sentinel-Flow", "S5B_sybil_dual": "5B Smart Sybil vs. dual"}


def e1(results, out, tables):
    R = load(results, "e1")
    if not R:
        return
    groups = defaultdict(list)
    for r in R:
        proto, scen = tag_of(r).split("_", 1)
        groups[(proto, scen)].append(r)
    scen_order = [s for s in SCEN_LABEL if any(k[1] == s for k in groups)]

    # --- dot plot of final accuracy / ASR with 95% CI, both protocols ---
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.2), sharey=True)
    for ax, metric, lab in zip(axes, ["acc", "asr"], ["Global accuracy (%)", "Attack success rate (%)"]):
        for k, (proto, col, mk, off) in enumerate([("orig", SERIES[0], "o", 0.14), ("fixed", SERIES[1], "s", -0.14)]):
            ys, xs, es = [], [], []
            for i, s in enumerate(scen_order[::-1]):
                vals = [r["final"][metric] for r in groups.get((proto, s), [])]
                if vals:
                    ys.append(i + off); xs.append(np.mean(vals)); es.append(ci95(vals))
            ax.errorbar(xs, ys, xerr=es, fmt=mk, color=col, ms=4, capsize=2, lw=1,
                        label={"orig": "Original protocol", "fixed": "Corrected protocol"}[proto])
        ax.set_xlabel(lab)
        ax.set_yticks(range(len(scen_order))); ax.set_yticklabels([SCEN_LABEL[s] for s in scen_order[::-1]])
        ax.grid(axis="y", visible=False)
    axes[0].set_xlim(45, 101); axes[1].set_xlim(-3, 103)
    axes[1].legend(loc="center right")
    save(fig, out, "e1_final_ci")

    # --- ASR over rounds, corrected protocol, mean and 95% CI band ---
    fig, ax = plt.subplots(figsize=(SINGLE, 2.5))
    show = [("S2_attack", SERIES[1], "-", "2  Poisoning, no defense"),
            ("S5A_sybil_sentinel", SERIES[7], (0, (4, 2)), "5A Smart Sybil vs. Sentinel-Flow"),
            ("S5B_sybil_dual", SERIES[6], "-", "5B Smart Sybil vs. dual attestation"),
            ("S3_sentinel", SERIES[0], "-", "3  Sentinel-Flow"),
            ("S1_baseline", MUTED, (0, (1, 1.5)), "1  Baseline, no attack")]
    for s_, col, ls, lab in show:
        runs = groups.get(("fixed", s_), [])
        if not runs:
            continue
        H = np.array([r["history"]["asr"] for r in runs])
        x = runs[0]["history"]["round"]
        m = H.mean(0); e = np.array([ci95(H[:, j]) for j in range(H.shape[1])])
        ax.fill_between(x, np.clip(m - e, 0, 100), np.clip(m + e, 0, 100), color=col, alpha=0.18, lw=0)
        ax.plot(x, m, color=col, lw=1.5, ls=ls, label=lab)
    ax.set_xlabel("Communication round"); ax.set_ylabel("Attack success rate (%)")
    ax.set_ylim(-2, 102); ax.set_xlim(1, x[-1])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=6, handlelength=2.4)
    save(fig, out, "e1_asr_rounds")

    # --- LaTeX table ---
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r" & \multicolumn{2}{c}{Original protocol} & \multicolumn{2}{c}{Corrected protocol} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Scenario & Accuracy (\%) & ASR (\%) & Accuracy (\%) & ASR (\%) \\", r"\midrule"]
    for s in scen_order:
        row = [SCEN_LABEL[s].replace("  ", " ")]
        for proto in ("orig", "fixed"):
            for metric in ("acc", "asr"):
                v = [r["final"][metric] for r in groups.get((proto, s), [])]
                row.append(f"{np.mean(v):.2f} $\\pm$ {np.std(v, ddof=1) if len(v) > 1 else 0:.2f}" if v else "--")
        lines.append(" & ".join(row) + r" \\")
    n = len(groups.get(("fixed", "S1_baseline"), []))
    lines += [r"\bottomrule", r"\end{tabular}"]
    tables["e1_table.tex"] = f"% mean +/- sd over {n} seeds\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# E2: attack landscape
# ---------------------------------------------------------------------------
def e2(results, out, tables):
    R = load(results, "e2")
    if not R:
        return
    cell = defaultdict(list)
    ref = defaultdict(list)
    for r in R:
        c = r["config"]
        if c["attack"] == "none":
            ref[c["defense"]].append(r["final"]["acc"])
            continue
        cell[(c["defense"], c["n_malicious"], c["alpha"])].append(r["final"]["asr"])
    defenses = [d for d in ["fedavg", "median", "trimmed", "krum", "fltrust", "sentinel"] if any(k[0] == d for k in cell)]
    mals = sorted({k[1] for k in cell}); alphas = sorted({k[2] for k in cell})
    n_total = 20
    xt = [f"{a:g}" for a in alphas]; yt = [f"{100 * m / n_total:.0f}" for m in mals]
    mats = {}
    for d in defenses:
        M = np.full((len(mals), len(alphas)), np.nan)
        for i, m in enumerate(mals):
            for j, a in enumerate(alphas):
                v = cell.get((d, m, a))
                if v:
                    M[i, j] = np.mean(v)
        mats[d] = M
    ncol = 3; nrow = math.ceil(len(defenses) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(DOUBLE, 1.75 * nrow + 0.3), squeeze=False, constrained_layout=True)
    for k, (ax, d) in enumerate(zip(axes.flat, defenses)):
        im = heatmap(ax, mats[d], xt, yt)
        acc0 = np.mean(ref[d]) if ref.get(d) else np.nan
        ax.set_title(f"{DEF_STYLE[d][0]}   clean acc. {acc0:.1f}%", fontsize=7, color=INK, loc="left", pad=3)
        if k // ncol == nrow - 1:
            ax.set_xlabel("Boosting factor $\\alpha$")
        if k % ncol == 0:
            ax.set_ylabel("Malicious clients (%)")
    for ax in list(axes.flat)[len(defenses):]:
        ax.axis("off")
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01, aspect=30)
    cb.set_label("Attack success rate (%)"); cb.outline.set_visible(False)
    save(fig, out, "e2_landscape_heatmaps")

    # 3-D: a "skyline" of the attack success rate for three defenses
    pick = [d for d in ["fedavg", "trimmed", "median", "sentinel"] if d in mats]
    fig = plt.figure(figsize=(DOUBLE, 2.5))
    for k, d in enumerate(pick):
        ax = fig.add_subplot(1, len(pick), k + 1, projection="3d")
        bars3d(ax, mats[d], [f"{a:g}" for a in alphas], [f"{100 * m / n_total:.0f}" for m in mals],
               "ASR (%)", "$\\alpha$", "Malicious (%)", zlab=(k == 0))
        ax.set_title(DEF_STYLE[d][0], fontsize=7, pad=-4)
    fig.subplots_adjust(wspace=0.02, left=0.0, right=1.0)
    save(fig, out, "e2_landscape_3d")


# ---------------------------------------------------------------------------
# E3: Sybil scaling
# ---------------------------------------------------------------------------
def e3(results, out, tables):
    R = load(results, "e3")
    if not R:
        return
    cell = defaultdict(list)
    noisy = defaultdict(list)
    for r in R:
        c = r["config"]
        if abs(c["sybil_jitter"] - 0.01) > 1e-9:
            noisy[(c["defense"], c["sybil_jitter"])].append(r["final"]["asr"])
            continue
        cell[(c["defense"], c["n_clients"] - 1, c["n_virtual"])].append(r["final"]["asr"])
    if noisy:
        lines = [r"\begin{tabular}{lccc}", r"\toprule", r"Jitter of the Sybil copies & FoolsGold & Multi-Krum & Dual \\", r"\midrule"]
        for j in sorted({k[1] for k in noisy}):
            vals = [f"{np.mean(noisy[(d, j)]):.1f}" if noisy.get((d, j)) else "--" for d in ("foolsgold", "krum", "dual")]
            lines.append(f"{100 * j:.0f}\\% & " + " & ".join(vals) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        tables["e3_noisy_sybils.tex"] = "% ASR (%), 20 honest clients, 10 virtual identities\n" + "\n".join(lines)
    defenses = [d for d in ["fedavg", "median", "krum", "fltrust", "foolsgold", "sentinel", "dual"] if any(k[0] == d for k in cell)]
    hs = sorted({k[1] for k in cell}); vs = sorted({k[2] for k in cell})
    fig, axes = plt.subplots(1, len(hs), figsize=(DOUBLE, 2.1), sharey=True, squeeze=False)
    xs = np.arange(len(vs))
    for ax, h in zip(axes[0], hs):
        for d in defenses:
            lab, col, mk, ls = DEF_STYLE[d]
            m = [np.mean(cell[(d, h, v)]) if cell.get((d, h, v)) else np.nan for v in vs]
            ax.plot(xs, m, color=col, marker=mk, ls=ls, label=lab, ms=3.5)
        ax.set_xticks(xs); ax.set_xticklabels([str(v) for v in vs])
        ax.set_xlabel("Virtual identities $V$")
        ax.set_title(f"{h} honest clients", fontsize=7, loc="left", pad=3)
        ax.set_ylim(-3, 103)
    axes[0][0].set_ylabel("Attack success rate (%)")
    axes[0][-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, out, "e3_sybil_scaling")

    # heatmaps (H x V) per defense, and a 3-D pair
    ncol = 4; nrow = math.ceil(len(defenses) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(DOUBLE, 1.45 * nrow + 0.2), squeeze=False, constrained_layout=True)
    for k, (ax, d) in enumerate(zip(axes.flat, defenses)):
        M = np.array([[np.mean(cell[(d, h, v)]) if cell.get((d, h, v)) else np.nan for v in vs] for h in hs])
        im = heatmap(ax, M, [str(v) for v in vs], [str(h) for h in hs], fs=5)
        ax.set_title(DEF_STYLE[d][0], fontsize=7, loc="left", pad=3)
        ax.tick_params(axis="x", labelsize=6)
        if k // ncol == nrow - 1 or k + ncol >= len(defenses):
            ax.set_xlabel("Virtual identities $V$")
        if k % ncol == 0:
            ax.set_ylabel("Honest clients")
        else:
            ax.set_yticklabels([])
    for ax in list(axes.flat)[len(defenses):]:
        ax.axis("off")
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01, aspect=25); cb.set_label("ASR (%)"); cb.outline.set_visible(False)
    # breaking points: smallest V at which the ASR exceeds 10 %
    lines = [r"\begin{tabular}{@{}l" + "c" * len(hs) + "@{}}", r"\toprule",
             "Defense & " + " & ".join(f"$H = {h}$" for h in hs) + r" \\", r"\midrule"]
    for d in defenses:
        row = []
        for h in hs:
            broke = [v for v in vs if cell.get((d, h, v)) and np.mean(cell[(d, h, v)]) > 10]
            row.append(str(min(broke)) if broke else "--")
        lines.append(DEF_STYLE[d][0] + " & " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tables["e3_breaking_points.tex"] = "% smallest number of virtual identities with ASR > 10% (-- = never, up to 40)\n" + "\n".join(lines)
    save(fig, out, "e3_sybil_heatmaps")

    pick = [d for d in ["fedavg", "median", "sentinel", "dual"] if d in defenses]
    fig = plt.figure(figsize=(DOUBLE, 2.5))
    for k, d in enumerate(pick):
        ax = fig.add_subplot(1, len(pick), k + 1, projection="3d")
        Z = np.array([[np.mean(cell[(d, h, v)]) if cell.get((d, h, v)) else np.nan for v in vs] for h in hs])
        bars3d(ax, Z, [str(v) for v in vs], [str(h) for h in hs], "ASR (%)", "Virtual identities", "Honest", zlab=(k == 0))
        ax.set_title(DEF_STYLE[d][0], fontsize=7, pad=-4)
    fig.subplots_adjust(wspace=0.02, left=0.0, right=1.0)
    save(fig, out, "e3_sybil_3d")


# ---------------------------------------------------------------------------
# E4: heterogeneity (non-IID)
# ---------------------------------------------------------------------------
def e4(results, out, tables):
    R = load(results, "e4")
    if not R:
        return
    byp = defaultdict(lambda: defaultdict(list))
    for r in R:
        part, rest = tag_of(r).split("_", 1)
        byp[part][rest].append(r)
    parts = [p for p in ["b0.05", "b0.1", "b0.3", "b1", "b10", "iid"] if p in byp]
    plab = {"b0.05": "0.05", "b0.1": "0.1", "b0.3": "0.3", "b1": "1", "b10": "10", "iid": "IID"}
    taus = sorted({float(k.split("tau")[1]) for p in parts for k in byp[p] if "tau" in k})
    deltas = sorted({float(k.split("delta")[1]) for p in parts for k in byp[p] if k.startswith("delta")})

    rows = [(f"Sentinel-Flow, $\\tau$ = {t:g}", f"sentinel_tau{t:g}") for t in taus] + \
           [(f"Differential, $\\delta$ = {d:g}", f"delta{d:g}") for d in deltas]

    def mat(fn):
        return np.array([[fn(byp[p].get(key, [])) for p in parts] for _, key in rows])

    hon = mat(lambda rs: 100 * np.mean([x["honest_acceptance_rate"] for x in rs]) if rs else np.nan)
    asr = mat(lambda rs: np.mean([x["final"]["asr"] for x in rs]) if rs else np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 0.3 * len(rows) + 0.8), sharey=True, constrained_layout=True)
    heatmap(axes[0], hon, [plab[p] for p in parts], [r[0] for r in rows], vmin=50, vmax=100, fs=6)
    axes[0].set_title("Honest updates accepted (%)", fontsize=7, loc="left", pad=3)
    heatmap(axes[1], asr, [plab[p] for p in parts], [r[0] for r in rows], vmin=0, vmax=20, fs=6)
    axes[1].set_title("Attack success rate (%), color scale 0-20", fontsize=7, loc="left", pad=3)
    for ax in axes:
        ax.set_xlabel("Dirichlet $\\beta$ (smaller = more heterogeneous)")
    save(fig, out, "e4_noniid_heatmaps")

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.1), constrained_layout=True)
    series = [("fedavg_noattack", "FedAvg, no attack", MUTED, "o", "--")] + \
             [(d, DEF_STYLE[d][0], DEF_STYLE[d][1], DEF_STYLE[d][2], "-") for d in ["median", "trimmed", "krum", "fltrust"]] + \
             [("sentinel_tau0.6", "Sentinel-Flow ($\\tau$=0.6)", SERIES[0], "o", "-")]
    if deltas:
        series.append((f"delta{0.2 if 0.2 in deltas else deltas[0]:g}", "Differential ($\\delta$=0.2)", INK, "x", "-"))
    xs = np.arange(len(parts))
    for key, lab, col, mk, ls in series:
        for ax, metric in zip(axes, ["acc", "asr"]):
            m = [np.mean([x["final"][metric] for x in byp[p][key]]) if byp[p].get(key) else np.nan for p in parts]
            ax.plot(xs, m, color=col, marker=mk, ls=ls, ms=3.5, lw=1.2, label=lab)
    for ax, lab in zip(axes, ["Global accuracy (%)", "Attack success rate (%)"]):
        ax.set_xticks(xs); ax.set_xticklabels([plab[p] for p in parts]); ax.set_xlabel("Dirichlet concentration $\\beta$")
        ax.set_ylabel(lab)
    axes[0].set_ylim(84, 100); axes[1].set_ylim(0, 32)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, out, "e4_noniid_defenses")

    names = [("fedavg_noattack", "FedAvg, no attack"), ("fedavg", "FedAvg"), ("median", "Median"), ("trimmed", "Trimmed mean"),
             ("krum", "Multi-Krum"), ("fltrust", "FLTrust"), ("foolsgold", "FoolsGold"),
             ("sentinel_tau0.6", "\\SF{}, $\\tau_{safe}=0.6$"), ("sentinel_tau0.8", "\\SF{}, $\\tau_{safe}=0.8$"),
             ("delta0.2", "Two-sided differential, $\\delta=0.2$")]
    lines = [r"\begin{tabular}{@{}l" + "c" * len(parts) + "@{}}", r"\toprule",
             "Defense & " + " & ".join(("$\\beta$ = " + plab[p]) if p != "iid" else "IID" for p in parts) + r" \\", r"\midrule"]
    for key, nm in names:
        cells = []
        for p in parts:
            rs = byp[p].get(key, [])
            cells.append(f"{np.mean([x['final']['acc'] for x in rs]):.1f} / {np.mean([x['final']['asr'] for x in rs]):.1f}" if rs else "--")
        lines.append(nm + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tables["e4_baselines.tex"] = "% accuracy / ASR (%), mean of seeds\n" + "\n".join(lines)

    if len(taus) > 1 and len(parts) > 1:
        fig = plt.figure(figsize=(SINGLE, 2.6))
        ax = fig.add_subplot(111, projection="3d")
        X, Y = np.meshgrid(np.arange(len(parts)), taus)
        Z = 100 - hon[: len(taus)]
        surface3d(ax, X, Y, np.nan_to_num(Z), "Honest rejected (%)", "$\\beta$", "$\\tau$",
                  xticks=(np.arange(len(parts)), [plab[p] for p in parts]))
        save(fig, out, "e4_false_rejection_3d")


# ---------------------------------------------------------------------------
# E5: adaptive attackers vs canary design
# ---------------------------------------------------------------------------
def e5(results, out, tables):
    R = load(results, "e5")
    if not R:
        return
    by = defaultdict(list)
    for r in R:
        by[tag_of(r)].append(r)
    attackers = [a for a in ["DoS", "Probe", "R2L", "flood"] if any(k.startswith(a + "_") for k in by)]
    designs = [d for d in ["fedavg", "orig", "strat", "delta"] if any(k.endswith("_" + d) for k in by)]
    dlab = {"fedavg": "No\ndefense", "orig": "Original\ncanaries", "strat": "Stratified\ncanaries",
            "delta": "Two-sided\ndifferential"}
    alab = {"DoS": "Evade DoS", "Probe": "Evade Probe", "R2L": "Evade R2L", "flood": "False-alarm flood"}

    def success(rs, a):
        if not rs:
            return np.nan
        if a == "flood":
            return np.mean([x["final"]["fpr"] for x in rs])
        return np.mean([x["final"]["family_miss"].get(a, np.nan) for x in rs])

    M = np.array([[success(by.get(f"{a}_{d}", []), a) for d in designs] for a in attackers])
    rows_h = attackers + ["none"]
    H = np.array([[100 * np.mean([x["honest_acceptance_rate"] for x in by.get(f"{a}_{d}", [])])
                   if by.get(f"{a}_{d}") and d != "fedavg" else np.nan for d in designs] for a in rows_h])
    alab2 = dict(alab, none="No attack")
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.05), constrained_layout=True)
    heatmap(axes[0], M[::-1], [dlab[d] for d in designs], [alab[a] for a in attackers][::-1], fs=6.5)
    axes[0].set_title("Attack success (%)", fontsize=7, loc="left", pad=3)
    heatmap(axes[1], H[::-1][:, 1:] if designs[0] == "fedavg" else H[::-1],
            [dlab[d] for d in designs if d != "fedavg"], [alab2[a] for a in rows_h][::-1], fs=6.5)
    axes[1].set_title("Honest updates accepted (%)", fontsize=7, loc="left", pad=3)
    save(fig, out, "e5_adaptive_heatmaps")

    hon = {d: 100 * np.mean([x["honest_acceptance_rate"] for x in by.get(f"none_{d}", [])])
           for d in designs if by.get(f"none_{d}")}
    acc = {d: np.mean([x["final"]["acc"] for x in by.get(f"none_{d}", [])]) for d in designs if by.get(f"none_{d}")}
    lines = [r"\begin{tabular}{lcc}", r"\toprule", r"Canary design & Honest updates accepted (\%) & Accuracy, no attack (\%) \\", r"\midrule"]
    for d in hon:
        lines.append(f"{dlab[d].replace(chr(10), ' ')} & {hon[d]:.1f} & {acc[d]:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tables["e5_honest_table.tex"] = "\n".join(lines)

    ks = sorted({int(k.split("_k")[1]) for k in by if k.startswith("ksweep")})
    if ks:
        fig, ax = plt.subplots(figsize=(SINGLE, 2.0))
        for dn, col, mk in [("orig", SERIES[0], "o"), ("delta", INK, "x")]:
            m = [np.mean([x["final"]["family_miss"]["Probe"] for x in by.get(f"ksweep_Probe_{dn}_k{k}", [])]) for k in ks]
            ax.plot(ks, m, color=col, marker=mk, label={"orig": "Original canaries", "delta": "Two-sided differential"}[dn])
        ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("Canary flows per round $K$"); ax.set_ylabel("Probe miss rate (%)"); ax.set_ylim(-3, 103)
        ax.legend(loc="center right")
        save(fig, out, "e5_canary_budget")


# ---------------------------------------------------------------------------
# E6: drift x ADC
# ---------------------------------------------------------------------------
TLAB = {1.0: "1 s", 60.0: "1 min", 3600.0: "1 h", 86400.0: "1 day", 604800.0: "1 wk", 2592000.0: "1 mo", 31536000.0: "1 yr"}


def e6(results, out, tables):
    R = load(results, "e6")
    if not R:
        return
    rows = [g for r in R for g in r["grid"]]
    ts = sorted({g["t_inference"] for g in rows}); bs = sorted({g["out_bits"] for g in rows})

    def M(gdc, key="acc_mean"):
        return np.array([[np.mean([g[key] for g in rows if g["gdc"] == gdc and g["t_inference"] == t and g["out_bits"] == b])
                          for t in ts] for b in bs])
    ref = np.mean([r["reference"]["digital"]["acc_mean"] for r in R])
    pre = np.mean([r["reference"]["preset"]["acc_mean"] for r in R])
    A_on, A_off = M(True), M(False)
    gdc_gap = float(np.nanmax(np.abs(A_on - A_off)))
    lo = float(np.floor(min(np.nanmin(A_on), np.nanmin(A_off)) / 5) * 5)
    fig, ax = plt.subplots(figsize=(SINGLE, 2.0), constrained_layout=True)
    im = heatmap(ax, A_on, [TLAB.get(t, f"{t:g}") for t in ts], [str(b) for b in bs], vmin=lo, vmax=100, fmt="{:.1f}", fs=5.5)
    ax.set_xlabel("Time since programming"); ax.set_ylabel("ADC resolution (bits)")
    cb = fig.colorbar(im, ax=ax, pad=0.01, aspect=25); cb.outline.set_visible(False); cb.set_label("Accuracy (%)")
    save(fig, out, "e6_drift_adc_heatmap")
    tables["e6_notes.txt"] = (f"digital accuracy {ref:.2f}%, PCMPreset {pre:.2f}%; "
                              f"largest |GDC on - GDC off| difference {gdc_gap:.2f} pp over the grid")

    fig = plt.figure(figsize=(SINGLE, 2.8))
    ax = fig.add_subplot(111, projection="3d")
    X, Y = np.meshgrid(np.log10(ts), bs)
    keep_t = [t for t in ts if t in (1.0, 3600.0, 604800.0, 31536000.0)]
    surface3d(ax, X, Y, A_on, "Accuracy (%)", "Time since programming", "ADC bits", zlim=(lo, 100),
              xticks=(np.log10(keep_t), [TLAB[t] for t in keep_t]), azim=-60)
    save(fig, out, "e6_drift_adc_3d")

    fig, ax = plt.subplots(figsize=(SINGLE, 2.0))
    for k, b in enumerate([b for b in bs if b >= 4]):
        i = bs.index(b)
        ax.plot(range(len(ts)), A_on[i], color=SERIES[k], marker="o", ms=3, label=f"{b} bits")
    ax.axhline(ref, color=MUTED, lw=0.8, ls=(0, (1, 1.5)))
    ax.text(len(ts) - 1, ref + 0.8, "digital", fontsize=6, color=INK2, ha="right")
    ax.set_xticks(range(len(ts))); ax.set_xticklabels([TLAB.get(t, "") for t in ts])
    ax.set_xlabel("Time since programming"); ax.set_ylabel("Accuracy (%)")
    ax.legend(title="ADC resolution", ncol=3, fontsize=6, title_fontsize=6, loc="center right")
    save(fig, out, "e6_drift_lines")


# ---------------------------------------------------------------------------
# E7: PUF characterization
# ---------------------------------------------------------------------------
def e7(results, out, tables):
    d = results / "e7"
    if not d.exists():
        return
    J = {p.stem.split("__")[1]: json.loads(p.read_text()) for p in d.glob("*.json")}
    if "reliability_grid" in J:
        G = J["reliability_grid"]
        sig = sorted({g["sigma_static"] for g in G}); keep = sorted({g["keep"] for g in G}, reverse=True)
        reps = sorted({g["reps"] for g in G})
        fig, axes = plt.subplots(1, len(reps), figsize=(DOUBLE, 1.95), sharey=True, squeeze=False, constrained_layout=True)
        target = 100 * ber_for_frr(1e-3)
        for ax, rp in zip(axes[0], reps):
            M = np.array([[100 * next(g["ber"] for g in G if g["sigma_static"] == s and g["keep"] == k and g["reps"] == rp)
                           for k in keep] for s in sig])
            im = heatmap(ax, M, [f"{100 * k:g}" for k in keep], [f"{s:g}" for s in sig], vmin=0, vmax=35, fmt="{:.1f}", fs=5.5)
            outline(ax, M <= target)
            ax.set_title(f"{rp} programming cycle{'s' if rp > 1 else ''}", fontsize=7, loc="left", pad=3)
            ax.set_xlabel("Pairs kept by masking (%)")
        axes[0][0].set_ylabel("Static offset $\\sigma_s$ ($\\mu$S)")
        cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01, aspect=25); cb.set_label("Bit-error rate (%)"); cb.outline.set_visible(False)
        save(fig, out, "e7_puf_reliability_heatmaps")

    if "level_time" in J:
        G = J["level_time"]
        gs = sorted({g["g_t"] for g in G}); ts = sorted({g["t_read_delay"] for g in G})
        Z = np.array([[100 * next(x["ber"] for x in G if x["g_t"] == g and x["t_read_delay"] == t) for t in ts] for g in gs])
        fig = plt.figure(figsize=(SINGLE, 2.7))
        ax = fig.add_subplot(111, projection="3d")
        X, Y = np.meshgrid(np.log10(ts), gs)
        surface3d(ax, X, Y, Z, "BER (%)", "Read delay (s)", "$g_T$ ($\\mu$S)", zlim=(0, max(20, float(np.ceil(Z.max() / 5) * 5))),
                  xticks=(np.log10(ts)[::2], [f"$10^{{{int(np.log10(t))}}}$" for t in ts][::2]), azim=-50)
        save(fig, out, "e7_puf_level_time_3d")
        fig, ax = plt.subplots(figsize=(SINGLE, 2.0))
        im = heatmap(ax, Z, [f"$10^{{{int(np.log10(t))}}}$" for t in ts], [f"{g:g}" for g in gs], vmin=0,
                     vmax=max(20, float(np.ceil(Z.max() / 5) * 5)), fmt="{:.1f}", fs=5)
        ax.set_xlabel("Programming-to-read delay (s)"); ax.set_ylabel("Challenge level $g_T$ ($\\mu$S)")
        cb = fig.colorbar(im, ax=ax, pad=0.01); cb.set_label("Bit-error rate (%)"); cb.outline.set_visible(False)
        save(fig, out, "e7_puf_level_time_heatmap")

    if "genuine_impostor" in J:
        D = J["genuine_impostor"]
        gen = np.concatenate([x["genuine"] for x in D["default"]]); imp = np.concatenate([x["impostor"] for x in D["default"]])
        nm = np.array(D["nomask_reps1"]["genuine"])
        tau = D["default"][0]["params"]["tau"]
        fig, ax = plt.subplots(figsize=(SINGLE, 1.9))
        bins = np.linspace(0.35, 1.0, 66)
        ax.hist(imp, bins=bins, color=SERIES[1], alpha=0.85, label="Impostor chip", edgecolor="white", linewidth=0.3)
        ax.hist(nm, bins=bins, color=MUTED, alpha=0.7, label="Genuine, no masking, 1 cycle", edgecolor="white", linewidth=0.3)
        ax.hist(gen, bins=bins, color=SERIES[0], alpha=0.9, label="Genuine, masking, 3 cycles", edgecolor="white", linewidth=0.3)
        ax.axvline(tau, color=INK, lw=0.9)
        ax.text(tau - 0.005, ax.get_ylim()[1] * 0.92, f"$\\tau_{{puf}}$ = {tau:g}", ha="right", fontsize=6)
        ax.set_xlabel("Similarity to the enrolled response"); ax.set_ylabel("Verifications")
        ax.legend(loc="upper left", fontsize=6)
        save(fig, out, "e7_puf_similarity_hist")
        x = D["default"][0]
        tables["e7_puf_table.tex"] = "\n".join([
            r"\begin{tabular}{lc}", r"\toprule", r"Metric & Value \\", r"\midrule",
            f"Uniqueness (inter-chip HD) & {np.mean([d['uniqueness'] for d in D['default']]):.3f} \\\\",
            f"Uniformity & {np.mean([d['uniformity'] for d in D['default']]):.3f} \\\\",
            f"Bit-error rate, masking + 3 cycles & {100 * np.mean([d['ber'] for d in D['default']]):.2f}\\% \\\\",
            f"Bit-error rate, no masking, 1 cycle & {100 * D['nomask_reps1']['ber']:.2f}\\% \\\\",
            f"FRR at $\\tau_{{puf}}$ = {tau:g} (binomial) & {np.mean([d['frr_binomial'] for d in D['default']]):.1e} \\\\",
            f"FAR, impostor chip (binomial) & {np.mean([d['far_binomial_impostor'] for d in D['default']]):.1e} \\\\",
            f"Lowest genuine / highest impostor similarity & {min(d['genuine_min'] for d in D['default']):.3f} / {max(d['impostor_max'] for d in D['default']):.3f} \\\\",
            r"\bottomrule", r"\end{tabular}"])

    if "modeling_attack" in J:
        G = J["modeling_attack"]
        fig, ax = plt.subplots(figsize=(SINGLE, 2.0))
        for keep, col, lab in [(1.0, MUTED, "Random pairs (no masking)"), (0.25, SERIES[7], "Masked pairs (25% kept)")]:
            rows = sorted([g for g in G if g["keep"] == keep and g["crps"] > 0], key=lambda g: g["crps"])
            if rows:
                x = [g["crps"] / 256 for g in rows]
                ax.plot(x, [g["attacker_similarity"] for g in rows], color=col, marker="o", ms=3, label=lab)
        ax.axhline(0.9, color=INK, lw=0.8); ax.text(0.95, 0.905, "$\\tau_{puf}$", fontsize=6, transform=ax.get_yaxis_transform(), ha="right", va="bottom")
        ax.set_xscale("log"); ax.set_xlabel("Eavesdropped rounds (256 CRPs each)")
        ax.set_ylabel("Attacker similarity"); ax.set_ylim(0.45, 1.0)
        ax.legend(loc="lower right", fontsize=6)
        save(fig, out, "e7_puf_modeling_attack")


# ---------------------------------------------------------------------------
# E8: PUF in the loop
# ---------------------------------------------------------------------------
def e8(results, out, tables):
    R = load(results, "e8")
    if not R:
        return
    by = defaultdict(list)
    for r in R:
        by[tag_of(r)].append(r)
    sig = sorted({r["config"]["puf_sigma_static"] for r in R if tag_of(r).startswith("sig")})
    taus = sorted({r["config"]["puf_tau"] for r in R if tag_of(r).startswith("sig")})
    if sig and taus:
        H = np.array([[100 * np.mean([x["honest_acceptance_rate"] for x in by[f"sig{s:g}_tau{t:g}"]]) for t in taus] for s in sig])
        A = np.array([[np.mean([x["final"]["acc"] for x in by[f"sig{s:g}_tau{t:g}"]]) for t in taus] for s in sig])
        fig, axes = plt.subplots(1, 2, figsize=(DOUBLE * 0.7, 1.7), sharey=True)
        heatmap(axes[0], H, [f"{t:g}" for t in taus], [f"{s:g}" for s in sig], fmt="{:.1f}", fs=6)
        axes[0].set_title("Honest updates accepted (%)", fontsize=7, loc="left", pad=3)
        heatmap(axes[1], A, [f"{t:g}" for t in taus], [f"{s:g}" for s in sig], vmin=50, vmax=100, fmt="{:.1f}", fs=6)
        axes[1].set_title("Global accuracy (%)", fontsize=7, loc="left", pad=3)
        for ax in axes:
            ax.set_xlabel("$\\tau_{puf}$")
        axes[0].set_ylabel("Static offset $\\sigma_s$ ($\\mu$S)")
        save(fig, out, "e8_puf_in_loop")
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Variant & Honest acc. (\%) & Adversary acc. (\%) & Accuracy (\%) & ASR (\%) \\", r"\midrule"]
    for key in sorted(by):
        rs = by[key]
        lines.append(f"{key.replace('_', ' ')} & {100 * np.mean([x['honest_acceptance_rate'] for x in rs]):.1f} & "
                     f"{100 * np.mean([x['adversary_acceptance_rate'] or 0 for x in rs]):.1f} & "
                     f"{np.mean([x['final']['acc'] for x in rs]):.2f} & {np.mean([x['final']['asr'] for x in rs]):.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tables["e8_table.tex"] = "\n".join(lines)


# ---------------------------------------------------------------------------
# E9: overhead
# ---------------------------------------------------------------------------
def e9(results, out, tables):
    R = load(results, "e9")
    if not R:
        return
    rows = defaultdict(dict)
    for r in R:
        c = r["config"]
        rows[c["defense"]][c["n_clients"]] = {k: 1000 * v / c["rounds"] for k, v in r["timing"].items()}
    ns = sorted({n for d in rows for n in rows[d]})
    comps = [("train", "Local training", SERIES[0]), ("canary", "Canary probing", SERIES[1]),
             ("puf", "PUF verification", SERIES[7]), ("aggregate", "Aggregation", SERIES[2])]
    defs = [d for d in ["fedavg", "sentinel", "dual"] if d in rows]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.1))
    width = 0.26
    for k, d in enumerate(defs):
        xs = np.arange(len(ns)) + (k - 1) * (width + 0.02)
        bottom = np.zeros(len(ns))
        for key, lab, col in comps:
            v = np.array([rows[d].get(n, {}).get(key, 0) / 1000 for n in ns])
            ax.bar(xs, v, width, bottom=bottom, color=col, label=lab if k == 0 else None, edgecolor="white", linewidth=0.6)
            bottom += v
        for x, b in zip(xs, bottom):
            ax.text(x, b, {"fedavg": "F", "sentinel": "S", "dual": "D"}[d], ha="center", va="bottom", fontsize=5.5, color=INK2)
    ax.set_xticks(np.arange(len(ns))); ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Clients"); ax.set_ylabel("Simulated time per round (s)")
    ax.legend(loc="upper left", fontsize=6)
    ax.text(0.99, 0.99, "F  FedAvg\nS  Sentinel-Flow\nD  dual attestation", transform=ax.transAxes,
            ha="right", va="top", fontsize=5.5, color=INK2, linespacing=1.3)
    save(fig, out, "e9_overhead")
    n_params = 122 * 64 + 64 + 64 + 1
    tables["e9_bytes.tex"] = "\n".join([
        r"\begin{tabular}{lr}", r"\toprule", r"Message (per client and round) & Size \\", r"\midrule",
        f"Model update ({n_params} float32 parameters) & {4 * n_params / 1024:.1f} KiB \\\\",
        f"Canary flows (100 x 122 float32) & {100 * 122 * 4 / 1024:.1f} KiB \\\\",
        f"PUF challenge (256 pairs of 13-bit cell indices + level) & {math.ceil(256 * 26 / 8 + 4)} B \\\\",
        r"PUF response (256 bits) & 32 B \\", r"\bottomrule", r"\end{tabular}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--out", default=str(HERE / "figures"))
    a = ap.parse_args()
    style()
    res, out = Path(a.results), Path(a.out)
    tables: dict[str, str] = {}
    for fn in (e1, e2, e3, e4, e5, e6, e7, e8, e9):
        print(fn.__name__)
        try:
            fn(res, out, tables)
        except Exception as exc:  # noqa: BLE001 - keep going, report
            import traceback
            print(f"  {fn.__name__} failed: {exc}")
            traceback.print_exc()
    (out / "tables").mkdir(parents=True, exist_ok=True)
    for name, txt in tables.items():
        (out / "tables" / name).write_text(txt + "\n")
        print("  wrote tables/" + name)


if __name__ == "__main__":
    main()
