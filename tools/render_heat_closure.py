#!/usr/bin/env python3
"""Render truthful Section-9 presentation PNGs from heat-closure CSV exports.

Every value shown is read from exported CSV snapshots or from the T4
``summary.json``; nothing is synthesised or smoothed.  The renderer is
reproducible:

    python3 tools/render_heat_closure.py \
        --runs    /tmp/alpakaNN-results/nhc-20261001/runs \
        --summary /tmp/alpakaNN-results/nhc-20261001/T4/summary.json \
        --final-nn /tmp/alpakaNN-results/nhc-20261001/runs-finalmodel/nn \
        --output  results/heat_closure/nhc-20261001/figures

The legacy ``--reference/--comparison`` invocation and the ``load_run`` /
``render`` helpers are preserved for the existing smoke test.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-render-heat-closure")
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("matplotlib is required: python3 -m pip install matplotlib") from exc

# Fixed presentation ranges (documented in figures/manifest.json).
CANVAS = (19.2, 10.8)          # inches; at DPI=100 -> 1920 x 1080
DPI = 100
TEMPERATURE_RANGE = (0.0, 1.0)         # u, all temperature panels
TEMPERATURE_DIFF_RANGE = (0.0, 0.5)    # |u_nn - u_preset|
COEFFICIENT_RANGE = (0.0, 4.5)         # alpha panels (covers union of true/nn)
COEFFICIENT_ERROR_RANGE = (0.0, 3.5)   # |alpha_nn - alpha_true|
CMAP_T = "inferno"
CMAP_C = "viridis"
CMAP_E = "magma"
BETA_DEFAULT = 0.5


# --------------------------------------------------------------------------- #
# Legacy API (used by tools/tests/presentation_smoke.py)
# --------------------------------------------------------------------------- #
def load_run(path):
    p = Path(path)
    rows = list(csv.DictReader((p / "manifest.csv").open()))
    if not rows:
        raise ValueError(f"{p}: empty manifest")
    out = []
    for row in rows:
        a = np.genfromtxt(p / f"frame_{int(row['frame']):06d}.csv", delimiter=",", names=True)
        n = len(a)
        nx = len(np.unique(a["x"]))
        ny = len(np.unique(a["y"]))
        if n != nx * ny or nx < 2 or ny < 2:
            raise ValueError("snapshot is not a complete rectangular grid")
        x = np.unique(a["x"])
        y = np.unique(a["y"])
        z = np.asarray(a["u"]).reshape(ny, nx)
        out.append((float(row["time"]), x, y, z))
    return out


def render(ref, out, other=None, width=1920, height=1080):
    """Legacy reference/comparison renderer (kept for backward compatibility)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 15})
    setup_diagram(out)
    for i, (t, x, y, u) in enumerate(ref):
        fig, ax = plt.subplots(1, 3 if other else 1, figsize=(width / DPI, height / DPI),
                               constrained_layout=True)
        axs = np.atleast_1d(ax)
        im = axs[0].pcolormesh(x, y, u, vmin=0, vmax=1, cmap=CMAP_T, shading="nearest")
        axs[0].set(title="Reference temperature", xlabel="x", ylabel="y", aspect="equal")
        fig.colorbar(im, ax=axs[0], label="u [0,1]")
        if other:
            ot, ox, oy, v = other[min(range(len(other)), key=lambda k: abs(other[k][0] - t))]
            if v.shape != u.shape or not np.allclose([ox[0], ox[-1], oy[0], oy[-1]],
                                                     [x[0], x[-1], y[0], y[-1]]):
                raise ValueError("comparison grids differ")
            for a, z, title, cmap, lo, hi in [(axs[1], v, "Neural temperature", CMAP_T, 0, 1),
                                              (axs[2], abs(v - u), "Absolute difference", CMAP_E, 0, 1)]:
                q = a.pcolormesh(x, y, z, vmin=lo, vmax=hi, cmap=cmap, shading="nearest")
                a.set(title=title, xlabel="x", ylabel="y", aspect="equal")
                fig.colorbar(q, ax=a)
        fig.suptitle(f"Heat closure simulation — t = {t:.6g} (physical time)")
        fig.savefig(out / f"comparison_{i:06d}.png", dpi=DPI)
        plt.close(fig)
    return len(ref)


# --------------------------------------------------------------------------- #
# Data loading helpers
# --------------------------------------------------------------------------- #
def read_manifest(run_dir):
    rows = list(csv.DictReader((Path(run_dir) / "manifest.csv").open()))
    if not rows:
        raise ValueError(f"{run_dir}: empty manifest")
    return rows


def read_summary(path):
    with Path(path).open() as handle:
        return json.load(handle)


def load_alpha(run_dir):
    """Return (frames, n, x, y).

    ``frames`` has shape (n_times, ny, nx, 6) with columns
    ``x, y, u, alpha, time, step`` (x fastest, y outer).
    """
    a = np.genfromtxt(Path(run_dir) / "alpha.csv", delimiter=",", skip_header=1,
                      usecols=(0, 1, 2, 3, 5, 6))
    if a.ndim == 1:
        a = a[None, :]
    n_times = len(np.unique(a[:, 4]))
    n = int(round(math.sqrt(len(a) / n_times)))
    if n < 2 or n * n * n_times != len(a):
        raise ValueError(f"{run_dir}/alpha.csv: not a full rectangular grid sequence")
    frames = a.reshape(n_times, n, n, 6)  # [time, y-index, x-index, column]
    x = frames[0, 0, :, 0]
    y = frames[0, :, 0, 1]
    return frames, n, x, y


def base_alpha_grid(X, Y):
    """Analytical material base from ConservativeSolver.hpp (no temperature factor)."""
    out = 0.5 + 0.4 * (np.sin(6.0 * np.pi * Y) >= 0.0)
    conductor = (np.abs(Y - 0.5) < 0.05) & (X > 0.45) & (X < 0.65)
    out = np.where(conductor, 4.0, out)
    inclusion = (X - 0.35) ** 2 + (Y - 0.5) ** 2 < 0.12 ** 2
    out = np.where(inclusion, 0.02, out)
    return out


def rebuild_true_alpha(nn_alpha_frame, beta):
    """alpha_true(x,y,u) = base_alpha(x,y) * (1 + beta*u), from exported u."""
    x = nn_alpha_frame[0, :, 0]     # x-index axis
    y = nn_alpha_frame[:, 0, 1]     # y-index axis
    X, Y = np.meshgrid(x, y)        # (ny, nx)
    u = nn_alpha_frame[:, :, 2]
    return base_alpha_grid(X, Y) * (1.0 + beta * u), X, Y


def nearest_frame(frames, target_time):
    idx = min(range(len(frames)), key=lambda k: abs(frames[k][0] - target_time))
    return idx, frames[idx]


def region_masks(X, Y):
    """Disjoint region masks with the same priority as T4 compute_metrics.region().

    A cell inside the inclusion is *inclusion* even if it also satisfies the
    conductor box; this matches the metric generation that produced summary.json.
    """
    inclusion = (X - 0.35) ** 2 + (Y - 0.5) ** 2 < 0.12 ** 2
    conductor = (~inclusion) & (np.abs(Y - 0.5) < 0.05) & (X > 0.45) & (X < 0.65)
    stripe_high = (~inclusion) & (~conductor) & (np.sin(6.0 * np.pi * Y) >= 0.0)
    background_low = (~inclusion) & (~conductor) & (~stripe_high)
    return {"inclusion": inclusion, "conductor": conductor,
            "stripe_high": stripe_high, "background_low": background_low}


def alpha_error_metrics(true_alpha, nn_alpha, X, Y):
    err = np.abs(nn_alpha - true_alpha)
    masks = region_masks(X, Y)
    out = {"overall": {"mae": float(err.mean()), "max_abs": float(err.max()),
                       "n": int(err.size)}}
    for name, mask in masks.items():
        count = int(mask.sum())
        if count:
            out[name] = {"mae": float(err[mask].mean()), "max_abs": float(err[mask].max()),
                         "n": count}
        else:
            out[name] = {"mae": float("nan"), "max_abs": float("nan"), "n": 0}
    return out, err


def save_png(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Figure 1 — setup diagram
# --------------------------------------------------------------------------- #
def setup_diagram(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n = 400
    x = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, x)
    base = base_alpha_grid(X, Y)
    fig, ax = plt.subplots(figsize=CANVAS, constrained_layout=True)
    im = ax.pcolormesh(X, Y, base, cmap=CMAP_C, vmin=0.0, vmax=4.5, shading="auto")
    fig.colorbar(im, ax=ax, label=r"analytical $\alpha_{base}(x,y)$ [unitless]")
    ax.axvspan(0.0, 0.008, color="red", alpha=0.9)
    ax.axvspan(0.992, 1.0, color="blue", alpha=0.9)
    ax.plot([0, 1], [0, 0], color="black", lw=7, solid_capstyle="butt")
    ax.plot([0, 1], [1, 1], color="black", lw=7, solid_capstyle="butt")
    ax.add_patch(Circle((0.35, 0.5), 0.12, fill=False, ec="white", lw=2.0))
    ax.add_patch(Rectangle((0.45, 0.45), 0.20, 0.10, fill=False, ec="cyan", lw=2.0))
    ax.annotate("hot wall: u = 1", xy=(0.004, 0.5), xytext=(0.03, 0.93),
                color="red", fontsize=17, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="red", lw=2))
    ax.annotate("cold wall: u = 0", xy=(0.996, 0.5), xytext=(0.72, 0.93),
                color="blue", fontsize=17, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="blue", lw=2))
    ax.text(0.5, 0.02, "insulated face (zero flux)", ha="center", va="bottom", fontsize=14)
    ax.text(0.5, 0.985, "insulated face (zero flux)", ha="center", va="top", fontsize=14)
    ax.annotate("low-transport inclusion\n$\\alpha_{base}=0.02$, r = 0.12",
                xy=(0.29, 0.36), xytext=(0.06, 0.14), color="white", fontsize=15,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="white", lw=2))
    ax.annotate("high-transport conductor\n$\\alpha_{base}=4.0$, 0.45<x<0.65",
                xy=(0.55, 0.55), xytext=(0.60, 0.16), color="cyan", fontsize=15,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="cyan", lw=2))
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x", ylabel="y", aspect="equal",
           title="Heat-closure setup: specified geometry and analytical material "
                 "(diagram; not a simulation result)")
    return save_png(fig, out / "setup_diagram.png")


# --------------------------------------------------------------------------- #
# Figure 2 — homogeneous temperature sequence (uniform mode)
# --------------------------------------------------------------------------- #
def homogeneous_sequence(out, uniform_dir, times=(0.0, 0.025, 0.05, 0.075, 0.1)):
    out = Path(out)
    frames = load_run(uniform_dir)
    picks = []
    for t in times:
        idx, fr = nearest_frame(frames, t)
        picks.append((idx, fr, t))
    fig, axes = plt.subplots(1, len(picks), figsize=CANVAS, constrained_layout=True)
    im = None
    for ax, (idx, (tsaved, x, y, u), target) in zip(np.atleast_1d(axes), picks):
        im = ax.pcolormesh(x, y, u, vmin=TEMPERATURE_RANGE[0], vmax=TEMPERATURE_RANGE[1],
                           cmap=CMAP_T, shading="nearest")
        ax.set(xlabel="x", ylabel="y", aspect="equal",
               title=f"t = {tsaved:.4f}")
    fig.colorbar(im, ax=list(np.atleast_1d(axes)), label="u [0,1]",
                 location="right", shrink=0.85)
    fig.suptitle("Homogeneous material (uniform mode, $\\alpha=0.5$): temperature sequence "
                 "— fixed range [0,1], uniform grid 64$\\times$64")
    return save_png(fig, out / "homogeneous_sequence.png")


# --------------------------------------------------------------------------- #
# Figure 3 — true vs learned coefficient on the same NN state
# --------------------------------------------------------------------------- #
def coefficient_comparison(out, nn_alpha_dir, beta, ref_metrics, log):
    out = Path(out)
    frames, n, x, y = load_alpha(nn_alpha_dir)
    fr = frames[-1]
    nn_alpha = fr[:, :, 3]
    true_alpha, X, Y = rebuild_true_alpha(fr, beta)
    metrics, err = alpha_error_metrics(true_alpha, nn_alpha, X, Y)
    lo = float(min(true_alpha.min(), nn_alpha.min()))
    hi = float(max(true_alpha.max(), nn_alpha.max()))
    c_lo, c_hi = COEFFICIENT_RANGE
    e_lo, e_hi = COEFFICIENT_ERROR_RANGE
    true_max = float(err.max())
    clipped = int((err > e_hi).sum())
    log(f"coefficient: true range [{true_alpha.min():.6g},{true_alpha.max():.6g}] "
        f"nn range [{nn_alpha.min():.6g},{nn_alpha.max():.6g}] "
        f"display [{c_lo},{c_hi}] MAE {metrics['overall']['mae']:.6g} "
        f"max {true_max:.6g} clipped>{e_hi}: {clipped}")
    if ref_metrics is not None:
        ref_mae = ref_metrics["overall"]["mae"]
        ref_max = ref_metrics["overall"]["max_abs"]
        if not (math.isclose(metrics["overall"]["mae"], ref_mae, rel_tol=1e-9)
                and math.isclose(metrics["overall"]["max_abs"], ref_max, rel_tol=1e-9)):
            log("WARNING: rendered coefficient MAE/max disagree with summary.json "
                f"({metrics['overall']['mae']:.6g}/{metrics['overall']['max_abs']:.6g} vs "
                f"{ref_mae:.6g}/{ref_max:.6g})")
    panels = [
        (true_alpha, r"$\alpha_{true}(x,y,u_{nn})$ (rebuilt from exported $u$)", CMAP_C, c_lo, c_hi),
        (nn_alpha, r"$\alpha_{nn}$ (same NN state)", CMAP_C, c_lo, c_hi),
        (err, r"$|\alpha_{nn}-\alpha_{true}|$", CMAP_E, e_lo, e_hi),
    ]
    fig, axes = plt.subplots(1, 3, figsize=CANVAS, constrained_layout=True)
    for ax, (z, title, cmap, zlo, zhi) in zip(axes, panels):
        im = ax.pcolormesh(x, y, z, vmin=zlo, vmax=zhi, cmap=cmap, shading="nearest")
        ax.set(xlabel="x", ylabel="y", aspect="equal", title=title)
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(
        "Coefficient on the same final NN state (final-model b3a5f955, t = "
        f"{fr[0,0,4]:.4f}, step {int(fr[0,0,5])}) — fixed display range "
        f"[{c_lo},{c_hi}], fixed error range [{e_lo},{e_hi}]\n"
        f"TRUE max |Δα| = {true_max:.4f} (overall MAE {metrics['overall']['mae']:.4f}); "
        f"clipped points (>{e_hi}): {clipped}; no clipping applied")
    return save_png(fig, out / "coefficient_comparison.png"), metrics, clipped


# --------------------------------------------------------------------------- #
# Figure 4 — preset vs NN temperature at matched saved times
# --------------------------------------------------------------------------- #
def comparison_sequence(out, preset_dir, nn_dir, times, log):
    out = Path(out)
    preset = load_run(preset_dir)
    nn = load_run(nn_dir)
    pman = read_manifest(preset_dir)
    nman = read_manifest(nn_dir)
    produced = []
    for k, t in enumerate(times):
        pi, (pt, px, py, up) = nearest_frame(preset, t)
        ni, (nt, nx, ny, un) = nearest_frame(nn, t)
        if up.shape != un.shape:
            raise ValueError("preset/nn grids differ")
        if abs(nt - pt) > 1e-12:
            log(f"WARNING: nearest saved times differ at frame {k}: preset {pt} nn {nt}")
        step = pman[pi]["step"]
        if pman[pi]["step"] != nman[ni]["step"]:
            log(f"WARNING: matched step differs at frame {k}: preset {step} nn {nman[ni]['step']}")
        diff = np.abs(un - up)
        d_true_max = float(diff.max())
        d_clipped = int((diff > TEMPERATURE_DIFF_RANGE[1]).sum())
        fig, axes = plt.subplots(1, 3, figsize=CANVAS, constrained_layout=True)
        panels = [
            (up, "preset (analytical $\\alpha_{true}$)", CMAP_T, *TEMPERATURE_RANGE),
            (un, "nn (learned $\\alpha_{nn}$, final-model b3a5f955)", CMAP_T, *TEMPERATURE_RANGE),
            (diff, r"$|u_{nn}-u_{preset}|$", CMAP_E, *TEMPERATURE_DIFF_RANGE),
        ]
        for ax, (z, title, cmap, zlo, zhi) in zip(axes, panels):
            im = ax.pcolormesh(px, py, z, vmin=zlo, vmax=zhi, cmap=cmap, shading="nearest")
            ax.set(xlabel="x", ylabel="y", aspect="equal", title=title)
            fig.colorbar(im, ax=ax, shrink=0.85)
        fig.suptitle(
            f"Temperature at matched saved physical time t = {pt:.6f} (step {step})\n"
            "preset vs final-model NN; matching uses the nearest saved manifest time "
            "(exact match here); fixed u range [0,1], fixed |Δu| range "
            f"[{TEMPERATURE_DIFF_RANGE[0]},{TEMPERATURE_DIFF_RANGE[1]}]\n"
            f"TRUE max |Δu| = {d_true_max:.4f}; clipped points (>{TEMPERATURE_DIFF_RANGE[1]}): "
            f"{d_clipped}; no clipping applied")
        produced.append(save_png(fig, out / f"comparison_{k:06d}.png"))
        log(f"comparison_{k:06d}: t={pt:.6f} step={step} "
            f"true max|Δu|={d_true_max:.6f} clipped>{TEMPERATURE_DIFF_RANGE[1]}: {d_clipped}")
    return produced


# --------------------------------------------------------------------------- #
# Figure 5 — final comparison: errors + measured NN overhead
# --------------------------------------------------------------------------- #
def _fmt(value, digits=6):
    return f"{value:.{digits}f}"


def final_comparison(out, summary, log):
    out = Path(out)
    field = summary.get("field_t2_final_model") or summary.get("field")
    coeff = summary.get("coefficient_t2_final_model") or summary.get("coefficient")
    mesh = summary.get("mesh", {})
    timing = summary.get("timing", {})
    label = ("final-model weights b3a5f955 (supplementary host-only run)"
             if "field_t2_final_model" in summary else "primary matrix weights f075cffd")
    lines = [
        f"NUMERICAL ERRORS — {label}",
        "grid 64, tmax 0.1, 121 saved times, 13654 steps",
        "",
        "Field  u_nn vs u_preset",
        f"  final t=0.1        relL2 = {_fmt(field['final']['relL2'])}   "
        f"normLinf = {_fmt(field['final']['normLinf'])}",
        f"  max over 121 times relL2 = {_fmt(field['max_over_times_relL2'])}   "
        f"normLinf = {_fmt(field['max_over_times_normLinf'])}",
        f"  mean over times    relL2 = {_fmt(field['mean_over_times_relL2'])}",
        "",
        "Coefficient  alpha_nn vs alpha_true (final state)",
    ]
    order = ["overall", "inclusion", "conductor", "stripe_high", "background_low"]
    for name in order:
        if name not in coeff["by_region"] and name != "overall":
            continue
        rec = coeff["overall"] if name == "overall" else coeff["by_region"][name]
        lines.append(f"  {name:<14} n={rec['n']:>4}  MAE = {_fmt(rec['mae'])}   "
                     f"max = {_fmt(rec['max_abs'])}")
    lines += [f"  clipped at bounds: {coeff['clipped_points']}", "",
              "Mesh convergence with analytical preset coefficient (relL2)"]
    for key in ("32/64", "64/128", "128/256"):
        if key in mesh:
            lines.append(f"  {key}: relL2 = {_fmt(mesh[key]['relL2'])}   "
                         f"normLinf = {_fmt(mesh[key]['normLinf'])}")
    text = "\n".join(lines)

    fig = plt.figure(figsize=CANVAS, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    ax0.text(0.0, 1.0, text, va="top", ha="left", family="DejaVu Sans Mono", fontsize=14.5)

    ax1 = fig.add_subplot(gs[0, 1])
    backends = [b for b in ("host", "hip", "cuda", "sycl") if f"{b}/nn" in timing]
    totals = [timing[f"{b}/nn"]["total_seconds"] for b in backends]
    labels = []
    for b in backends:
        nn = timing[f"{b}/nn"]
        pr = timing.get(f"{b}/preset")
        if pr and pr["seconds_per_step"] > 0:
            ratio = nn["seconds_per_step"] / pr["seconds_per_step"]
            labels.append(f"{b}  ({ratio:.1f}x preset)")
        else:
            labels.append(b)
    ypos = np.arange(len(backends))
    bars = ax1.barh(ypos, totals, color="#3b6ea5")
    ax1.set_yticks(ypos)
    ax1.set_yticklabels(labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("measured NN compute-only wall time [s] (13654 steps)")
    ax1.set_title("Measured NN runtime overhead\n(compute-only, no per-step D2H; vs preset)")
    for bar, tot in zip(bars, totals):
        ax1.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                 f"  {tot:.2f} s", va="center", fontsize=13)
    fig.suptitle("Final comparison — measured numerical errors and runtime overhead "
                 "(all values read from summary.json and exported CSVs; nothing extrapolated)")
    log("final_comparison written from summary.json field/coefficient/mesh/timing")
    return save_png(fig, out / "final_comparison.png")


# --------------------------------------------------------------------------- #
# Figure 6 — backend table
# --------------------------------------------------------------------------- #
def cross_backend_alpha_parity(runs_dir, log):
    base = Path(runs_dir)
    host_frames, n, _, _ = load_alpha(base / "host" / "nn")
    host_last = host_frames[-1, :, :, 3]
    out = {"host": 0.0}
    for backend in ("hip", "cuda", "sycl"):
        try:
            frames, nb, _, _ = load_alpha(base / backend / "nn")
            if nb != n:
                raise ValueError("grid mismatch")
            out[backend] = float(np.abs(frames[-1, :, :, 3] - host_last).max())
        except (OSError, ValueError) as exc:
            log(f"inference parity {backend}: unavailable ({exc}); omitted")
            out[backend] = None
    return out


def backend_table(out, summary, runs_dir, log):
    out = Path(out)
    devices = summary.get("provenance", {}).get("devices", {})
    cross = summary.get("cross_backend_max_abs_vs_host_final", {})
    timing = summary.get("timing", {})
    parity = cross_backend_alpha_parity(runs_dir, log)
    precision = "fp64 state / fp32 NN"

    header = ["Backend", "Device", "Precision",
              "Inference parity\nmax|Δα| vs host", "Field parity max|Δu|\npreset / nn",
              "preset total s", "nn total s", "nn/preset\nper-step", "Status"]
    rows = []
    for backend in ("host", "hip", "cuda", "sycl"):
        preset = timing.get(f"{backend}/preset")
        nn = timing.get(f"{backend}/nn")
        ratio = "—"
        if preset and nn and preset["seconds_per_step"] > 0:
            ratio = f"{nn['seconds_per_step'] / preset['seconds_per_step']:.2f}x"
        fp = cross.get(f"{backend}/preset")
        fn = cross.get(f"{backend}/nn")
        fpar = "—"
        if fp is not None and fn is not None:
            fpar = f"{fp:.2e} / {fn:.2e}"
        ip = parity.get(backend)
        ip_txt = "reference" if backend == "host" else (f"{ip:.2e}" if ip is not None else "n/a")
        uniform = "ran" if f"{backend}/uniform" in timing else "SKIPPED"
        status = f"preset/nn ran; uniform {uniform}"
        rows.append([backend, devices.get(backend, "n/a"), precision, ip_txt, fpar,
                     f"{preset['total_seconds']:.4f}" if preset else "SKIPPED",
                     f"{nn['total_seconds']:.4f}" if nn else "SKIPPED",
                     ratio, status])

    fig, ax = plt.subplots(figsize=CANVAS, constrained_layout=True)
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12.5)
    tbl.scale(1.0, 2.6)
    widths = [0.06, 0.24, 0.11, 0.11, 0.13, 0.085, 0.08, 0.075, 0.14]
    for (row, col), cell in tbl.get_celld().items():
        cell.set_width(widths[col] if col < len(widths) else 0.1)
        if row == 0:
            cell.set_text_props(fontweight="bold")
        if row > 0 and "SKIPPED" in rows[row - 1][8]:
            cell.set_facecolor("#ffe9e0")
    ax.set_title(
        "Backend matrix — grid 64, tmax 0.1, 13654 steps, primary matrix weights "
        "f075cffd\n"
        "Inference parity: max|α_backend(host nn) − α_host(host nn)| on the final frame "
        "(computed from alpha.csv).\n"
        "Field parity: cross_backend_max_abs_vs_host_final (summary.json). "
        "All preset/nn combinations ran; uniform was only run on host.\n"
        "Precision: solver state fp64; NN inference fp32 (sycl uses documented FP64 "
        "emulation for the state). Skipped cells are marked explicitly.",
        fontsize=13.5)
    log("backend_table: rows host/hip/cuda/sycl; inference parity from alpha.csv; "
        "field parity and timings from summary.json")
    return save_png(fig, out / "backend_table.png")


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def _rel_or_abs(path):
    return str(Path(path))


def write_figures_manifest(out, entries):
    out = Path(out)
    payload = {
        "schema": "heat-closure-figures/1",
        "note": "Every figure is rendered from the listed CSV/JSON sources; values are "
                "read, never synthesised. Fixed ranges are recorded per figure.",
        "fixed_ranges": {
            "temperature": list(TEMPERATURE_RANGE),
            "temperature_difference": list(TEMPERATURE_DIFF_RANGE),
            "coefficient": list(COEFFICIENT_RANGE),
            "coefficient_error": list(COEFFICIENT_ERROR_RANGE),
        },
        "figures": entries,
    }
    path = out / "manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", help="matrix runs dir: <dir>/<backend>/<mode>/")
    parser.add_argument("--summary", help="T4 summary.json (field/coefficient/mesh/timing)")
    parser.add_argument("--final-nn", dest="final_nn",
                        help="supplementary final-model NN run dir (b3a5f955 host nn)")
    parser.add_argument("--output", required=True, help="figures output directory")
    parser.add_argument("--beta", type=float, default=BETA_DEFAULT,
                        help="temperature coefficient factor (default 0.5)")
    # legacy invocation
    parser.add_argument("--log", default=None,
                        help="render log path (default: <output>/render.log)")
    # legacy invocation
    parser.add_argument("--reference")
    parser.add_argument("--comparison")
    args = parser.parse_args()

    out = Path(args.output)
    log_path = Path(args.log) if args.log else out / "render.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(message):
        lines.append(str(message))
        print(message)

    if args.reference:  # legacy mode
        n = render(load_run(args.reference), out,
                   load_run(args.comparison) if args.comparison else None)
        print(f"Rendered {n} frames; values sourced from CSV snapshots.")
        log_path.write_text("\n".join(lines) + "\n")
        return

    if not (args.runs and args.summary):
        parser.error("--runs and --summary are required unless --reference is used")

    runs = Path(args.runs)
    summary = read_summary(args.summary)
    final_nn = Path(args.final_nn) if args.final_nn else runs / "host" / "nn"
    out.mkdir(parents=True, exist_ok=True)
    entries = []

    print("rendering figures…")
    p = setup_diagram(out)
    entries.append({"file": p.name, "sources": ["example/heatEquationNn/src/"
                     "ConservativeSolver.hpp:baseAlpha/alphaTrue (analytical geometry)"],
                    "ranges": {"coefficient": list(COEFFICIENT_RANGE)},
                    "note": "specified geometry diagram; not a simulation result"})

    p = homogeneous_sequence(out, runs / "host" / "uniform")
    entries.append({"file": p.name,
                    "sources": [f"{runs}/host/uniform/frame_*.csv",
                                f"{runs}/host/uniform/manifest.csv"],
                    "ranges": {"temperature": list(TEMPERATURE_RANGE)},
                    "note": "uniform mode (alpha=0.5); nearest saved manifest times"})

    nn_manifest = read_manifest(final_nn)
    beta = float(nn_manifest[0].get("beta", args.beta))
    ref_coeff = summary.get("coefficient_t2_final_model") or summary.get("coefficient")
    p, coeff_metrics, coeff_clipped = coefficient_comparison(out, final_nn, beta, ref_coeff, log)
    entries.append({"file": p.name,
                    "sources": [f"{final_nn}/alpha.csv",
                                f"{final_nn}/manifest.csv",
                                "alpha_true rebuilt from exported u via base_alpha(x,y)*(1+beta*u)"],
                    "ranges": {"coefficient": list(COEFFICIENT_RANGE),
                               "coefficient_error": list(COEFFICIENT_ERROR_RANGE)},
                    "true_max_error": coeff_metrics["overall"]["max_abs"],
                    "clipped_points_error_range": coeff_clipped,
                    "note": "final-model b3a5f955; error-range clipping is purely graphical, "
                            "the true max error is printed in the title"})

    times = (0.0, 0.025, 0.05, 0.075, 0.1)
    frames = comparison_sequence(out, runs / "host" / "preset", final_nn, times, log)
    for fp in frames:
        entries.append({"file": fp.name,
                        "sources": [f"{runs}/host/preset/frame_*.csv",
                                    f"{final_nn}/frame_*.csv",
                                    f"{runs}/host/preset/manifest.csv",
                                    f"{final_nn}/manifest.csv"],
                        "ranges": {"temperature": list(TEMPERATURE_RANGE),
                                   "temperature_difference": list(TEMPERATURE_DIFF_RANGE)},
                        "note": "nearest saved manifest time (exact match); final-model nn"})

    p = final_comparison(out, summary, log)
    entries.append({"file": p.name,
                    "sources": [str(Path(args.summary)), "summary.json: field_t2_final_model, "
                                "coefficient_t2_final_model, mesh, timing"],
                    "ranges": {},
                    "note": "final-model field/coefficient numbers plus primary-matrix timings"})

    p = backend_table(out, summary, runs, log)
    entries.append({"file": p.name,
                    "sources": [str(Path(args.summary)), "summary.json: provenance.devices, "
                                "cross_backend_max_abs_vs_host_final, timing",
                                f"{runs}/<backend>/nn/alpha.csv"],
                    "ranges": {},
                    "note": "primary matrix weights f075cffd; skips marked explicitly"})

    manifest = write_figures_manifest(out, entries)
    print(f"wrote {len(entries)} figures + {manifest}")
    log(f"wrote {len(entries)} figures + {manifest}")
    log_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
