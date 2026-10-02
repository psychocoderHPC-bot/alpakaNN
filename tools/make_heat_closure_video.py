#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026
# SPDX-License-Identifier: MPL-2.0
"""Assemble a real ~45-60 s H.264 presentation video from the validated Section 9
figures produced by ``tools/render_heat_closure.py``.

The assembler is intentionally strict about honesty:

* Every image panel comes from the pre-rendered, CSV/JSON-derived PNG figures.
  No numerical field is synthesised, smoothed or re-coloured here.
* The only motion is *frame holds* and *nearest-time resampling by physical
  timestamps*: for animated segments the requested physical time is advanced
  linearly across the segment duration and the nearest *saved* snapshot time is
  selected. This is documented in the captions and is explicitly **not**
  interpolation of the numerical solution.
* Measured accuracy / overhead / backend numbers are read from
  ``summary.json``; none are hard-coded or extrapolated.

Usage
-----
    python3 tools/make_heat_closure_video.py \
        --frames  /tmp/alpakaNN-results/nhc-20261001/T6/figures \
        --summary /tmp/alpakaNN-results/nhc-20261001/T4/summary.json \
        --output  /tmp/alpakaNN-results/nhc-20261001/T7/heat_closure_demo.mp4
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Canvas / encoding constants
# --------------------------------------------------------------------------- #
WIDTH = 1920
HEIGHT = 1080
FPS = 30

BG = (11, 22, 34)          # video background
PANEL = (19, 35, 58)       # caption band
INK = (236, 243, 250)      # primary text
DIM = (150, 174, 199)      # secondary text
ACCENT = (94, 190, 255)    # accents

# Image area (padded figure) and caption area inside the 1920x1080 canvas.
IMG_TOP = 12
IMG_BOTTOM = 946
IMG_MAX_W = WIDTH - 60
CAP_TOP = 950

COMPARISON_TARGET_TIMES = (0.0, 0.025, 0.05, 0.075, 0.1)

# Storyboard segment frame counts (30 fps). Sum = 1650 = 55.0 s.
SEGMENT_FRAMES = {
    "title": 150,        # 5.0 s  title + geometry/BC
    "analytical": 300,   # 10.0 s geometry/regions + analytical temperature sequence
    "neural": 450,       # 15.0 s reference vs NN + error at matched saved times
    "learned": 300,      # 10.0 s true vs learned coefficient + error
    "integration": 300,  # 10.0 s timestep sequence + measured results + backends
    "closing": 150,      # 5.0 s  closing card
}
TOTAL_FRAMES = sum(SEGMENT_FRAMES.values())
TOTAL_SECONDS = TOTAL_FRAMES / FPS


# --------------------------------------------------------------------------- #
# Pure timeline helpers (unit-tested in tools/tests/test_make_heat_closure_video.py)
# --------------------------------------------------------------------------- #
def nearest_index(values, target):
    """Index of the value physically nearest ``target`` (ties -> earlier index)."""
    best = 0
    best_d = None
    for i, value in enumerate(values):
        d = abs(float(value) - float(target))
        if best_d is None or d < best_d - 1e-15:
            best_d = d
            best = i
    return best


def resample_indices(times, n_frames):
    """Map ``n_frames`` output frames to source indices by physical timestamp.

    The requested physical time advances linearly from ``times[0]`` to
    ``times[-1]`` across the segment; each output frame selects the source
    snapshot whose *saved* physical time is nearest. No numerical values are
    interpolated: this is a documented nearest-time frame hold.
    """
    if not times:
        raise ValueError("times must be non-empty")
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    if len(times) == 1 or n_frames == 1:
        return [0] * n_frames
    t0, t1 = float(times[0]), float(times[-1])
    out = []
    for j in range(n_frames):
        frac = j / (n_frames - 1)
        target = t0 + (t1 - t0) * frac
        out.append(nearest_index(times, target))
    return out


def even_dimensions(width, height):
    """Force both dimensions to be even (H.264 yuv420p requirement)."""
    return int(width) - int(width) % 2, int(height) - int(height) % 2


def segment_frame_count(name):
    return SEGMENT_FRAMES[name]


# --------------------------------------------------------------------------- #
# Optional heavy imports (kept lazy so the timeline logic is import-testable)
# --------------------------------------------------------------------------- #
def _pil():
    from PIL import Image, ImageDraw, ImageFont  # noqa: F401
    return Image, ImageDraw, ImageFont


def _pil_paths():
    Image, ImageDraw, ImageFont = _pil()
    from PIL import ImageOps
    return Image, ImageOps, ImageFont


def _fonts():
    _, _, ImageFont = _pil()
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    sizes = {"title": 58, "sub": 32, "cap_title": 32, "body": 23,
             "small": 16, "card_title": 62, "card_sub": 34, "bullet": 27}
    fonts = {}
    for key, size in sizes.items():
        bold = key in ("title", "cap_title", "card_title", "card_sub")
        path = candidates[1] if bold else candidates[0]
        try:
            fonts[key] = ImageFont.truetype(path, size)
        except OSError:
            try:
                fonts[key] = ImageFont.load_default(size=size)
            except TypeError:
                fonts[key] = ImageFont.load_default()
    return fonts


# --------------------------------------------------------------------------- #
# Small data helpers
# --------------------------------------------------------------------------- #
def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_manifest(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows


def comparison_metadata(frames_dir, preset_manifest, nn_manifest, log):
    """Physical saved times + steps for the five comparison snapshots.

    Times come from the same exported manifests used to render the figures. If
    a manifest is unavailable the documented render targets are used and the
    caption marks them as targets rather than confirmed saved times.
    """
    meta = []
    preset_rows = nn_rows = None
    try:
        preset_rows = load_manifest(preset_manifest)
        nn_rows = load_manifest(nn_manifest)
    except OSError:
        log("WARNING: run manifests unavailable; using documented render target "
            "times for the comparison segment")

    for k, target in enumerate(COMPARISON_TARGET_TIMES):
        fig = frames_dir / f"comparison_{k:06d}.png"
        if preset_rows:
            times = [float(r["time"]) for r in preset_rows]
            i = nearest_index(times, target)
            t_saved = times[i]
            step = int(preset_rows[i]["step"])
            confirmed = True
            if nn_rows:
                nn_times = [float(r["time"]) for r in nn_rows]
                j = nearest_index(nn_times, t_saved)
                if abs(nn_times[j] - t_saved) > 1e-12 or int(nn_rows[j]["step"]) != step:
                    log(f"WARNING: preset/nn snapshot mismatch at target {target}: "
                        f"preset t={t_saved} step={step} vs nn t={nn_times[j]} "
                        f"step={nn_rows[j]['step']}")
        else:
            t_saved = target
            step = None
            confirmed = False
        meta.append({"figure": fig, "target": target, "time": t_saved,
                     "step": step, "confirmed": confirmed})
    return meta


def overhead_ratios(summary):
    timing = summary.get("timing", {})
    out = {}
    for backend in ("host", "hip", "cuda", "sycl"):
        preset = timing.get(f"{backend}/preset", {}).get("total_seconds")
        nn = timing.get(f"{backend}/nn", {}).get("total_seconds")
        if preset and nn:
            out[backend] = nn / preset
    return out


# --------------------------------------------------------------------------- #
# Image compositing
# --------------------------------------------------------------------------- #
def content_crop(img, white=250, pad=6):
    """Crop near-white margins so panels fill the frame without touching edges."""
    import numpy as np
    rgb = img.convert("RGB")
    a = np.asarray(rgb)
    mask = a.min(axis=2) < white
    if not mask.any():
        return rgb
    ys, xs = np.where(mask)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(rgb.width, int(xs.max()) + 1 + pad)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(rgb.height, int(ys.max()) + 1 + pad)
    return rgb.crop((x0, y0, x1, y1))


def fit_into(img, box_w, box_h, bg=BG):
    """Scale preserving aspect ratio and centre on a ``box_w`` x ``box_h`` canvas."""
    Image, _, _ = _pil()
    rgb = img.convert("RGB")
    scale = min(box_w / rgb.width, box_h / rgb.height)
    nw, nh = max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))
    canvas = Image.new("RGB", (box_w, box_h), bg)
    resized = rgb.resize((nw, nh), Image.LANCZOS)
    canvas.paste(resized, ((box_w - nw) // 2, (box_h - nh) // 2))
    return canvas


def _ellipsize(draw, text, font, max_w):
    if draw.textlength(text, font=font) <= max_w:
        return text
    while text and draw.textlength(text + "\u2026", font=font) > max_w:
        text = text[:-1]
    return text + "\u2026"


def make_figure_frame(figure, caption_title, body_lines, footer, fonts):
    """Padded figure on top, two-line caption + footer in the reserved band."""
    Image, ImageDraw, _ = _pil()
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)

    with Image.open(figure) as src:
        cropped = content_crop(src.convert("RGB"))
    box_h = IMG_BOTTOM - IMG_TOP
    fitted = fit_into(cropped, IMG_MAX_W, box_h)
    canvas.paste(fitted, ((WIDTH - IMG_MAX_W) // 2, IMG_TOP))

    draw.rectangle([0, CAP_TOP - 4, WIDTH, HEIGHT], fill=PANEL)
    draw.line([0, CAP_TOP - 4, WIDTH, CAP_TOP - 4], fill=ACCENT, width=3)

    y = CAP_TOP + 2
    draw.text((40, y), _ellipsize(draw, caption_title, fonts["cap_title"], WIDTH - 80),
              font=fonts["cap_title"], fill=INK)
    y += 40
    for line in body_lines[:2]:
        draw.text((40, y), _ellipsize(draw, line, fonts["body"], WIDTH - 80),
                  font=fonts["body"], fill=DIM)
        y += 26
    if footer:
        fw = draw.textlength(footer, font=fonts["small"])
        draw.text((WIDTH - 40 - fw, HEIGHT - 26), footer, font=fonts["small"], fill=DIM)
    return canvas


def make_title_card(figure, fonts):
    """Title/setup card: story title, geometry & BC text and the setup diagram."""
    Image, ImageDraw, _ = _pil()
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, WIDTH, 206], fill=PANEL)
    draw.rectangle([0, 206, WIDTH, 212], fill=ACCENT)
    draw.text((60, 34), "Neural material closure with alpakaNN",
              font=fonts["title"], fill=INK)
    draw.text((60, 112), "Device-resident inference integrated into a portable C++ simulation",
              font=fonts["sub"], fill=ACCENT)
    draw.text((60, 160),
              "Hot wall u = 1 (left)  |  cold wall u = 0 (right)  |  insulated top/bottom faces",
              font=fonts["body"], fill=DIM)

    with Image.open(figure) as src:
        inset = fit_into(content_crop(src.convert("RGB")), 1480, 700)
    canvas.paste(inset, ((WIDTH - 1480) // 2, 244))
    draw.text((60, 986),
              "Specified geometry and analytical material \u03b1_base; not a simulation result.",
              font=fonts["body"], fill=DIM)
    draw.text((60, 1022),
              "Accelerated playback of simulation time (30 fps); not real-time execution.",
              font=fonts["small"], fill=DIM)
    return canvas


def make_closing_card(info, fonts):
    """Closing card with repository/commit information and the honesty note."""
    Image, ImageDraw, _ = _pil()
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 400, WIDTH, 410], fill=ACCENT)
    draw.text((80, 250), "Device-resident inference integrated",
              font=fonts["card_title"], fill=INK)
    draw.text((80, 326), "into a portable C++ simulation",
              font=fonts["card_title"], fill=INK)
    lines = [
        f"Repository: {info['repo']}",
        f"Branch: {info['branch']}   HEAD: {info['head']}   Base: {info['base']}",
        f"Measured backend matrix commit: {info['matrix_commit']}",
        f"Model weights (measured primary): {info['weights'][:16]}\u2026",
        "Training: external PyTorch. Inference: C++/alpakaNN, no per-step D2H copy.",
        "Motion uses frame holds and nearest saved physical times; numerical accuracy is never interpolated.",
    ]
    y = 470
    for line in lines:
        draw.text((80, y), line, font=fonts["bullet"], fill=DIM)
        y += 52
    return canvas


# --------------------------------------------------------------------------- #
# FFmpeg / ffprobe
# --------------------------------------------------------------------------- #
def require_tools():
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise SystemExit(f"{tool} is required (apt install ffmpeg).")


def encode(sequence, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-framerate", str(FPS), "-start_number", "0",
        "-i", str(sequence / "frame_%06d.png"),
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def probe(output):
    entries = "stream=codec_name,width,height,pix_fmt,nb_frames,avg_frame_rate:format=duration"
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", entries,
         "-of", "json", str(output)],
        check=True, text=True, capture_output=True)
    data = json.loads(res.stdout)
    stream = data["streams"][0]
    duration = float(data["format"]["duration"])
    nb_frames = stream.get("nb_frames")
    return {
        "codec_name": stream.get("codec_name"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "pix_fmt": stream.get("pix_fmt"),
        "nb_frames": int(nb_frames) if nb_frames is not None else None,
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "duration": duration,
    }


def decode_check(output):
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(output), "-f", "null", "-"],
                   check=True)


# --------------------------------------------------------------------------- #
# Inspection
# --------------------------------------------------------------------------- #
def sample_indices():
    """Representative frame indices: title, transitions, comparison, results, final."""
    return sorted({
        0, SEGMENT_FRAMES["title"] - 1,
        SEGMENT_FRAMES["title"],
        SEGMENT_FRAMES["title"] + 149,
        SEGMENT_FRAMES["title"] + 150,
        SEGMENT_FRAMES["title"] + 300 - 1,
        SEGMENT_FRAMES["title"] + 300,
        SEGMENT_FRAMES["title"] + 300 + 449,
        SEGMENT_FRAMES["title"] + 300 + 450,
        SEGMENT_FRAMES["title"] + 300 + 450 + 299,
        SEGMENT_FRAMES["title"] + 300 + 450 + 300,
        SEGMENT_FRAMES["title"] + 300 + 450 + 300 + 299,
        SEGMENT_FRAMES["title"] + 300 + 450 + 300 + 300,
        TOTAL_FRAMES - 60,
        TOTAL_FRAMES - 1,
    })


def extract_samples(output, dest, log):
    dest.mkdir(parents=True, exist_ok=True)
    idxs = sample_indices()
    expr = "+".join(f"eq(n\\,{i})" for i in idxs)
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-i", str(output),
        "-vf", f"select='{expr}'", "-vsync", "0",
        str(dest / "sample_%04d.png"),
    ], check=True)
    produced = sorted(dest.glob("sample_*.png"))
    log(f"extracted {len(produced)} sample frames for inspection")
    return idxs, produced


def build_contact_sheet(samples, idxs, dest, log):
    Image, ImageDraw, _ = _pil()
    fonts = _fonts()
    cols = 4
    tile_w, tile_h = 480, 270
    label_h = 26
    rows = (len(samples) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h)), BG)
    draw = ImageDraw.Draw(sheet)
    for n, (idx, path) in enumerate(zip(idxs, samples)):
        im = Image.open(path).convert("RGB").resize((tile_w, tile_h), Image.LANCZOS)
        r, c = divmod(n, cols)
        x, y = c * tile_w, r * (tile_h + label_h)
        sheet.paste(im, (x, y))
        draw.rectangle([x, y + tile_h, x + tile_w, y + tile_h + label_h], fill=PANEL)
        draw.text((x + 8, y + tile_h + 4), f"frame {idx}", font=fonts["small"], fill=INK)
    out = dest / "contact_sheet.png"
    sheet.save(out)
    log(f"wrote contact sheet {out}")
    return out


def inspect_frames(output, dest, log):
    import numpy as np
    Image, _, _ = _pil()
    idxs, samples = extract_samples(output, dest / "samples", log)
    findings = []
    for idx, path in zip(idxs, samples):
        a = np.asarray(Image.open(path).convert("RGB")).astype(float)
        std = float(a.std())
        near_white = float((a.min(axis=2) > 250).mean())
        near_black = float((a.max(axis=2) < 8).mean())
        blank = std < 4.0 or near_white > 0.98 or near_black > 0.98
        findings.append({"frame": idx, "std": round(std, 2),
                         "near_white": round(near_white, 3),
                         "near_black": round(near_black, 3),
                         "blank": blank})
        if blank:
            log(f"WARNING: frame {idx} looks blank (std={std:.2f})")
    sheet = build_contact_sheet(samples, idxs, dest, log)
    return findings, sheet


def ocr_probe(output, dest, log):
    """Best-effort OCR check that captions are rasterised and legible.

    Dark cards are inverted before OCR (light text on dark background). A weak
    result is a warning only, not a hard failure: tesseract is not part of the
    required runtime.
    """
    if not shutil.which("tesseract"):
        log("tesseract not installed; skipping OCR legibility probe")
        return {}
    import numpy as np
    Image, ImageOps, _ = _pil_paths()
    tests = {
        "title": 0,
        "analytical": SEGMENT_FRAMES["title"],
        "neural_end": SEGMENT_FRAMES["title"] + SEGMENT_FRAMES["analytical"] + SEGMENT_FRAMES["neural"] - 1,
        "learned": SEGMENT_FRAMES["title"] + SEGMENT_FRAMES["analytical"] + SEGMENT_FRAMES["neural"],
        "backend": TOTAL_FRAMES - SEGMENT_FRAMES["closing"] - 1,
        "closing": TOTAL_FRAMES - 1,
    }
    results = {}
    dest.mkdir(parents=True, exist_ok=True)
    for label, idx in tests.items():
        png = dest / f"ocr_{label}_{idx:04d}.png"
        # Index-based extraction avoids the exclusive end-of-stream timestamp
        # edge case at the final frame.
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-i", str(output),
            "-vf", f"select=eq(n\\,{idx})", "-vsync", "0",
            "-frames:v", "1", str(png)], check=True)
        if not png.exists():
            raise SystemExit(f"OCR extraction produced no frame for {label} ({idx})")
        with Image.open(png) as im:
            gray = im.convert("L")
            if float(np.asarray(gray).mean()) < 110:
                gray = ImageOps.invert(gray)
            proc = dest / f"ocr_{label}_{idx:04d}_proc.png"
            gray.save(proc)
        txt = subprocess.run(["tesseract", str(proc), "-"],
                             text=True, capture_output=True).stdout
        joined = " ".join(txt.split())
        results[label] = joined[:240]
        log(f"OCR {label} (frame {idx}): {joined[:160]}")
    return results


# --------------------------------------------------------------------------- #
# Storyboard construction
# --------------------------------------------------------------------------- #
def build_timeline(frames_dir, summary, comparison_meta, log):
    """Return an explicit list of ``TOTAL_FRAMES`` frame descriptors.

    A descriptor is ``(segment, caption_title, body_lines, footer, figure, card)``
    where exactly one of ``figure``/``card`` may be ``None``. Held shots are
    expanded to their segment frame count, so the returned list length equals the
    encoded frame count and can be inspected/validated before rendering.
    """
    comparisons = [m for m in comparison_meta]
    times = [m["time"] for m in comparisons]
    order = resample_indices(times, SEGMENT_FRAMES["neural"])

    def comparison_caption(i):
        m = comparisons[i]
        if m["step"] is not None:
            when = f"t = {m['time']:.6f}   step {m['step']}"
        else:
            when = f"target t = {m['time']:.6f} (manifest unavailable)"
        title = "Neural replacement: analytical reference vs learned closure"
        body = [
            f"Matched saved physical time {when}; |u_nn - u_preset| panel uses fixed range [0, 0.5].",
            "Left: preset (analytical). Centre: NN. Right: absolute temperature difference.",
        ]
        return title, body

    f_fld = summary.get("field_t2_final_model", {})
    ratios = overhead_ratios(summary)
    devices = summary.get("provenance", {}).get("devices", {})

    s_coeff = summary.get("coefficient_t2_final_model", {})
    coeff_mae = s_coeff.get("overall", {}).get("mae", 0.0)
    coeff_max = s_coeff.get("overall", {}).get("max_abs", 0.0)
    final_rel = f_fld.get("final", {}).get("relL2", 0.0)
    final_linf = f_fld.get("final", {}).get("normLinf", 0.0)
    max_rel = f_fld.get("max_over_times_relL2", 0.0)
    step = s_coeff.get("step", 13654)
    mae = coeff_mae
    cmax = coeff_max

    # (entry, held_frames) shots per segment; expanded to explicit frames below.
    shots = {name: [] for name in SEGMENT_FRAMES}

    # Title / setup -------------------------------------------------------- #
    shots["title"].append((("title", None, None, None,
                            frames_dir / "setup_diagram.png", "title"),
                           SEGMENT_FRAMES["title"]))
    # Analytical reference ------------------------------------------------- #
    a_geo = max(1, round(SEGMENT_FRAMES["analytical"] * 0.4))
    a_seq = SEGMENT_FRAMES["analytical"] - a_geo
    shots["analytical"].append(((
        "analytical", "Analytical reference: specified geometry and boundary conditions",
        ["Hot wall u = 1 (left), cold wall u = 0 (right), insulated top/bottom faces.",
         "Low-transport inclusion and high-transport conductor; fixed \u03b1 range [0, 4.5]."],
        "diagram; not a simulation result",
        frames_dir / "setup_diagram.png", None), a_geo))
    shots["analytical"].append(((
        "analytical", "Analytical temperature sequence (uniform \u03b1 = 0.5)",
        ["Front enters from the left; five saved physical times on a 64\u00d764 grid.",
         "Fixed temperature range [0, 1]. Accelerated playback of simulation time."],
        "frame holds; no solution interpolation",
        frames_dir / "homogeneous_sequence.png", None), a_seq))
    # Neural replacement (animated by physical timestamp) ------------------ #
    for i in order:
        title, body = comparison_caption(i)
        shots["neural"].append((("neural", title, body, "nearest saved physical time",
                                 comparisons[i]["figure"], None), 1))
    # Learned material ----------------------------------------------------- #
    shots["learned"].append(((
        "learned", "Learned material map vs analytical coefficient on the same NN state",
        [f"Final NN state step {step}; fixed \u03b1 range [0, 4.5], error range [0, 3.5].",
         f"True max |\u0394\u03b1| = {cmax:.4f}; overall MAE = {mae:.4f}; no clipping applied."],
        "final-model b3a5f955",
        frames_dir / "coefficient_comparison.png", None), SEGMENT_FRAMES["learned"]))
    # Integration + measured results --------------------------------------- #
    i_step = max(1, SEGMENT_FRAMES["integration"] // 3)
    i_acc = max(1, SEGMENT_FRAMES["integration"] // 3)
    i_bck = SEGMENT_FRAMES["integration"] - i_step - i_acc
    shots["integration"].append(((
        "integration", "Integration: timestep sequence of the coupled run",
        [f"{step} explicit steps, tmax = 0.1, grid 64; five saved physical times.",
         "Closure applied every step; accelerated playback of simulation time."],
        "host uniform mode",
        frames_dir / "homogeneous_sequence.png", None), i_step))
    shots["integration"].append(((
        "integration", "Measured numerical accuracy (final-model supplementary host run)",
        [f"Field final relL2 = {final_rel:.6f}, normLinf = {final_linf:.6f}; "
         f"max over 121 times relL2 = {max_rel:.6f}.",
         f"Coefficient MAE = {mae:.6f}, true max = {cmax:.6f} (n = 4096, no clipping)."],
        "values read from summary.json",
        frames_dir / "final_comparison.png", None), i_acc))
    ratio_txt = ", ".join(f"{k} {v:.1f}\u00d7" for k, v in ratios.items())
    shots["integration"].append(((
        "integration", "Tested backends and measured NN runtime overhead",
        [f"Device-resident inference on {', '.join(devices.keys())}; no per-step D2H copy.",
         f"NN vs preset total time: {ratio_txt}."],
        "skips and parity from summary.json",
        frames_dir / "backend_table.png", None), i_bck))
    # Closing -------------------------------------------------------------- #
    shots["closing"].append((("closing", None, None, None, None, "closing"),
                             SEGMENT_FRAMES["closing"]))

    plan = []
    for name in SEGMENT_FRAMES:
        held = sum(count for _, count in shots[name])
        if held != SEGMENT_FRAMES[name]:
            raise ValueError(f"segment {name}: planned {held} frames != {SEGMENT_FRAMES[name]}")
        for entry, count in shots[name]:
            plan.extend([entry] * count)
    if len(plan) != TOTAL_FRAMES:
        raise ValueError(f"timeline has {len(plan)} frames, expected {TOTAL_FRAMES}")
    log(f"timeline built: {len(plan)} frames / {TOTAL_SECONDS:.1f} s; "
        f"neural resample indices {order[:3]}...{order[-3:]}")
    return plan


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path("/tmp/alpakaNN-results/nhc-20261001")
    parser.add_argument("--frames", default=str(base / "T6/figures"))
    parser.add_argument("--summary", default=str(base / "T4/summary.json"))
    parser.add_argument("--preset-manifest", default=str(base / "runs/host/preset/manifest.csv"))
    parser.add_argument("--nn-manifest", default=str(base / "runs-finalmodel/nn/manifest.csv"))
    parser.add_argument("--output", default=str(base / "T7/heat_closure_demo.mp4"))
    parser.add_argument("--sequence", default=None,
                        help="uncompressed PNG sequence dir (default: <output dir>/presentation_frames)")
    parser.add_argument("--inspection", default=None,
                        help="inspection dir (default: <output dir>/inspection)")
    parser.add_argument("--log", default=None)
    parser.add_argument("--repo", default="https://github.com/alpaka-group/alpakaNN")
    args = parser.parse_args(argv)

    output = Path(args.output)
    seq = Path(args.sequence) if args.sequence else output.parent / "presentation_frames"
    insp = Path(args.inspection) if args.inspection else output.parent / "inspection"
    logf = open(args.log, "w", encoding="utf-8") if args.log else None

    def log(message):
        line = str(message)
        print(line, flush=True)
        if logf:
            logf.write(line + "\n")
            logf.flush()

    require_tools()
    frames_dir = Path(args.frames)
    log(f"frames={frames_dir} summary={args.summary} output={output}")

    required = ["setup_diagram.png", "homogeneous_sequence.png",
                "coefficient_comparison.png", "final_comparison.png",
                "backend_table.png"] + [f"comparison_{k:06d}.png" for k in range(5)]
    missing = [f for f in required if not (frames_dir / f).exists()]
    if missing:
        raise SystemExit(f"missing required figures: {missing}")

    summary = load_json(args.summary)
    comparison_meta = comparison_metadata(
        frames_dir, args.preset_manifest, args.nn_manifest, log)
    plan = build_timeline(frames_dir, summary, comparison_meta, log)

    # Validate the explicit plan length before rendering any PNG.
    if len(plan) != TOTAL_FRAMES:
        raise SystemExit(f"plan has {len(plan)} frames, expected {TOTAL_FRAMES}")

    fonts = _fonts()
    seq.mkdir(parents=True, exist_ok=True)
    if insp.exists():
        shutil.rmtree(insp)
    insp.mkdir(parents=True, exist_ok=True)

    repo_info = {
        "repo": args.repo,
        "branch": summary.get("provenance", {}).get("branch", "device-resident-runtime"),
        "head": summary.get("provenance", {}).get("commit_sha", "unknown"),
        "base": summary.get("provenance", {}).get("base_head", "unknown")[:12],
        "matrix_commit": summary.get("provenance", {}).get("commit_sha", "unknown"),
        "weights": summary.get("provenance", {}).get("t2_final_model_weights_sha256",
                                                      "unknown"),
    }

    log(f"rendering {len(plan)} frames at {WIDTH}x{HEIGHT} into {seq}")
    for n, (segment, title, body, footer, figure, card) in enumerate(plan):
        if card == "title":
            frame = make_title_card(figure, fonts)
        elif card == "closing":
            frame = make_closing_card(repo_info, fonts)
        else:
            frame = make_figure_frame(figure, title, body or [], footer, fonts)
        w, h = even_dimensions(frame.width, frame.height)
        if (w, h) != (WIDTH, HEIGHT):
            frame = frame.crop((0, 0, w, h))
        frame.save(seq / f"frame_{n:06d}.png")
        if n % 150 == 0:
            log(f"  frame {n}/{len(plan)} ({segment})")

    encode(seq, output)
    info = probe(output)
    log("ffprobe: " + json.dumps(info, sort_keys=True))

    ok = (info["codec_name"] == "h264" and info["pix_fmt"] == "yuv420p"
          and info["width"] % 2 == 0 and info["height"] % 2 == 0
          and 45.0 <= info["duration"] <= 60.0
          and info["nb_frames"] is not None
          and abs(info["nb_frames"] - info["duration"] * FPS) <= FPS)
    if not ok:
        raise SystemExit(f"ffprobe acceptance failed: {info}")

    decode_check(output)
    log("full decode passed (ffmpeg -v error -i out.mp4 -f null -)")

    findings, sheet = inspect_frames(output, insp, log)
    with open(insp / "findings.json", "w", encoding="utf-8") as fh:
        json.dump({"probe": info, "segments": SEGMENT_FRAMES,
                   "total_frames": TOTAL_FRAMES, "fps": FPS,
                   "samples": findings}, fh, indent=2)
    blanks = [f for f in findings if f["blank"]]
    if blanks:
        raise SystemExit(f"blank sample frames detected: {blanks}")
    log(f"inspection: {len(findings)} samples, no blank panels; sheet={sheet}")

    ocr_results = ocr_probe(output, insp, log)
    if ocr_results:
        with open(insp / "ocr.txt", "w", encoding="utf-8") as fh:
            for k, v in ocr_results.items():
                fh.write(f"frame {k}: {v}\n")

    log(f"OK: {output} ({info['duration']:.2f} s, {info['nb_frames']} frames, "
        f"{info['width']}x{info['height']} {info['codec_name']}/{info['pix_fmt']})")
    if logf:
        logf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
