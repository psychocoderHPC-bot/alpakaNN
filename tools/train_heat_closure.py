#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026
# SPDX-License-Identifier: MPL-2.0
"""Generate deterministic heat-closure datasets and train/export the matching gated-SiLU model.

CSV rows contain [u,x,y,alpha_true,split]. Model binary arrays are contiguous
little-endian float32 row-major [in,out], in gate/up/down order; JSON records shapes.
PyTorch is optional for dataset generation and metadata/model serialization.

``train --variant raw`` exports the v1 raw ``[u,x,y]`` contract
(``alpakaNN-heat-closure-f32-v1``); ``train --variant B`` exports the v2
Fourier variant-B contract (``alpakaNN-heat-closure-f32-v2``,
``feature_encoding = fourier_xy_k0_5``, 27 inputs). The C++ loader selects the
contract from the metadata, so one binary format serves both.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import struct
import sys
import time
from pathlib import Path

ALPHA_MIN, ALPHA_MAX = 0.01, 6.0
BETA_DEFAULT = 0.5
WIDTH = 64
KMAX = 5  # variant-B Fourier band: k = 0..5 -> 4*(KMAX+1) extra columns
FORMULA = "base(x,y) * (1 + beta*u); inclusion if r2 < 0.12^2 => 0.02; else conductor if abs(y-0.5)<0.05 and 0.45<x<0.65 => 4.0; else 0.5+0.4*H(sin(6*pi*y)), H(z)=1 iff z>=0"
ARCH = "gated_silu_bias_free_v1"
FORMAT_V1 = "alpakaNN-heat-closure-f32-v1"
FORMAT_V2 = "alpakaNN-heat-closure-f32-v2"
FEATURE_ENCODING_V2 = "fourier_xy_k0_5"
WEIGHT_LAYOUT = "row-major [in,out], little-endian float32; gate,up,down concatenated"
OUTPUT_STRING = "alpha_min + (alpha_max - alpha_min) * sigmoid(z)"


def variant_contract(variant: str) -> tuple[int, list[str], str, str | None]:
    """Return (input_dim, feature_order, format, feature_encoding) for a variant.

    ``raw`` is the original v1 ``[u,x,y]`` contract; ``B`` is the Fourier
    variant-B contract with the same extra columns the C++ runtime emits.
    """
    if variant == "raw":
        return 3, ["u", "x", "y"], FORMAT_V1, None
    if variant != "B":
        raise ValueError(f"unknown variant {variant!r}")
    order = ["u", "x", "y"]
    for k in range(KMAX + 1):
        p = 2 ** k
        order += [f"sin{p}pi_x", f"cos{p}pi_x", f"sin{p}pi_y", f"cos{p}pi_y"]
    return len(order), order, FORMAT_V2, FEATURE_ENCODING_V2


def encode_features(u, x, y, variant: str):
    """Build the model input matrix matching the C++ ``PackFeatures`` order."""
    row = [u, x, y]
    if variant == "B":
        for k in range(KMAX + 1):
            w = (2.0 ** k) * math.pi
            row += [math.sin(w * x), math.cos(w * x), math.sin(w * y), math.cos(w * y)]
    return row


def alpha_true(u: float, x: float, y: float, beta: float = BETA_DEFAULT) -> float:
    if (x - 0.35) ** 2 + (y - 0.5) ** 2 < 0.12 ** 2:
        base = 0.02
    elif abs(y - 0.5) < 0.05 and 0.45 < x < 0.65:
        base = 4.0
    else:
        base = 0.9 if math.sin(6.0 * math.pi * y) >= 0.0 else 0.5
    return base * (1.0 + beta * u)


def alpha_max_for_beta(beta: float) -> float:
    return max(ALPHA_MAX, 4.0 * (1.0 + beta))


def metadata(beta: float, variant: str = "raw", width: int = WIDTH, **extra) -> dict:
    in_dim, order, fmt, encoding = variant_contract(variant)
    upper = alpha_max_for_beta(beta)
    result = {
        "format": fmt, "architecture": ARCH,
        "width": width, "feature_order": order,
        "weight_layout": WEIGHT_LAYOUT,
        "weight_shapes": [[in_dim, width], [in_dim, width], [width, 1]],
        "output": OUTPUT_STRING,
        "alpha_min": ALPHA_MIN, "alpha_max": upper,
        "beta": beta, "beta_mismatch_policy": "reject",
        "material_formula": FORMULA, "coordinate_domain": [0.0, 1.0],
        "temperature_domain": [0.0, 1.0], "dtype": "float32",
    }
    if encoding is not None:
        result["feature_encoding"] = encoding
    result.update(extra)
    return result


def acceptance_report(metrics: dict, beta: float) -> dict:
    """Summarize acceptance for the evaluated splits.

    Targets match the tracked v2 manifest: MAE <= 0.05*alpha_max and
    max_abs <= 0.20*alpha_max. Validation and the coordinate-held-out
    ``clean_holdout`` split decide the verdict; ``test`` is informational.
    """
    targets = {"mae": 0.05 * alpha_max_for_beta(beta), "max_abs": 0.20 * alpha_max_for_beta(beta)}
    splits = {}
    decision_splits = ("validation", "clean_holdout")
    for name, split in metrics.items():
        if not isinstance(split, dict) or "mae" not in split or "max_abs" not in split:
            continue
        mae_pass = split["mae"] <= targets["mae"]
        max_abs_pass = split["max_abs"] <= targets["max_abs"]
        evaluated = name in decision_splits
        splits[name] = {
            "evaluated": evaluated,
            "mae": split["mae"],
            "max_abs": split["max_abs"],
            "mse": split.get("mse"),
            "mae_pass": mae_pass,
            "max_abs_pass": max_abs_pass,
            "pass": bool(mae_pass and max_abs_pass) if evaluated else None,
        }
    decided = [s for s in splits.values() if s["evaluated"]]
    verdict = "PASS" if decided and all(s["pass"] for s in decided) else "FAIL"
    return {
        "basis": "MAE <= 0.05*alpha_max and max_abs <= 0.20*alpha_max on validation and clean holdout "
                 "(test informational)",
        "targets": targets,
        "splits": splits,
        "verdict": verdict,
    }


def trained_manifest(
    beta: float,
    variant: str,
    metrics: dict,
    weights_sha256: str,
    training: dict,
    weights_file: str,
    history_file: str,
) -> dict:
    """Build the exported model manifest for a trained checkpoint.

    Pure (no PyTorch) so the exported contract can be tested directly. The C++
    loader selects the feature contract from ``format``/``feature_encoding`` and
    validates ``weight_shapes``/``width``; the ``metrics``/``acceptance`` blocks
    are informational evidence and do not affect loading.
    """
    result = metadata(
        beta,
        variant,
        kind="trained_model",
        weights_file=weights_file,
        weights_sha256=weights_sha256,
        training=training,
        metrics=metrics,
        history_file=history_file,
    )
    result["dtype"] = "float32"
    if variant == "B":
        result["feature_count"] = len(result["feature_order"])
        result["bias_free"] = True
        result["acceptance"] = acceptance_report(metrics, beta)
    return result


def encode_weights(gate, up, down, in_dim: int = 3, width: int = WIDTH) -> bytes:
    """Serialize nested [in,out] rows in fixed gate/up/down order."""
    chunks = []
    for rows, expected in ((gate, (in_dim, width)), (up, (in_dim, width)), (down, (width, 1))):
        if len(rows) != expected[0] or any(len(row) != expected[1] for row in rows):
            raise ValueError(f"weight array must have shape {expected}")
        chunks.extend(struct.pack("<f", float(value)) for row in rows for value in row)
    return b"".join(chunks)


def decode_weights(payload: bytes, in_dim: int = 3, width: int = WIDTH):
    """Read the exact three row-major little-endian arrays from the binary format."""
    count = (in_dim * width) + (in_dim * width) + width
    if len(payload) != count * 4:
        raise ValueError(f"expected {count * 4} bytes, got {len(payload)}")
    values = struct.unpack("<" + "f" * count, payload)
    offset = 0
    result = []
    for rows, cols in ((in_dim, width), (in_dim, width), (width, 1)):
        arr = [list(values[offset + r * cols:offset + (r + 1) * cols]) for r in range(rows)]
        result.append(arr); offset += rows * cols
    return tuple(result)


def write_metadata(path: Path, beta: float, **extra) -> None:
    variant = extra.pop("variant", "raw")
    width = extra.pop("width", WIDTH)
    path.write_text(json.dumps(metadata(beta, variant, width, **extra), indent=2, sort_keys=True) + "\n")


def dataset(args) -> None:
    rng = random.Random(args.seed)
    rows: list[tuple[float, float, float, str]] = []
    # Stratified regular grid covers complete domain and temperature interval.
    side = max(2, int(math.sqrt(args.grid_samples)))
    temps = max(2, args.temperature_levels)
    for ti in range(temps):
        u = ti / (temps - 1)
        for iy in range(side):
            y = (iy + 0.5) / side
            for ix in range(side):
                x = (ix + 0.5) / side
                rows.append((u, x, y, "grid"))
    # Uniform random samples, with independent temperature draws.
    for _ in range(args.random_samples):
        rows.append((rng.random(), rng.random(), rng.random(), "random"))
    # Oversample both sides of each geometric interface; epsilon avoids ambiguity.
    eps = args.interface_epsilon
    for _ in range(args.interface_samples):
        u = rng.random()
        angle = rng.random() * 2 * math.pi
        radius = 0.12 + (-eps if rng.random() < 0.5 else eps)
        x, y = 0.35 + radius * math.cos(angle), 0.5 + radius * math.sin(angle)
        rows.append((u, min(1., max(0., x)), min(1., max(0., y)), "interface_inclusion"))
        x = 0.45 + rng.random() * 0.20
        y = 0.45 + (-eps if rng.random() < 0.5 else eps)
        rows.append((u, x, y, "interface_conductor_y"))
        y = (rng.randrange(1, 6) / 6.0) + (-eps if rng.random() < 0.5 else eps)
        rows.append((u, rng.random(), min(1., max(0., y)), "interface_stripe"))
    rng.shuffle(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["u", "x", "y", "alpha_true", "split", "sampling_source"])
        for u, x, y, source in rows:
            # Spatially shifted evaluation held out completely by coordinate.
            if x >= args.spatial_holdout:
                split = "spatial_eval"
            else:
                draw = rng.random()
                split = "train" if draw < .8 else ("validation" if draw < .9 else "test")
            w.writerow([f"{u:.9g}", f"{x:.9g}", f"{y:.9g}", f"{alpha_true(u,x,y,args.beta):.9g}", split, source])
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    write_metadata(output.with_suffix(output.suffix + ".metadata.json"), args.beta,
                   kind="dataset", csv_sha256=digest, csv_columns=["u", "x", "y", "alpha_true", "split", "sampling_source"],
                   seed=args.seed, grid_samples_per_temperature=side * side,
                   temperature_levels=temps, random_samples=args.random_samples,
                   interface_samples_per_interface=args.interface_samples,
                   interface_epsilon=args.interface_epsilon, spatial_holdout_x_min=args.spatial_holdout,
                   split_fractions="random 80/10/10 for x < spatial_holdout; x >= threshold reserved spatial_eval",
                   row_count=len(rows))
    print(f"wrote {len(rows)} samples to {output} (sha256 {digest})")


def _torch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise RuntimeError("training requires PyTorch; install a CPU build (dataset/metadata commands do not)") from e


def train(args) -> None:
    torch = _torch()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    data = []
    with open(args.csv, newline="") as f:
        data = list(csv.DictReader(f))
    in_dim, _, _, _ = variant_contract(args.variant)
    def tensors(split):
        subset = [r for r in data if r["split"] == split]
        if not subset: raise ValueError(f"empty {split} dataset")
        raw = torch.tensor([[float(r[k]) for k in ("u", "x", "y")] for r in subset], dtype=torch.float32)
        if args.variant == "B":
            cols = [raw]
            for k in range(KMAX + 1):
                w = (2.0 ** k) * math.pi
                cols += [torch.sin(w * raw[:, 1:2]), torch.cos(w * raw[:, 1:2]),
                         torch.sin(w * raw[:, 2:3]), torch.cos(w * raw[:, 2:3])]
            x = torch.cat(cols, dim=1)
        else:
            x = raw
        y = torch.tensor([float(r["alpha_true"]) for r in subset], dtype=torch.float32).view(-1, 1)
        return x, y
    trX,trY=tensors("train"); vaX,vaY=tensors("validation")
    # Never use the final test split for checkpoint selection.
    test_pair=None if args.skip_test else tensors("test")
    shiftX,shiftY=tensors("spatial_eval")
    gate=torch.nn.Parameter(torch.empty(in_dim,WIDTH)); up=torch.nn.Parameter(torch.empty(in_dim,WIDTH)); down=torch.nn.Parameter(torch.empty(WIDTH,1))
    torch.nn.init.xavier_uniform_(gate); torch.nn.init.xavier_uniform_(up); torch.nn.init.xavier_uniform_(down)
    params=[gate,up,down]; opt=torch.optim.Adam(params, lr=args.lr)
    def forward(x):
        z=(torch.nn.functional.silu(x @ gate) * (x @ up)) @ down
        upper=alpha_max_for_beta(args.beta)
        return ALPHA_MIN + (upper-ALPHA_MIN)*torch.sigmoid(z)
    best=float("inf"); best_state=None; best_epoch=0; history=[]; start=time.time()
    for epoch in range(1,args.epochs+1):
        perm=torch.randperm(len(trX))
        for ids in perm.split(args.batch_size):
            pred=forward(trX[ids]); loss=torch.mean((pred-trY[ids])**2)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            val=float(torch.mean((forward(vaX)-vaY)**2))
        history.append({"epoch":epoch,"validation_mse":val})
        if val < best:
            best=val; best_epoch=epoch; best_state=[p.detach().clone() for p in params]
    for p,v in zip(params,best_state): p.data.copy_(v)
    def metrics(x,y):
        with torch.no_grad():
            e=(forward(x)-y).abs()
            return {"count":len(x),"mse":float(torch.mean(e**2)),"mae":float(torch.mean(e)),"max_abs":float(torch.max(e))}
    def regional_metrics(x,y):
        u,xcoord,ycoord=x[:,0],x[:,1],x[:,2]
        inclusion=(xcoord-.35).square()+(ycoord-.5).square() < .12**2
        conductor=(~inclusion)&((ycoord-.5).abs()<.05)&(xcoord>.45)&(xcoord<.65)
        stripe=(~inclusion)&(~conductor)&(torch.sin(6.0*math.pi*ycoord)>=0)
        regions={"inclusion":inclusion,"conductor":conductor,"stripe_high":stripe,"background_low":~(inclusion|conductor|stripe)}
        with torch.no_grad():
            err=(forward(x)-y).abs()
            return {name:{"count":int(mask.sum()),"mse":float((err[mask]**2).mean()),"mae":float(err[mask].mean()),"max_abs":float(err[mask].max())} for name,mask in regions.items() if mask.any()}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    arrays=[p.detach().cpu().tolist() for p in params]
    out.write_bytes(encode_weights(*arrays, in_dim=in_dim, width=WIDTH))
    # The coordinate-held-out `spatial_eval` split is recorded as `clean_holdout`
    # so the acceptance block matches the tracked v2 manifest contract.
    model_metrics={"validation":metrics(vaX,vaY),"validation_by_region":regional_metrics(vaX,vaY),
                   "clean_holdout":metrics(shiftX,shiftY),"clean_holdout_by_region":regional_metrics(shiftX,shiftY)}
    if test_pair is not None:
        model_metrics["test"] = metrics(*test_pair)
        model_metrics["test_by_region"] = regional_metrics(*test_pair)
    m=trained_manifest(
      args.beta, args.variant, model_metrics,
      hashlib.sha256(out.read_bytes()).hexdigest(),
      {"seed":args.seed,"epochs_requested":args.epochs,"epochs_run":args.epochs,"best_epoch":best_epoch,"best_validation_mse":best,"learning_rate":args.lr,"batch_size":args.batch_size,"optimizer":"Adam","loss":"physical coefficient MSE","device":"cpu","duration_seconds":time.time()-start},
      out.name, out.with_suffix(".history.json").name)
    m["training_csv_sha256"]=hashlib.sha256(Path(args.csv).read_bytes()).hexdigest()
    out.with_suffix(out.suffix+".metadata.json").write_text(json.dumps(m,indent=2,sort_keys=True)+"\n")
    out.with_suffix(".history.json").write_text(json.dumps(history,indent=2)+"\n")
    print(json.dumps(m["metrics"],indent=2)); print(f"best epoch: {best_epoch}; weights: {out}; duration {m['training']['duration_seconds']:.3f}s")


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="command",required=True)
    d=sub.add_parser("dataset",help="generate CSV and dataset metadata"); d.add_argument("--output",default="heat_closure.csv"); d.add_argument("--beta",type=float,default=BETA_DEFAULT); d.add_argument("--seed",type=int,default=0); d.add_argument("--grid-samples",type=int,default=4096); d.add_argument("--temperature-levels",type=int,default=11); d.add_argument("--random-samples",type=int,default=20000); d.add_argument("--interface-samples",type=int,default=4000); d.add_argument("--interface-epsilon",type=float,default=1e-4); d.add_argument("--spatial-holdout",type=float,default=.9); d.set_defaults(func=dataset)
    t=sub.add_parser("train",help="train gated-SiLU closure on CPU and export model"); t.add_argument("--csv",required=True); t.add_argument("--output",default="models/heat_closure/weights.bin"); t.add_argument("--variant",choices=("raw","B"),default="raw",help="raw = v1 [u,x,y] contract; B = v2 Fourier variant-B contract"); t.add_argument("--beta",type=float,default=BETA_DEFAULT); t.add_argument("--seed",type=int,default=0); t.add_argument("--epochs",type=int,default=200); t.add_argument("--batch-size",type=int,default=4096); t.add_argument("--lr",type=float,default=1e-3); t.add_argument("--threads",type=int,default=1); t.add_argument("--skip-test",action="store_true",help="do not evaluate the final test split"); t.set_defaults(func=train)
    args=p.parse_args()
    if not math.isfinite(args.beta) or args.beta < 0: p.error("--beta must be finite and nonnegative")
    if args.command=="dataset" and (args.grid_samples<4 or args.temperature_levels<2 or args.random_samples<0 or args.interface_samples<0 or args.interface_epsilon<=0 or not 0<args.spatial_holdout<1): p.error("invalid dataset sampling options")
    if args.command=="train" and (args.epochs<1 or args.batch_size<1 or args.lr<=0 or args.threads<1): p.error("invalid training options")
    args.func(args)

if __name__ == "__main__":
    try: main()
    except RuntimeError as e: print(f"error: {e}",file=sys.stderr); sys.exit(2)
