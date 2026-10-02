# SPDX-FileCopyrightText: 2026
# SPDX-License-Identifier: MPL-2.0
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "train_heat_closure.py"
spec = importlib.util.spec_from_file_location("train_heat_closure", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class TrainingToolsTest(unittest.TestCase):
    def test_material_priority_and_temperature(self):
        self.assertAlmostEqual(mod.alpha_true(.0, .35, .5), .02)
        self.assertAlmostEqual(mod.alpha_true(1., .5, .5), 6.)
        self.assertAlmostEqual(mod.alpha_true(.5, .8, .3), .625)

    def test_metadata_and_binary_roundtrip_deterministic(self):
        gate=[[float(r*mod.WIDTH+c) for c in range(mod.WIDTH)] for r in range(3)]
        up=[[float(-r*mod.WIDTH-c) for c in range(mod.WIDTH)] for r in range(3)]
        down=[[float(r)] for r in range(mod.WIDTH)]
        blob=mod.encode_weights(gate,up,down)
        self.assertEqual(blob, mod.encode_weights(gate,up,down))
        self.assertEqual((gate,up,down), mod.decode_weights(blob))
        with self.assertRaises(ValueError): mod.decode_weights(blob[:-1])
        m=mod.metadata(.5)
        self.assertEqual(m["weight_shapes"], [[3,64],[3,64],[64,1]])
        self.assertEqual(m["beta_mismatch_policy"], "reject")
        self.assertNotIn("feature_encoding", m)

    def test_variant_b_metadata_and_binary_roundtrip(self):
        m=mod.metadata(.5, "B")
        self.assertEqual(m["format"], "alpakaNN-heat-closure-f32-v2")
        self.assertEqual(m["feature_encoding"], "fourier_xy_k0_5")
        self.assertEqual(len(m["feature_order"]), 27)
        self.assertEqual(m["weight_shapes"], [[27,64],[27,64],[64,1]])
        gate=[[float(r*27+c) for c in range(mod.WIDTH)] for r in range(27)]
        up=[[float(-r*27-c) for c in range(mod.WIDTH)] for r in range(27)]
        down=[[float(r)] for r in range(mod.WIDTH)]
        blob=mod.encode_weights(gate,up,down,in_dim=27)
        self.assertEqual(len(blob), 4*((27*64)+(27*64)+64))
        self.assertEqual((gate,up,down), mod.decode_weights(blob,in_dim=27))
        # Feature builder matches the documented column order for one point.
        row=mod.encode_features(1.0, 0.25, 0.75, "B")
        self.assertEqual(len(row), 27)
        self.assertAlmostEqual(row[0], 1.0)
        self.assertAlmostEqual(row[1], 0.25)
        self.assertAlmostEqual(row[2], 0.75)
        self.assertAlmostEqual(row[3], math.sin(math.pi*0.25))
        self.assertAlmostEqual(row[4], math.cos(math.pi*0.25))
        self.assertEqual(len(mod.encode_features(1.0, 0.25, 0.75, "raw")), 3)

    def test_small_csv_is_reproducible_and_spatial_split_disjoint(self):
        import argparse, csv
        args=argparse.Namespace(seed=5, grid_samples=16, temperature_levels=2,
            random_samples=12, interface_samples=4, interface_epsilon=1e-4,
            spatial_holdout=.9, beta=.5, output="")
        with tempfile.TemporaryDirectory() as td:
            a=Path(td)/"a.csv"; b=Path(td)/"b.csv"
            args.output=str(a); mod.dataset(args)
            args.output=str(b); mod.dataset(args)
            self.assertEqual(a.read_bytes(),b.read_bytes())
            with a.open(newline="") as f:
                rows=list(csv.DictReader(f))
            self.assertTrue(any(r["split"]=="spatial_eval" for r in rows))
            self.assertTrue(all(float(r["x"]) >= .9 for r in rows if r["split"]=="spatial_eval"))
            self.assertTrue(all(float(r["x"]) < .9 for r in rows if r["split"] in ("train","validation","test")))
            meta=json.loads((Path(str(a)+".metadata.json")).read_text())
            self.assertEqual(meta["row_count"],len(rows))

if __name__=="__main__": unittest.main()
