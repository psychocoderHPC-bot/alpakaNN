# SPDX-FileCopyrightText: 2026
# SPDX-License-Identifier: MPL-2.0
"""Unit tests for tools/render_heat_closure.py pure helpers."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "render_heat_closure.py"
spec = importlib.util.spec_from_file_location("render_heat_closure", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class RenderHelpersTest(unittest.TestCase):
    def test_base_alpha_matches_reference_formula(self):
        n = 64
        x = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(x, x)
        base = mod.base_alpha_grid(X, Y)
        # inclusion cell centre nearest (0.35, 0.5)
        ix = int(0.35 * n)
        iy = int(0.5 * n)
        self.assertAlmostEqual(base[iy, ix], 0.02)
        # conductor box centre (0.55, 0.5)
        self.assertAlmostEqual(base[iy, int(0.55 * n)], 4.0)
        # stripe formula elsewhere: values are in {0.5, 0.9}
        self.assertTrue(np.all(np.isin(np.round(base, 6), [0.02, 0.5, 0.9, 4.0])))

    def test_rebuild_true_alpha_uses_exported_u(self):
        n = 4
        frame = np.zeros((n, n, 6))
        x = (np.arange(n) + 0.5) / n
        frame[0, :, 0] = x  # x column
        frame[:, 0, 1] = x  # y column
        frame[:, :, 2] = 0.5
        true, X, Y = mod.rebuild_true_alpha(frame, beta=0.5)
        base = mod.base_alpha_grid(X, Y)
        np.testing.assert_allclose(true, base * (1.0 + 0.5 * 0.5))

    def test_region_masks_disjoint_and_priority(self):
        n = 64
        x = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(x, x)
        masks = mod.region_masks(X, Y)
        stack = np.stack(list(masks.values()), axis=0).astype(int)
        self.assertTrue(np.all(stack.sum(axis=0) <= 1))
        # inclusion wins over the overlapping conductor box
        inclusion = masks["inclusion"]
        conductor = masks["conductor"]
        self.assertTrue(np.all((inclusion & conductor) == 0))

    def test_alpha_error_metrics_zero_for_identical(self):
        n = 8
        x = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(x, x)
        a = mod.base_alpha_grid(X, Y)
        metrics, err = mod.alpha_error_metrics(a, a, X, Y)
        self.assertEqual(metrics["overall"]["max_abs"], 0.0)
        self.assertEqual(metrics["overall"]["mae"], 0.0)


if __name__ == "__main__":
    unittest.main()
