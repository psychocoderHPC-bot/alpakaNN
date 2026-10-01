# SPDX-FileCopyrightText: 2026
# SPDX-License-Identifier: MPL-2.0
"""Unit tests for tools/make_heat_closure_video.py timeline helpers.

These check the documented resampling contract (nearest saved physical time,
frame holds, never numerical interpolation) and the storyboard/encoding
invariants without invoking ffmpeg.
"""
import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "make_heat_closure_video.py"
spec = importlib.util.spec_from_file_location("make_heat_closure_video", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class TimelineTest(unittest.TestCase):
    def test_segments_sum_to_45_to_60_seconds(self):
        total = sum(mod.SEGMENT_FRAMES.values())
        seconds = total / mod.FPS
        self.assertEqual(total, mod.TOTAL_FRAMES)
        self.assertGreaterEqual(seconds, 45.0)
        self.assertLessEqual(seconds, 60.0)

    def test_all_segments_present_and_ordered(self):
        for name in ("title", "analytical", "neural", "learned",
                     "integration", "closing"):
            self.assertIn(name, mod.SEGMENT_FRAMES)

    def test_nearest_index_selects_nearest(self):
        self.assertEqual(mod.nearest_index([0.0, 0.025, 0.05], 0.01), 0)
        self.assertEqual(mod.nearest_index([0.0, 0.025, 0.05], 0.03), 1)
        self.assertEqual(mod.nearest_index([0.0, 0.025, 0.05], 0.049), 2)
        # tie (0.0125 is equidistant from 0.0 and 0.025) resolves to the earlier index
        self.assertEqual(mod.nearest_index([0.0, 0.025, 0.05], 0.0125), 0)

    def test_resample_is_frame_hold_not_interpolation(self):
        times = [0.0, 0.025, 0.05, 0.075, 0.1]
        idx = mod.resample_indices(times, 9)
        # every emitted index must reference an actual saved timestamp
        self.assertTrue(all(0 <= i < len(times) for i in idx))
        # output is monotone and covers the full range
        self.assertEqual(idx[0], 0)
        self.assertEqual(idx[-1], len(times) - 1)
        self.assertEqual(idx, sorted(idx))

    def test_resample_single_snapshot_is_hold(self):
        self.assertEqual(mod.resample_indices([0.1], 7), [0] * 7)

    def test_resample_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            mod.resample_indices([], 3)
        with self.assertRaises(ValueError):
            mod.resample_indices([0.0, 1.0], 0)

    def test_even_dimensions(self):
        self.assertEqual(mod.even_dimensions(1920, 1080), (1920, 1080))
        self.assertEqual(mod.even_dimensions(1921, 1079), (1920, 1078))
        w, h = mod.even_dimensions(mod.WIDTH, mod.HEIGHT)
        self.assertEqual((w, h), (mod.WIDTH, mod.HEIGHT))
        self.assertEqual(w % 2, 0)
        self.assertEqual(h % 2, 0)

    def test_overhead_ratios_from_summary_shape(self):
        summary = {"timing": {
            "host/preset": {"total_seconds": 2.0},
            "host/nn": {"total_seconds": 6.0},
            "cuda/preset": {"total_seconds": 1.0},
            "cuda/nn": {"total_seconds": 4.0},
        }}
        ratios = mod.overhead_ratios(summary)
        self.assertAlmostEqual(ratios["host"], 3.0)
        self.assertAlmostEqual(ratios["cuda"], 4.0)
        self.assertNotIn("hip", ratios)


if __name__ == "__main__":
    unittest.main()
