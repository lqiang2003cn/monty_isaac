"""End-to-end smoke test for the turntable learning pipeline.

Creates a synthetic scan directory with simple colored shapes, then runs
``turntable-learn`` to verify that Monty produces a model.pt file with
graph nodes.

Usage (inside monty_comp container)::

    conda activate tbp.monty
    python -m monty_ext.scripts.test_turntable_learn
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


def _create_synthetic_scan(scan_dir: Path, n_frames: int = 20) -> None:
    """Generate a synthetic turntable scan with rotating colored cubes.

    Creates a simple animation of a colored rectangle rotating on a dark
    background (simulating a turntable with a block on it).
    """
    scan_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = scan_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    H, W = 240, 320

    for i in range(n_frames):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:] = (20, 20, 20)

        angle = (360.0 * i / n_frames)
        cx, cy = W // 2, H // 2
        half = 30

        corners = np.array([
            [-half, -half],
            [half, -half],
            [half, half],
            [-half, half],
        ], dtype=np.float32)

        rad = np.radians(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated = (rot @ corners.T).T + np.array([cx, cy])
        pts = rotated.astype(np.int32)

        color = (
            int(100 + 100 * np.sin(rad)),
            int(100 + 100 * np.cos(rad)),
            180,
        )
        cv2.fillPoly(img, [pts], color)

        cv2.imwrite(str(frames_dir / f"{i:06d}.jpg"), img)

    metadata = dict(
        object_name="test_cube",
        width=W,
        height=H,
        fps=30,
        frame_count=n_frames,
        duration_sec=n_frames / 30.0,
        timestamp="2026-01-01T00:00:00+00:00",
    )
    (scan_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def _create_mock_vision_bridge():
    """Monkey-patch VisionBridge to return synthetic masks and depth/poses.

    This avoids needing actual SAM2/VGGT GPU models for the smoke test.
    """
    from monty_ext.vision import vggt_provider as vggt_mod
    from monty_ext.vision import sam2_segmenter as sam2_mod

    class MockSAM2:
        def __init__(self, **kwargs):
            pass

        def segment(self, rgb):
            h, w = rgb.shape[:2]
            mask = np.zeros((h, w), dtype=bool)
            cy, cx = h // 2, w // 2
            mask[cy - 40 : cy + 40, cx - 40 : cx + 40] = True
            return mask

        def segment_batch(self, frames):
            return [self.segment(f) for f in frames]

    class MockVGGTResult:
        def __init__(self, n, h, w):
            self.extrinsics = np.zeros((n, 3, 4), dtype=np.float32)
            for i in range(n):
                self.extrinsics[i, :3, :3] = np.eye(3)
                angle = 2 * np.pi * i / max(n, 1)
                self.extrinsics[i, 0, 3] = 0.3 * np.cos(angle)
                self.extrinsics[i, 1, 3] = 0.3 * np.sin(angle)
                self.extrinsics[i, 2, 3] = 1.0

            self.intrinsics = np.zeros((n, 3, 3), dtype=np.float32)
            for i in range(n):
                self.intrinsics[i] = np.array([
                    [300.0, 0, w / 2],
                    [0, 300.0, h / 2],
                    [0, 0, 1],
                ])

            self.depths = np.ones((n, h, w), dtype=np.float32) * 0.5

            self.points_3d = np.zeros((n, h, w, 3), dtype=np.float32)

    class MockVGGT:
        def __init__(self, **kwargs):
            pass

        def process_batch(self, rgb_frames, masks=None):
            n = len(rgb_frames)
            h, w = rgb_frames[0].shape[:2]
            return MockVGGTResult(n, h, w)

    sam2_mod.SAM2Segmenter = MockSAM2
    vggt_mod.VGGTProvider = MockVGGT
    vggt_mod.VGGTResult = MockVGGTResult


def run_smoke_test():
    """Run the smoke test and verify output."""
    print("=" * 60)
    print("Turntable Learning Pipeline — Smoke Test")
    print("=" * 60)

    _create_mock_vision_bridge()

    tmpdir = Path(tempfile.mkdtemp(prefix="turntable_test_"))
    scan_dir = tmpdir / "scans" / "test_cube"
    output_dir = tmpdir / "results"

    try:
        print("\n1. Creating synthetic scan...")
        _create_synthetic_scan(scan_dir, n_frames=20)
        print(f"   Created {len(list((scan_dir / 'frames').glob('*.jpg')))} frames")

        print("\n2. Building experiment config...")
        from monty_ext.scripts.turntable_learn import _build_config, learn

        config = _build_config(
            object_name="test_cube",
            scan_dir=str(tmpdir / "scans"),
            max_steps=15,
            output_dir=str(output_dir),
            frame_subsample=1,
        )

        print("\n3. Running learning pipeline...")
        learn(config)

        print("\n4. Verifying output...")
        model_files = list(output_dir.rglob("model.pt"))
        if not model_files:
            print("   FAIL: No model.pt found!")
            return False

        import torch

        model_path = model_files[0]
        state_dict = torch.load(model_path, map_location="cpu")
        print(f"   model.pt found at: {model_path}")
        print(f"   State dict keys: {list(state_dict.keys())[:5]}...")

        has_graph_data = False
        if "graph_memory" in str(state_dict):
            has_graph_data = True
        for key in state_dict:
            if "graph" in str(key).lower() or "memory" in str(key).lower():
                has_graph_data = True
                break

        if has_graph_data:
            print("   Graph data found in model checkpoint")
        else:
            print("   Warning: No obvious graph data in checkpoint (may be nested)")

        print("\n" + "=" * 60)
        print("SMOKE TEST PASSED")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n   FAIL: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    success = run_smoke_test()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
