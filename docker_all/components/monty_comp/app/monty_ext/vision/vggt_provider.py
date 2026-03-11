"""Camera poses + depth maps from RGB via VGGT, running in the ``vision`` subprocess.

VGGT is a feed-forward transformer that takes 1-to-N RGB images and outputs
camera poses and per-frame dense depth maps in a single forward pass.  This
eliminates all calibration.

The actual VGGT model runs in the ``vision`` conda env (Python 3.11 +
torch 2.5) via the ``VisionBridge`` subprocess.  This wrapper runs in
the ``tbp.monty`` env (Python 3.8 + torch 1.13).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VGGTResult:
    """Output from a VGGT batch inference."""

    extrinsics: np.ndarray
    """(N, 3, 4) camera-to-world transforms for each frame."""

    intrinsics: np.ndarray
    """(N, 3, 3) camera intrinsic matrices for each frame."""

    depths: np.ndarray
    """(N, H, W) dense depth maps in metres."""

    points_3d: np.ndarray
    """(N, H, W, 3) dense 3D point maps in world coordinates."""


class VGGTProvider:
    """Thin wrapper around VisionBridge.vggt_batch() for API consistency."""

    def __init__(self, **kwargs):
        self._bridge = None

    def _ensure_bridge(self):
        if self._bridge is None:
            from monty_ext.vision._bridge import VisionBridge
            self._bridge = VisionBridge.get()
            self._bridge.load_vggt()

    def process_batch(
        self,
        rgb_frames: List[np.ndarray],
        masks: Optional[List[np.ndarray]] = None,
    ) -> VGGTResult:
        """Run VGGT on a batch of RGB frames.

        Args:
            rgb_frames: List of (H, W, 3) uint8 arrays in RGB order.
            masks: Optional list of (H, W) bool masks.

        Returns:
            VGGTResult with poses, depths, and 3D points for each frame.
        """
        self._ensure_bridge()
        bridge_result = self._bridge.vggt_batch(rgb_frames, masks)
        return VGGTResult(
            extrinsics=bridge_result.extrinsics,
            intrinsics=bridge_result.intrinsics,
            depths=bridge_result.depths,
            points_3d=bridge_result.points_3d,
        )
