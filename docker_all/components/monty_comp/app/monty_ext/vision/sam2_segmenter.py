"""Object segmentation via SAM2, running in the ``vision`` subprocess.

For turntable scanning, the object is the only foreground item on the
turntable.  SAM2's automatic mask generator selects the dominant
non-background mask.

The actual SAM2 model runs in the ``vision`` conda env (Python 3.11 +
torch 2.5) via the ``VisionBridge`` subprocess.  This wrapper runs in
the ``tbp.monty`` env (Python 3.8 + torch 1.13).
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


class SAM2Segmenter:
    """Thin wrapper around VisionBridge.segment() for API consistency."""

    def __init__(self, **kwargs):
        self._bridge = None

    def _ensure_bridge(self):
        if self._bridge is None:
            from monty_ext.vision._bridge import VisionBridge
            self._bridge = VisionBridge.get()
            self._bridge.load_sam2()

    def segment(self, rgb: np.ndarray) -> np.ndarray:
        """Segment the dominant foreground object.

        Args:
            rgb: (H, W, 3) uint8 array in RGB order.

        Returns:
            (H, W) bool mask — True = object pixel.
        """
        self._ensure_bridge()
        return self._bridge.segment(rgb)

    def segment_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Segment each frame independently."""
        self._ensure_bridge()
        return self._bridge.segment_batch(frames)
