"""Object segmentation via SAM2, running in the ``vision`` subprocess.

For turntable scanning, the object is the only foreground item on the
turntable.  SAM2's automatic mask generator selects the dominant
non-background mask.  When a ``text_prompt`` is supplied, Grounding DINO
generates a bounding box and SAM2's image predictor refines it into a mask.

The actual SAM2 model runs in the ``vision`` conda env (Python 3.11 +
torch 2.5) via the ``VisionBridge`` subprocess.  This wrapper runs in
the ``tbp.monty`` env (Python 3.8 + torch 1.13).
"""

from __future__ import annotations

import logging
from typing import List, Optional

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

    def segment(
        self, rgb: np.ndarray, text_prompt: Optional[str] = None
    ) -> np.ndarray:
        """Segment the dominant foreground object.

        Args:
            rgb: (H, W, 3) uint8 array in RGB order.
            text_prompt: Optional description of the object for guided
                segmentation (uses auto mask generator if not provided).

        Returns:
            (H, W) bool mask — True = object pixel.
        """
        self._ensure_bridge()
        return self._bridge.segment(rgb, text_prompt=text_prompt)

    def segment_batch(
        self,
        frames: List[np.ndarray],
        text_prompt: Optional[str] = None,
    ) -> List[np.ndarray]:
        """Segment each frame independently."""
        self._ensure_bridge()
        return self._bridge.segment_batch(frames, text_prompt=text_prompt)

    def segment_video(
        self,
        frames_dir: str,
        frame_indices: Optional[List[int]] = None,
        text_prompt: Optional[str] = None,
        debug_vis_dir: str = "",
        reanchor_interval: int = 30,
    ) -> List[np.ndarray]:
        """Segment frames using SAM2 VideoPredictor with temporal tracking.

        Uses Grounding DINO for initial prompting and periodic drift
        detection.  Falls back to center-point prompt if GDINO is
        unavailable.

        Args:
            frames_dir: Directory containing extracted JPEG frames.
            frame_indices: Which frame indices to use (subsampled).
            text_prompt: Object description for Grounding DINO.
            debug_vis_dir: Directory for debug visualizations.
            reanchor_interval: Check GDINO every N frames for drift.

        Returns:
            List of (H, W) bool masks, one per selected frame.
        """
        self._ensure_bridge()
        return self._bridge.segment_video(
            frames_dir=frames_dir,
            frame_indices=frame_indices,
            text_prompt=text_prompt,
            debug_vis_dir=debug_vis_dir,
            reanchor_interval=reanchor_interval,
        )
