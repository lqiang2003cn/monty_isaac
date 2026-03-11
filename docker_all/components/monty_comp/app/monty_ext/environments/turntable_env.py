"""TurntableEnvironment — EmbodiedEnvironment for real-world turntable scanning.

Pipeline per step:
  D455 RGB -> SAM2 mask -> buffer -> VGGT (batched) -> RGBD + pose -> Monty

The turntable rotates the object; the camera is static.  VGGT infers
relative camera poses from RGB alone, so the turntable speed and camera
intrinsics need not be known.

Reference: everything_is_awesome/EverythingIsAwesomeEnvironment
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

VectorXYZ = Tuple[float, float, float]
QuaternionWXYZ = Tuple[float, float, float, float]


@dataclass
class TurntableConfig:
    """All knobs for the turntable environment."""

    vggt_batch_size: int = 5
    vggt_resolution: int = 518
    vggt_device: str = "cuda"
    vggt_max_batch: int = 8

    sam2_device: str = "cuda"
    sam2_points_per_side: int = 32

    patch_size: int = 64
    patch_center_x: int = 0
    patch_center_y: int = 0

    sensor_id: str = "patch"
    agent_id: str = "agent_id_0"


class TurntableObservations(dict):
    """Observation dict returned by TurntableEnvironment.step().

    Structure follows the Monty convention:
        obs[agent_id][sensor_id.modality] = array
    """
    pass


class TurntableEnvironment:
    """EmbodiedEnvironment for turntable scanning with a static RealSense D455.

    Implements the minimal interface that Monty's EnvironmentDataset expects:
    ``step(action)``, ``get_state()``, ``reset()``, ``close()``.
    """

    def __init__(self, config: Optional[TurntableConfig] = None):
        self._cfg = config or TurntableConfig()

        self._camera = None
        self._segmenter = None
        self._vggt = None

        self._frame_buffer: List[np.ndarray] = []
        self._mask_buffer: List[np.ndarray] = []
        self._ready_queue: deque = deque()

        self._step_count = 0
        self._current_pose: Optional[np.ndarray] = None
        self._current_position: VectorXYZ = (0.0, 0.0, 0.0)
        self._current_rotation: QuaternionWXYZ = (1.0, 0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------

    def _ensure_init(self) -> None:
        if self._camera is not None:
            return

        from monty_ext.sensor_modules.realsense_sm import (
            RealsenseCapture,
            RealsenseConfig,
        )
        from monty_ext.vision.sam2_segmenter import SAM2Segmenter
        from monty_ext.vision.vggt_provider import VGGTProvider

        self._camera = RealsenseCapture(RealsenseConfig())
        self._segmenter = SAM2Segmenter(
            device=self._cfg.sam2_device,
            points_per_side=self._cfg.sam2_points_per_side,
        )
        self._vggt = VGGTProvider(
            device=self._cfg.vggt_device,
            max_batch_size=self._cfg.vggt_max_batch,
            input_resolution=self._cfg.vggt_resolution,
        )

    # ------------------------------------------------------------------
    # EmbodiedEnvironment interface
    # ------------------------------------------------------------------

    @property
    def action_space(self):
        return ["turntable_step"]

    def step(self, action=None) -> TurntableObservations:
        """Capture a frame, process it, and return an observation.

        If the internal queue has pre-computed observations (from a VGGT
        batch), returns the next one immediately.  Otherwise captures
        frames until a full VGGT batch is ready, processes it, and
        returns the first result.
        """
        self._ensure_init()

        if self._ready_queue:
            obs = self._ready_queue.popleft()
            self._step_count += 1
            return obs

        self._fill_batch()
        obs = self._ready_queue.popleft()
        self._step_count += 1
        return obs

    def get_state(self) -> dict:
        """Return proprioceptive state in Monty's expected format.

        The "agent" position and rotation are the VGGT-estimated camera
        pose for the most recently returned observation.
        """
        aid = self._cfg.agent_id
        sid = self._cfg.sensor_id

        return {
            aid: {
                "position": self._current_position,
                "rotation": self._current_rotation,
                "sensors": {
                    f"{sid}.rgba": {
                        "position": (0.0, 0.0, 0.0),
                        "rotation": (1.0, 0.0, 0.0, 0.0),
                    },
                    f"{sid}.depth": {
                        "position": (0.0, 0.0, 0.0),
                        "rotation": (1.0, 0.0, 0.0, 0.0),
                    },
                },
            }
        }

    def reset(self) -> TurntableObservations:
        """Reset the environment and return the first observation."""
        self._ensure_init()
        self._camera.start()
        self._frame_buffer.clear()
        self._mask_buffer.clear()
        self._ready_queue.clear()
        self._step_count = 0
        self._current_position = (0.0, 0.0, 0.0)
        self._current_rotation = (1.0, 0.0, 0.0, 0.0)

        self._fill_batch()
        return self._ready_queue.popleft()

    def add_object(self, name, position=None, rotation=None, **kwargs):
        pass

    def remove_all_objects(self):
        pass

    def close(self):
        if self._camera is not None:
            self._camera.stop()

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _fill_batch(self) -> None:
        """Capture frames until a VGGT batch is full, then process."""
        batch_size = self._cfg.vggt_batch_size

        while len(self._frame_buffer) < batch_size:
            rgb = self._camera.capture()
            mask = self._segmenter.segment(rgb)
            self._frame_buffer.append(rgb)
            self._mask_buffer.append(mask)

        result = self._vggt.process_batch(
            self._frame_buffer[:batch_size],
            masks=self._mask_buffer[:batch_size],
        )

        for i in range(batch_size):
            obs = self._make_observation(
                rgb=self._frame_buffer[i],
                mask=self._mask_buffer[i],
                depth=result.depths[i],
                extrinsic=result.extrinsics[i],
                intrinsic=result.intrinsics[i],
            )
            self._ready_queue.append(obs)

        self._frame_buffer = self._frame_buffer[batch_size:]
        self._mask_buffer = self._mask_buffer[batch_size:]

    def _make_observation(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        depth: np.ndarray,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
    ) -> TurntableObservations:
        """Build one Monty-compatible observation dict from processed frame data."""
        aid = self._cfg.agent_id
        sid = self._cfg.sensor_id

        rgba = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = (mask * 255).astype(np.uint8)

        ps = self._cfg.patch_size
        h, w = rgb.shape[:2]
        cy = self._cfg.patch_center_y or h // 2
        cx = self._cfg.patch_center_x or w // 2
        y0 = max(0, cy - ps // 2)
        x0 = max(0, cx - ps // 2)
        rgba_patch = rgba[y0 : y0 + ps, x0 : x0 + ps]
        depth_patch = depth[
            y0 * depth.shape[0] // h : (y0 + ps) * depth.shape[0] // h,
            x0 * depth.shape[1] // w : (x0 + ps) * depth.shape[1] // w,
        ]

        position, rotation = self._extrinsic_to_pose(extrinsic)
        self._current_position = position
        self._current_rotation = rotation

        obs = TurntableObservations()
        obs[aid] = {
            f"{sid}.rgba": rgba_patch,
            f"{sid}.depth": depth_patch,
        }
        return obs

    @staticmethod
    def _extrinsic_to_pose(
        extrinsic: np.ndarray,
    ) -> Tuple[VectorXYZ, QuaternionWXYZ]:
        """Convert a (3,4) camera-to-world extrinsic matrix to position + quaternion."""
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        position = tuple(float(v) for v in t)

        # Rotation matrix to quaternion (wxyz)
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return position, (float(w), float(x), float(y), float(z))
