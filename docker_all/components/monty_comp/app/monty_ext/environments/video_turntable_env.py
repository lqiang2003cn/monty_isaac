"""VideoTurntableEnvironment — EmbodiedEnvironment for pre-recorded turntable scans.

Reads frames from an MP4 video recorded by ``record-scan``, runs them through
the SAM2 + VGGT vision pipeline, and returns Monty-compatible
(observations, ProprioceptiveState) tuples.

The environment replaces the live-camera ``TurntableEnvironment`` for offline
learning.  It follows the same patterns as the everything_is_awesome
``EverythingIsAwesomeEnvironment`` so that the upstream ``EnvironmentInterface``
can drive it without modifications.

Pipeline per step::

    video frame → SAM2 mask → VGGT (batched) → RGBD + pose → Monty observations
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import quaternion as qt

from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    ProprioceptiveState,
    SensorState,
)

logger = logging.getLogger(__name__)

AGENT_ID = "agent_id_0"
SENSOR_ID = "patch"
PATCH_SIZE = 70
DEPTH_VOID_VALUE = 10.0


@dataclass
class VideoTurntableConfig:
    """Configuration for the video turntable environment."""

    scan_dir: str = "/data/scans/unknown"
    object_name: str = "unknown"
    description: str = ""
    patch_size: int = PATCH_SIZE
    frame_subsample: int = 5
    vggt_batch_size: int = 5
    agent_id: str = AGENT_ID
    sensor_id: str = SENSOR_ID
    depth_void_value: float = DEPTH_VOID_VALUE
    debug_vis_dir: str = ""


class VideoTurntableEnvironment:
    """EmbodiedEnvironment backed by a pre-recorded turntable scan video.

    Implements the interface expected by the upstream ``EnvironmentInterface``:
    ``reset()``, ``step(actions)``, ``get_state()``, ``close()``.

    ``step()`` ignores the actions argument — the turntable video determines
    the exploration trajectory.  When all frames are consumed the environment
    signals completion by raising ``StopIteration``.
    """

    def __init__(self, config: Optional[VideoTurntableConfig] = None, **kwargs):
        self._cfg = config or VideoTurntableConfig(**kwargs)
        self._scan_dir = Path(self._cfg.scan_dir)

        self._frames: List[np.ndarray] = []
        self._masks: List[np.ndarray] = []
        self._depths: Optional[np.ndarray] = None
        self._extrinsics: Optional[np.ndarray] = None
        self._intrinsics: Optional[np.ndarray] = None

        self._frame_idx = 0
        self._prepared = False

        self._current_position = [0.0, 0.0, 1.0]
        self._current_rotation = qt.from_float_array([1.0, 0.0, 0.0, 0.0])

    # ------------------------------------------------------------------
    # Lazy preparation
    # ------------------------------------------------------------------

    def _prepare(self):
        """Extract frames, run SAM2 + VGGT on all frames at once."""
        if self._prepared:
            return

        self._frames_dir = self._scan_dir / "frames"
        video_path = self._scan_dir / "video.mp4"

        if not self._frames_dir.exists() or \
                len(list(self._frames_dir.glob("*.jpg"))) == 0:
            self._extract_frames(video_path, self._frames_dir)

        all_frame_paths = sorted(self._frames_dir.glob("*.jpg"))
        subsample = self._cfg.frame_subsample
        frame_paths = all_frame_paths[::subsample]
        self._frame_indices = list(range(0, len(all_frame_paths), subsample))

        logger.info(
            "Loading %d frames (subsample=%d) from %s",
            len(frame_paths),
            subsample,
            self._frames_dir,
        )

        self._frames = []
        for p in frame_paths:
            img = cv2.imread(str(p))
            if img is not None:
                self._frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not self._frames:
            raise RuntimeError(f"No frames loaded from {self._frames_dir}")

        self._run_vision_pipeline()
        self._prepared = True

    def _extract_frames(self, video_path: Path, frames_dir: Path):
        """Extract all frames from the MP4 into numbered JPEGs."""
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        frames_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(str(frames_dir / f"{idx:06d}.jpg"), frame)
            idx += 1
        cap.release()
        logger.info("Extracted %d frames from %s", idx, video_path)

    def _run_vision_pipeline(self):
        """Run SAM2 segmentation and VGGT depth/pose on all loaded frames."""
        from monty_ext.vision.sam2_segmenter import SAM2Segmenter
        from monty_ext.vision.vggt_provider import VGGTProvider

        segmenter = SAM2Segmenter()
        vggt = VGGTProvider()

        desc = self._cfg.description or self._cfg.object_name
        vis_dir = self._cfg.debug_vis_dir

        logger.info(
            "Running SAM2 video segmentation on %d frames (prompt=%r)...",
            len(self._frames), desc,
        )
        self._masks = segmenter.segment_video(
            frames_dir=str(self._frames_dir),
            frame_indices=self._frame_indices,
            text_prompt=desc,
            debug_vis_dir=vis_dir,
        )

        if vis_dir:
            self._save_sam2_debug(vis_dir)

        # VGGT produces camera poses relative to the first frame in each batch.
        # To get a consistent world frame, we use overlapping batches and align
        # each batch to the global coordinate system of the first batch.
        batch_size = self._cfg.vggt_batch_size
        overlap = 1

        all_extrinsics = np.zeros((len(self._frames), 3, 4), dtype=np.float64)
        all_intrinsics = np.zeros((len(self._frames), 3, 3), dtype=np.float64)
        all_depths = []

        global_transform = np.eye(4)
        start = 0
        batch_idx = 0

        while start < len(self._frames):
            end = min(start + batch_size, len(self._frames))
            batch_frames = self._frames[start:end]
            batch_masks = self._masks[start:end]

            logger.info(
                "VGGT batch %d: frames %d-%d / %d",
                batch_idx, start, end, len(self._frames),
            )
            result = vggt.process_batch(batch_frames, masks=batch_masks)

            if batch_idx == 0:
                all_extrinsics[start:end] = result.extrinsics
                all_intrinsics[start:end] = result.intrinsics
                all_depths.extend([result.depths[i] for i in range(len(batch_frames))])
            else:
                # Align this batch to the global frame using the overlap frame.
                # The last frame of the previous batch = first frame of this batch.
                # prev_ext is the global-frame extrinsic of the overlap frame.
                # curr_ext is this batch's local extrinsic of the same frame.
                prev_ext_34 = all_extrinsics[start]
                prev_ext = np.eye(4)
                prev_ext[:3, :] = prev_ext_34

                curr_ext_34 = result.extrinsics[0]
                curr_ext = np.eye(4)
                curr_ext[:3, :] = curr_ext_34

                # global = prev_ext @ inv(curr_ext) @ local
                curr_ext_inv = np.linalg.inv(curr_ext)
                alignment = prev_ext @ curr_ext_inv

                for j in range(len(batch_frames)):
                    local_ext = np.eye(4)
                    local_ext[:3, :] = result.extrinsics[j]
                    aligned = alignment @ local_ext
                    idx = start + j
                    if idx < len(self._frames):
                        all_extrinsics[idx] = aligned[:3, :]
                        all_intrinsics[idx] = result.intrinsics[j]

                # Only add depths for non-overlapping frames
                for j in range(overlap, len(batch_frames)):
                    idx = start + j
                    if idx < len(self._frames):
                        all_depths.append(result.depths[j])

            batch_idx += 1
            start = end - overlap
            if start >= len(self._frames) - 1:
                break

        self._extrinsics = all_extrinsics
        self._intrinsics = all_intrinsics
        self._depths = np.stack(all_depths[:len(self._frames)])

        logger.info(
            "Vision pipeline complete: %d frames, extrinsics %s, depths %s",
            len(self._frames),
            self._extrinsics.shape,
            self._depths.shape,
        )

        if vis_dir:
            self._save_vggt_debug(vis_dir)

    # ------------------------------------------------------------------
    # Debug visualization helpers
    # ------------------------------------------------------------------

    def _save_sam2_debug(self, vis_dir: str):
        """Save SAM2 segmentation overlays for visual inspection."""
        out = Path(vis_dir) / "sam2_masks"
        out.mkdir(parents=True, exist_ok=True)
        step = max(1, len(self._frames) // 20)
        for i in range(0, len(self._frames), step):
            rgb = self._frames[i]
            mask = self._masks[i]
            overlay = rgb.copy()
            overlay[mask] = (
                overlay[mask].astype(np.float32) * 0.5
                + np.array([0, 255, 0], dtype=np.float32) * 0.5
            ).astype(np.uint8)
            overlay[~mask] = (overlay[~mask].astype(np.float32) * 0.3).astype(np.uint8)
            pct = mask.sum() / mask.size * 100
            cv2.putText(
                overlay, f"frame {i}  mask {pct:.1f}%",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
            )
            cv2.imwrite(
                str(out / f"frame_{i:04d}.jpg"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )
        logger.info("SAM2 debug overlays saved to %s (%d images)", out, len(list(out.glob("*.jpg"))))

    def _save_vggt_debug(self, vis_dir: str):
        """Save VGGT depth maps and camera trajectory for inspection."""
        out = Path(vis_dir) / "vggt_debug"
        out.mkdir(parents=True, exist_ok=True)

        step = max(1, len(self._frames) // 20)
        for i in range(0, len(self._frames), step):
            depth = self._depths[i].squeeze()
            d_valid = depth[depth > 0]
            if len(d_valid) > 0:
                vmin, vmax = np.percentile(d_valid, [2, 98])
            else:
                vmin, vmax = 0, 1
            d_norm = np.clip((depth - vmin) / max(vmax - vmin, 1e-6), 0, 1)
            import matplotlib.cm as _cm
            d_color = (_cm.viridis(d_norm)[:, :, :3] * 255).astype(np.uint8)
            cv2.putText(
                d_color, f"frame {i}  depth [{vmin:.3f}, {vmax:.3f}]m",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )
            cv2.imwrite(
                str(out / f"depth_{i:04d}.jpg"),
                cv2.cvtColor(d_color, cv2.COLOR_RGB2BGR),
            )

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as mplt
        fig = mplt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        positions = np.array([self._extrinsics[i][:3, 3] for i in range(len(self._extrinsics))])
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], "b.-", alpha=0.5)
        ax.scatter(*positions[0], c="green", s=100, label="start")
        ax.scatter(*positions[-1], c="red", s=100, label="end")
        scale = 0.1
        for i in range(0, len(self._extrinsics), max(1, len(self._extrinsics) // 15)):
            R = self._extrinsics[i][:3, :3]
            t = self._extrinsics[i][:3, 3]
            look_dir = R[:, 2]
            ax.quiver(*t, *(-look_dir * scale), color="red", alpha=0.5, arrow_length_ratio=0.3)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        ax.set_title(f"VGGT Camera Trajectory ({len(self._extrinsics)} frames)")
        fig.savefig(str(out / "camera_trajectory.png"), dpi=150, bbox_inches="tight")
        mplt.close(fig)

        np.savez(
            str(out / "vggt_data.npz"),
            extrinsics=self._extrinsics,
            intrinsics=self._intrinsics,
            depth_stats=np.array([
                [self._depths[i].min(), self._depths[i].max(), self._depths[i].mean()]
                for i in range(len(self._depths))
            ]),
        )
        logger.info("VGGT debug visualizations saved to %s", out)

    # ------------------------------------------------------------------
    # EmbodiedEnvironment interface
    # ------------------------------------------------------------------

    def reset(self):
        """Reset the environment and return the first (observations, state)."""
        self._prepare()
        self._frame_idx = 0
        obs = self._make_observation(0)
        state = self._make_state(0)
        return obs, state

    def step(self, actions: Any = None):
        """Advance to the next frame and return (observations, state).

        The ``actions`` argument is ignored — the video determines the
        trajectory.

        Raises:
            StopIteration: When all frames have been consumed.
        """
        self._frame_idx += 1
        if self._frame_idx >= len(self._frames):
            raise StopIteration("All video frames consumed")

        obs = self._make_observation(self._frame_idx)
        state = self._make_state(self._frame_idx)
        return obs, state

    def get_state(self) -> ProprioceptiveState:
        """Return the current proprioceptive state."""
        return self._make_state(self._frame_idx)

    def close(self):
        self._frames = []
        self._masks = []
        self._depths = None
        self._extrinsics = None
        self._intrinsics = None
        self._prepared = False

    def add_object(self, *args, **kwargs):
        pass

    def remove_all_objects(self):
        pass

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _make_observation(self, idx: int) -> dict:
        """Build one Monty-compatible observation dict."""
        rgb = self._frames[idx]
        mask = self._masks[idx]
        depth = self._depths[idx]

        rgba = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = (mask.astype(np.uint8) * 255)

        ps = self._cfg.patch_size
        h, w = rgb.shape[:2]
        cy, cx = h // 2, w // 2
        y0 = max(0, cy - ps // 2)
        x0 = max(0, cx - ps // 2)
        rgba_patch = rgba[y0 : y0 + ps, x0 : x0 + ps]

        # Resize depth to match RGB, then crop the same patch
        if depth.shape[:2] != (h, w):
            depth_full = cv2.resize(
                depth.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
            )
        else:
            depth_full = depth.astype(np.float32)

        depth_patch = depth_full[y0 : y0 + ps, x0 : x0 + ps].copy()

        mask_patch = mask[y0 : y0 + ps, x0 : x0 + ps]

        # Build a semantic mask (1=on-object, 0=background) so that
        # DepthTo3DLocations uses our SAM2 mask instead of the depth-based
        # heuristic (which rejects pixels with depth >= 1m).
        semantic_patch = mask_patch.astype(np.int32)

        # Set background depth to a high value for safety
        depth_patch[~mask_patch] = self._cfg.depth_void_value

        aid = self._cfg.agent_id
        sid = self._cfg.sensor_id

        return {
            aid: {
                sid: {
                    "rgba": rgba_patch,
                    "depth": depth_patch,
                    "semantic": semantic_patch,
                }
            }
        }

    def _make_state(self, idx: int) -> ProprioceptiveState:
        """Build ProprioceptiveState from VGGT camera extrinsics."""
        if self._extrinsics is not None and idx < len(self._extrinsics):
            ext = self._extrinsics[idx]
            position, rotation = self._extrinsic_to_pose(ext)
        else:
            position = [0.0, 0.0, 1.0]
            rotation = qt.from_float_array([1.0, 0.0, 0.0, 0.0])

        self._current_position = position
        self._current_rotation = rotation

        aid = self._cfg.agent_id
        sid = self._cfg.sensor_id

        sensor_state = SensorState(
            position=[0.0, 0.0, 0.0],
            rotation=qt.from_float_array([1.0, 0.0, 0.0, 0.0]),
        )
        return ProprioceptiveState({
            aid: AgentState(
                sensors={
                    sid: sensor_state,
                },
                position=position,
                rotation=rotation,
            )
        })

    @staticmethod
    def _extrinsic_to_pose(extrinsic: np.ndarray):
        """Convert a (3,4) camera-to-world extrinsic to (position, quaternion).

        Returns:
            (position_list, quaternion) where quaternion is numpy-quaternion.
        """
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        position = [float(v) for v in t]

        # Rotation matrix → quaternion via numpy-quaternion
        rotation = qt.from_rotation_matrix(R)
        return position, rotation

    def get_hfov(self) -> float:
        """Compute horizontal field of view from VGGT intrinsics.

        Returns:
            HFOV in degrees.  Falls back to 90.0 if intrinsics unavailable.
        """
        if self._intrinsics is not None and len(self._intrinsics) > 0:
            K = self._intrinsics[0]
            fx = K[0, 0]
            w = self._frames[0].shape[1] if self._frames else 1280
            hfov_rad = 2.0 * math.atan(w / (2.0 * fx))
            return math.degrees(hfov_rad)
        return 90.0

    @property
    def num_frames(self) -> int:
        self._prepare()
        return len(self._frames)
