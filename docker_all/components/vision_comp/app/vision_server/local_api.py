"""Local (in-process) API for vision models.

Provides the same high-level interface as ``monty_ext.vision._bridge.VisionBridge``
but calls the server's model functions directly — no TCP, no IPC.

Use this for scripts that run *inside* the vision_comp container
(e.g. ``test_vision_pipeline.py``) where the models are available locally.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from vision_server import server


@dataclass
class VGGTResult:
    extrinsics: np.ndarray
    intrinsics: np.ndarray
    depths: np.ndarray
    points_3d: np.ndarray


def load_sam2() -> None:
    server._load_sam2()


def load_vggt() -> None:
    server._load_vggt()


def load_gdino() -> bool:
    """Load Grounding DINO.  Returns True if available."""
    server._load_gdino()
    return server._gdino_model is not None


def detect_gdino(rgb: np.ndarray, text_prompt: str) -> dict:
    """Run GDINO detection on a single frame.

    Returns dict with ``"box"`` (list or None) and ``"confidence"`` (float).
    """
    result = server._detect_box_gdino(rgb, text_prompt)
    if result is not None:
        return {"box": list(result[:4]), "confidence": result[4]}
    return {"box": None, "confidence": 0.0}


def segment(rgb: np.ndarray, text_prompt: Optional[str] = None) -> np.ndarray:
    """Segment the foreground object.  Returns (H, W) bool mask."""
    img_path = os.path.join(server._SHM, "vb_frame.npy")
    np.save(img_path, rgb)
    msg = {"image": img_path}
    if text_prompt:
        msg["text_prompt"] = text_prompt
    resp = server._handle_segment(msg)
    return np.load(resp["mask"])


def segment_video(
    frames_dir: str,
    frame_indices: Optional[List[int]] = None,
    text_prompt: str = "",
    debug_vis_dir: str = "",
    reanchor_interval: int = 30,
) -> List[np.ndarray]:
    """Segment frames using SAM2 VideoPredictor.  Returns list of bool masks."""
    msg = {
        "frames_dir": frames_dir,
        "text_prompt": text_prompt,
        "debug_vis_dir": debug_vis_dir,
        "reanchor_interval": reanchor_interval,
    }
    if frame_indices is not None:
        msg["frame_indices"] = frame_indices
    resp = server._handle_segment_video(msg)
    stacked = np.load(resp["masks"])
    return [stacked[i] for i in range(stacked.shape[0])]


def vggt_batch(
    rgb_frames: List[np.ndarray],
    masks: Optional[List[np.ndarray]] = None,
) -> VGGTResult:
    """Run VGGT on a batch of RGB frames.  Returns VGGTResult."""
    batch = np.stack(rgb_frames)
    batch_path = os.path.join(server._SHM, "vb_batch.npy")
    np.save(batch_path, batch)

    msg = {"images": batch_path}
    if masks is not None:
        mask_batch = np.stack(masks)
        mask_path = os.path.join(server._SHM, "vb_masks.npy")
        np.save(mask_path, mask_batch)
        msg["masks"] = mask_path

    resp = server._handle_vggt(msg)
    return VGGTResult(
        extrinsics=np.load(resp["extrinsics"]),
        intrinsics=np.load(resp["intrinsics"]),
        depths=np.load(resp["depths"]),
        points_3d=np.load(resp["points3d"]),
    )
