"""TCP bridge to the vision_comp container.

The ``tbp.monty`` env uses Python 3.8 + torch 1.13; SAM2/VGGT need
Python 3.10 + torch 2.5.  The vision models run in a separate container
(vision_comp) as a TCP JSON-lines server.  This module connects to that
server and provides a high-level API.

Binary data (images, masks, depths) is exchanged as numpy files on a
shared tmpfs volume (/vision_shm) for near-zero-copy speed.

Usage::

    bridge = VisionBridge.get()       # singleton, connects on first call
    mask = bridge.segment(rgb_array)
    result = bridge.vggt_batch(rgb_list, mask_list)
    bridge.shutdown()                 # optional — cleaned up at exit
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_SHM = os.environ.get("VISION_SHM", "/vision_shm")

_instance: Optional["VisionBridge"] = None
_lock = threading.Lock()

_CONNECT_RETRIES = 30
_CONNECT_DELAY = 2.0


@dataclass
class VGGTResult:
    extrinsics: np.ndarray
    intrinsics: np.ndarray
    depths: np.ndarray
    points_3d: np.ndarray


class VisionBridge:
    """Manages the TCP connection to the vision_comp server."""

    def __init__(self):
        self._sock: Optional[socket.socket] = None
        self._rfile = None
        self._wfile = None
        self._call_lock = threading.Lock()

    @classmethod
    def get(cls) -> "VisionBridge":
        """Return the singleton bridge (connects on first call)."""
        global _instance
        if _instance is None:
            with _lock:
                if _instance is None:
                    _instance = cls()
                    _instance._connect()
                    atexit.register(_instance.shutdown)
        return _instance

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        host = os.environ.get("VISION_HOST", "vision_comp")
        port = int(os.environ.get("VISION_PORT", "5570"))
        logger.info("Connecting to vision server at %s:%d ...", host, port)

        for attempt in range(1, _CONNECT_RETRIES + 1):
            try:
                self._sock = socket.create_connection((host, port), timeout=30)
                self._rfile = self._sock.makefile("r", buffering=1)
                self._wfile = self._sock.makefile("w", buffering=1)
                logger.info("Connected to vision server at %s:%d", host, port)
                return
            except (ConnectionRefusedError, OSError) as e:
                if attempt < _CONNECT_RETRIES:
                    logger.warning(
                        "Vision server not ready (attempt %d/%d): %s",
                        attempt, _CONNECT_RETRIES, e,
                    )
                    time.sleep(_CONNECT_DELAY)
                else:
                    raise RuntimeError(
                        f"Cannot connect to vision server at {host}:{port} "
                        f"after {_CONNECT_RETRIES} attempts"
                    ) from e

    def shutdown(self) -> None:
        if self._sock is not None:
            try:
                self._send({"cmd": "shutdown"})
            except Exception:
                pass
            try:
                self._rfile.close()
                self._wfile.close()
                self._sock.close()
            except Exception:
                pass
            logger.info("Vision bridge disconnected")
        self._sock = None
        self._rfile = None
        self._wfile = None

    def _ensure_connected(self) -> None:
        if self._sock is None:
            logger.warning("Vision bridge not connected, reconnecting ...")
            self._connect()

    # ------------------------------------------------------------------
    # Low-level IPC
    # ------------------------------------------------------------------

    def _send(self, msg: dict) -> dict:
        """Send a JSON command and wait for the JSON response."""
        with self._call_lock:
            self._ensure_connected()
            line = json.dumps(msg) + "\n"
            try:
                self._wfile.write(line)
                self._wfile.flush()
            except (BrokenPipeError, OSError):
                self._sock = None
                self._ensure_connected()
                self._wfile.write(line)
                self._wfile.flush()

            while True:
                resp_line = self._rfile.readline()
                if not resp_line:
                    raise RuntimeError(
                        "Vision server returned empty response. "
                        "Check vision_comp container logs."
                    )
                resp_line = resp_line.strip()
                if not resp_line:
                    continue
                break
            resp = json.loads(resp_line)
            if not resp.get("ok", False):
                raise RuntimeError(
                    f"Vision server error: {resp.get('error', 'unknown')}"
                )
            return resp

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    def load_sam2(self) -> None:
        self._send({"cmd": "load_sam2"})
        logger.info("SAM2 loaded in vision server")

    def load_vggt(self) -> None:
        self._send({"cmd": "load_vggt"})
        logger.info("VGGT loaded in vision server")

    def load_gdino(self) -> bool:
        """Load Grounding DINO.  Returns True if available."""
        resp = self._send({"cmd": "load_gdino"})
        avail = resp.get("available", False)
        logger.info("Grounding DINO loaded (available=%s)", avail)
        return avail

    def detect_gdino(
        self, rgb: np.ndarray, text_prompt: str,
    ) -> dict:
        """Run GDINO detection on a single frame.

        Returns:
            dict with ``"box"`` (list or None) and ``"confidence"`` (float).
        """
        img_path = os.path.join(_SHM, "vb_frame.npy")
        np.save(img_path, rgb)
        resp = self._send({
            "cmd": "detect_gdino",
            "image": img_path,
            "text_prompt": text_prompt,
        })
        return {"box": resp.get("box"), "confidence": resp.get("confidence", 0.0)}

    def segment(
        self, rgb: np.ndarray, text_prompt: Optional[str] = None
    ) -> np.ndarray:
        """Segment the foreground object from an RGB frame.

        Args:
            rgb: (H, W, 3) uint8 array.
            text_prompt: Optional text description for guided segmentation.

        Returns:
            (H, W) bool mask.
        """
        img_path = os.path.join(_SHM, "vb_frame.npy")
        np.save(img_path, rgb)
        msg = {"cmd": "segment", "image": img_path}
        if text_prompt:
            msg["text_prompt"] = text_prompt
        resp = self._send(msg)
        return np.load(resp["mask"])

    def segment_batch(
        self,
        frames: List[np.ndarray],
        text_prompt: Optional[str] = None,
    ) -> List[np.ndarray]:
        """Segment each frame independently."""
        return [self.segment(f, text_prompt=text_prompt) for f in frames]

    def segment_video(
        self,
        frames_dir: str,
        frame_indices: Optional[List[int]] = None,
        text_prompt: Optional[str] = None,
        debug_vis_dir: str = "",
        reanchor_interval: int = 30,
    ) -> List[np.ndarray]:
        """Segment frames using SAM2 VideoPredictor with temporal tracking.

        Args:
            frames_dir: Directory containing extracted JPEG frames.
            frame_indices: Which frame indices to use (subsampled).
                If None, uses all frames in the directory.
            text_prompt: Object description for Grounding DINO.
            debug_vis_dir: Directory for debug visualizations.
            reanchor_interval: Check GDINO every N frames for drift.

        Returns:
            List of (H, W) bool masks, one per selected frame.
        """
        msg = {
            "cmd": "segment_video",
            "frames_dir": frames_dir,
            "text_prompt": text_prompt or "",
            "debug_vis_dir": debug_vis_dir,
            "reanchor_interval": reanchor_interval,
        }
        if frame_indices is not None:
            msg["frame_indices"] = frame_indices

        resp = self._send(msg)
        stacked = np.load(resp["masks"])
        return [stacked[i] for i in range(stacked.shape[0])]

    def vggt_batch(
        self,
        rgb_frames: List[np.ndarray],
        masks: Optional[List[np.ndarray]] = None,
    ) -> VGGTResult:
        """Run VGGT on a batch of RGB frames.

        Args:
            rgb_frames: List of (H, W, 3) uint8 arrays.
            masks: Optional list of (H, W) bool masks.

        Returns:
            VGGTResult with extrinsics, intrinsics, depths, points_3d.
        """
        batch = np.stack(rgb_frames)
        batch_path = os.path.join(_SHM, "vb_batch.npy")
        np.save(batch_path, batch)

        msg = {"cmd": "vggt", "images": batch_path}
        if masks is not None:
            mask_batch = np.stack(masks)
            mask_path = os.path.join(_SHM, "vb_masks.npy")
            np.save(mask_path, mask_batch)
            msg["masks"] = mask_path

        resp = self._send(msg)
        return VGGTResult(
            extrinsics=np.load(resp["extrinsics"]),
            intrinsics=np.load(resp["intrinsics"]),
            depths=np.load(resp["depths"]),
            points_3d=np.load(resp["points3d"]),
        )
