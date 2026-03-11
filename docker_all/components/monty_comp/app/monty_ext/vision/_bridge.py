"""Subprocess bridge to the ``vision`` conda env.

The ``tbp.monty`` env uses Python 3.8 + torch 1.13; SAM2/VGGT need
Python 3.11 + torch 2.5.  This module spawns a long-running subprocess
in the ``vision`` env and communicates via JSON-lines on stdin/stdout,
with binary data (images, masks, depths) exchanged as numpy files on
/dev/shm for near-zero-copy speed.

Usage::

    bridge = VisionBridge.get()       # singleton, starts subprocess on first call
    mask = bridge.segment(rgb_array)
    result = bridge.vggt_batch(rgb_list, mask_list)
    bridge.shutdown()                 # optional — cleaned up at exit
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import subprocess
import threading
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_SHM = "/dev/shm"
_CONDA_RUN = ["conda", "run", "--no-capture-output", "-n", "vision", "python", "-m", "monty_ext.vision._server"]

_instance: Optional["VisionBridge"] = None
_lock = threading.Lock()


@dataclass
class VGGTResult:
    extrinsics: np.ndarray
    intrinsics: np.ndarray
    depths: np.ndarray
    points_3d: np.ndarray


class VisionBridge:
    """Manages the subprocess running ``_server.py`` in the vision env."""

    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self._call_lock = threading.Lock()

    @classmethod
    def get(cls) -> "VisionBridge":
        """Return the singleton bridge (starts subprocess on first call)."""
        global _instance
        if _instance is None:
            with _lock:
                if _instance is None:
                    _instance = cls()
                    _instance._start()
                    atexit.register(_instance.shutdown)
        return _instance

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _start(self) -> None:
        logger.info("Starting vision subprocess: %s", " ".join(_CONDA_RUN))
        log_path = os.path.join(
            os.environ.get("LOG_DIR", "/var/log/monty"), "vision_server.log"
        )
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._stderr_file = open(log_path, "a")
        self._proc = subprocess.Popen(
            _CONDA_RUN,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            text=True,
            bufsize=1,
        )
        logger.info("Vision subprocess started (pid=%d)", self._proc.pid)

    def shutdown(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._send({"cmd": "shutdown"})
            except Exception:
                pass
            self._proc.terminate()
            self._proc.wait(timeout=5)
            logger.info("Vision subprocess terminated")
        self._proc = None

    def _ensure_alive(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            logger.warning("Vision subprocess died, restarting ...")
            self._start()

    # ------------------------------------------------------------------
    # Low-level IPC
    # ------------------------------------------------------------------

    def _send(self, msg: dict) -> dict:
        """Send a JSON command and wait for the JSON response."""
        with self._call_lock:
            self._ensure_alive()
            line = json.dumps(msg) + "\n"
            self._proc.stdin.write(line)
            self._proc.stdin.flush()

            resp_line = self._proc.stdout.readline()
            if not resp_line:
                raise RuntimeError(
                    "Vision subprocess returned empty response. "
                    "Check /var/log/monty/vision_server.log for details."
                )
            resp = json.loads(resp_line)
            if not resp.get("ok", False):
                raise RuntimeError(
                    f"Vision subprocess error: {resp.get('error', 'unknown')}"
                )
            return resp

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    def load_sam2(self) -> None:
        self._send({"cmd": "load_sam2"})
        logger.info("SAM2 loaded in vision subprocess")

    def load_vggt(self) -> None:
        self._send({"cmd": "load_vggt"})
        logger.info("VGGT loaded in vision subprocess")

    def segment(self, rgb: np.ndarray) -> np.ndarray:
        """Segment the foreground object from an RGB frame.

        Args:
            rgb: (H, W, 3) uint8 array.

        Returns:
            (H, W) bool mask.
        """
        img_path = os.path.join(_SHM, "vb_frame.npy")
        np.save(img_path, rgb)
        resp = self._send({"cmd": "segment", "image": img_path})
        return np.load(resp["mask"])

    def segment_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Segment each frame independently."""
        return [self.segment(f) for f in frames]

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
