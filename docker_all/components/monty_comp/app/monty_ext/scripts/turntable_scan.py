"""Main entry point for the turntable scanning pipeline.

Runs inside monty_comp.  Listens for start/stop commands from ros2_comp via
ZMQ (tcp://*:5560), then drives Monty's learning loop.

For **live** scanning, observations from the camera are passed to the Monty
model.  For **pre-recorded** scanning, use ``turntable-learn`` instead.

Protocol (JSON over ZMQ REP socket):

    -> {"action": "start", "object_name": "cup"}
    <- {"status": "started", "object_name": "cup"}

    -> {"action": "status"}
    <- {"status": "scanning", "step": 42, "max_steps": 1000}
    <- {"status": "idle"}

    -> {"action": "stop"}
    <- {"status": "stopped"}

Usage (inside container)::

    conda activate tbp.monty
    turntable-scan              # uses pyproject.toml entry point
    python -m monty_ext.scripts.turntable_scan   # alternative
"""

from __future__ import annotations

import copy
import json
import logging
import os
import signal
import sys
import threading
import time
from typing import Optional

import numpy as np
import zmq

logger = logging.getLogger(__name__)

ZMQ_PORT = int(os.environ.get("ZMQ_SCAN_PORT", "5560"))
MAX_TRAIN_STEPS = 1000


class ScanOrchestrator:
    """Orchestrates turntable scanning sessions.

    Runs the ZMQ server on the main thread and scanning on a worker thread
    so commands (status, stop) can be handled while a scan is in progress.
    """

    def __init__(self, port: int = ZMQ_PORT):
        self._port = port
        self._ctx: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None
        self._shutdown = threading.Event()

        self._scan_thread: Optional[threading.Thread] = None
        self._scanning = False
        self._current_object: str = ""
        self._current_step: int = 0
        self._max_steps: int = MAX_TRAIN_STEPS
        self._stop_requested = threading.Event()

    def run(self) -> None:
        """Start the ZMQ server and block until shutdown."""
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.REP)
        self._socket.bind(f"tcp://*:{self._port}")
        logger.info("Scan orchestrator listening on tcp://*:%d", self._port)

        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)

        while not self._shutdown.is_set():
            events = dict(poller.poll(timeout=500))
            if self._socket in events:
                raw = self._socket.recv_string()
                reply = self._handle_message(raw)
                self._socket.send_string(json.dumps(reply))

        self._cleanup()

    def shutdown(self) -> None:
        self._shutdown.set()
        self._stop_requested.set()

    # ------------------------------------------------------------------
    # Command handling
    # ------------------------------------------------------------------

    def _handle_message(self, raw: str) -> dict:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return {"status": "error", "message": "invalid JSON"}

        action = msg.get("action", "")

        if action == "start":
            return self._cmd_start(msg)
        elif action == "stop":
            return self._cmd_stop()
        elif action == "status":
            return self._cmd_status()
        else:
            return {"status": "error", "message": f"unknown action: {action}"}

    def _cmd_start(self, msg: dict) -> dict:
        if self._scanning:
            return {
                "status": "error",
                "message": f"already scanning: {self._current_object}",
            }

        object_name = msg.get("object_name", "unknown")
        max_steps = msg.get("max_steps", MAX_TRAIN_STEPS)

        self._current_object = object_name
        self._max_steps = max_steps
        self._current_step = 0
        self._stop_requested.clear()
        self._scanning = True

        self._scan_thread = threading.Thread(
            target=self._scan_worker,
            args=(object_name, max_steps),
            daemon=True,
        )
        self._scan_thread.start()

        logger.info("Scan started: object=%s, max_steps=%d", object_name, max_steps)
        return {"status": "started", "object_name": object_name}

    def _cmd_stop(self) -> dict:
        if not self._scanning:
            return {"status": "stopped", "message": "no scan in progress"}

        self._stop_requested.set()
        if self._scan_thread is not None:
            self._scan_thread.join(timeout=10.0)

        self._scanning = False
        logger.info(
            "Scan stopped: object=%s at step %d",
            self._current_object,
            self._current_step,
        )
        return {"status": "stopped", "step": self._current_step}

    def _cmd_status(self) -> dict:
        if self._scanning:
            return {
                "status": "scanning",
                "object_name": self._current_object,
                "step": self._current_step,
                "max_steps": self._max_steps,
            }
        return {"status": "idle"}

    # ------------------------------------------------------------------
    # Scan worker
    # ------------------------------------------------------------------

    def _scan_worker(self, object_name: str, max_steps: int) -> None:
        """Run Monty's learning loop on a live turntable scan (worker thread).

        Captures frames from the camera via the ZMQ frame server in ros2_comp,
        processes each frame through SAM2 + VGGT, builds Monty-compatible
        observations, and passes them to model.step().
        """
        try:
            from tbp.monty.context import RuntimeContext

            from monty_ext.configs.turntable_train import CONFIGS
            from monty_ext.sensor_modules.realsense_sm import RealsenseCapture
            from monty_ext.vision.sam2_segmenter import SAM2Segmenter
            from monty_ext.vision.vggt_provider import VGGTProvider

            config = copy.deepcopy(CONFIGS["turntable_pretrain_base"])
            config["turntable_config"]["object_name"] = object_name
            config["max_train_steps"] = max_steps
            config["logging"]["output_dir"] = f"/results/turntable/{object_name}"

            experiment_class = config.pop("experiment_class")
            experiment = experiment_class(config)
            experiment.setup_experiment(config)

            model = experiment.model
            primary_target = {"object": object_name}
            model.pre_episode(primary_target)
            model.switch_to_exploratory_step()
            model.detected_object = object_name
            for lm in model.learning_modules:
                lm.detected_object = object_name

            camera = RealsenseCapture()
            camera.start()
            segmenter = SAM2Segmenter()
            vggt = VGGTProvider()

            ctx = RuntimeContext(rng=np.random.RandomState(42))

            for step in range(max_steps):
                if self._stop_requested.is_set():
                    break

                self._current_step = step

                rgb = camera.capture()
                mask = segmenter.segment(rgb)
                result = vggt.process_batch([rgb], masks=[mask])

                observation = self._build_observation(
                    rgb, mask, result.depths[0], result.extrinsics[0]
                )

                model.step(ctx, observation)

                if step % 50 == 0:
                    logger.info(
                        "Scan progress: %s step %d/%d",
                        object_name,
                        step,
                        max_steps,
                    )

            model.post_episode()
            experiment.save_state_dict()
            camera.stop()

            logger.info(
                "Scan complete: %s — %d steps",
                object_name,
                self._current_step,
            )

        except Exception:
            logger.exception("Scan worker failed for %s", object_name)
        finally:
            self._scanning = False

    @staticmethod
    def _build_observation(
        rgb: np.ndarray,
        mask: np.ndarray,
        depth: np.ndarray,
        extrinsic: np.ndarray,
    ) -> dict:
        """Build a Monty-compatible observation dict from a single frame."""
        import cv2

        PATCH_SIZE = 70
        VOID_DEPTH = 10.0

        h, w = rgb.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = mask.astype(np.uint8) * 255

        cy, cx = h // 2, w // 2
        y0 = max(0, cy - PATCH_SIZE // 2)
        x0 = max(0, cx - PATCH_SIZE // 2)

        rgba_patch = rgba[y0 : y0 + PATCH_SIZE, x0 : x0 + PATCH_SIZE]

        if depth.shape[:2] != (h, w):
            depth_full = cv2.resize(
                depth.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
            )
        else:
            depth_full = depth.astype(np.float32)

        depth_patch = depth_full[y0 : y0 + PATCH_SIZE, x0 : x0 + PATCH_SIZE].copy()
        mask_patch = mask[y0 : y0 + PATCH_SIZE, x0 : x0 + PATCH_SIZE]
        depth_patch[~mask_patch] = VOID_DEPTH

        return {
            "agent_id_0": {
                "patch": {
                    "rgba": rgba_patch,
                    "depth": depth_patch,
                }
            }
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        if self._scan_thread is not None and self._scan_thread.is_alive():
            self._stop_requested.set()
            self._scan_thread.join(timeout=5.0)
        if self._socket is not None:
            self._socket.close()
        if self._ctx is not None:
            self._ctx.term()
        logger.info("Scan orchestrator shut down")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    orchestrator = ScanOrchestrator()

    def _signal_handler(signum, frame):
        logger.info("Received signal %d, shutting down...", signum)
        orchestrator.shutdown()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    orchestrator.run()


if __name__ == "__main__":
    main()
