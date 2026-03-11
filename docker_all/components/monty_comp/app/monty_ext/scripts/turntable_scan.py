"""Main entry point for the turntable scanning pipeline.

Runs inside monty_comp.  Listens for start/stop commands from ros2_comp via
ZMQ (tcp://*:5560), then drives Monty's learning loop through the
TurntableDataLoader.

Protocol (JSON over ZMQ REP socket):

    → {"action": "start", "object_name": "cup"}
    ← {"status": "started", "object_name": "cup"}

    → {"action": "status"}
    ← {"status": "scanning", "step": 42, "max_steps": 1000}
    ← {"status": "idle"}

    → {"action": "stop"}
    ← {"status": "stopped"}

Usage (inside container)::

    conda activate tbp.monty
    turntable-scan              # uses pyproject.toml entry point
    python -m monty_ext.scripts.turntable_scan   # alternative
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from typing import Optional

import zmq

from monty_ext.environments.turntable_env import TurntableConfig
from monty_ext.environments.turntable_loader import TurntableDataLoader

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
        logger.info("Scan stopped: object=%s at step %d", self._current_object, self._current_step)
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
        """Run Monty's learning loop on a turntable scan (worker thread)."""
        try:
            loader = TurntableDataLoader(
                env_config=TurntableConfig(),
                max_train_steps=max_steps,
                object_name=object_name,
            )

            for step, observation in enumerate(loader):
                if self._stop_requested.is_set():
                    break
                self._current_step = step

                # In the future, pass observation to Monty's model here:
                #   self._monty_model.step(observation)
                # For now, log progress.
                if step % 50 == 0:
                    logger.info(
                        "Scan progress: %s step %d/%d",
                        object_name,
                        step,
                        max_steps,
                    )

            loader.finish()
            logger.info(
                "Scan complete: %s — %d steps",
                object_name,
                self._current_step,
            )
        except Exception:
            logger.exception("Scan worker failed for %s", object_name)
        finally:
            self._scanning = False

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
