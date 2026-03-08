"""
Shared logging utilities for the X3plus robot stack.

Provides **non-blocking** rotating file loggers that:
  - Never stall the calling thread (QueueHandler → background writer)
  - Rotate at 5 MB, keeping 3 backups (~20 MB max per log)
  - Write to LOG_DIR (/var/log/monty), bind-mounted to host ./logs/
  - Never propagate to the root logger (no accidental terminal spam)
  - Silently drop records if the queue is full (bounded at 10 000)

Usage in a ROS node::

    from monty_demo.opus_plan_and_imp.log_utils import make_file_logger, LOG_DIR
    self._flog = make_file_logger("x3plus_planner", f"{LOG_DIR}/x3plus_planner.log")
    self._flog.info("detailed message that only goes to the file")
"""

import atexit
import logging
import logging.handlers
import os
import queue

LOG_DIR = "/var/log/monty"

_DEFAULT_MAX_BYTES = 5 * 1024 * 1024   # 5 MB per file
_DEFAULT_BACKUP_COUNT = 3               # keep 3 rotated copies
_QUEUE_MAX = 10_000                     # bounded queue; drop if full
_LOG_FMT = "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

_listeners: list[logging.handlers.QueueListener] = []


def _shutdown_listeners():
    for listener in _listeners:
        listener.stop()


atexit.register(_shutdown_listeners)


class _DropQueueHandler(logging.handlers.QueueHandler):
    """QueueHandler that silently drops records when the queue is full."""

    def enqueue(self, record):
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            pass


def make_file_logger(
    name: str,
    path: str,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    backup_count: int = _DEFAULT_BACKUP_COUNT,
) -> logging.Logger:
    """Return a logger that writes ONLY to a rotating file, non-blocking.

    The caller's thread never touches the filesystem.  Log records are
    placed into a bounded in-memory queue (``queue.put_nowait``); a daemon
    thread drains the queue and writes to the rotating file.  If the queue
    is full (I/O stalled), new records are silently dropped rather than
    blocking the robot control loop.

    Safe to call multiple times with the same *name*: handlers are added
    only once.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if not logger.handlers:
        file_handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backup_count,
        )
        file_handler.setFormatter(
            logging.Formatter(_LOG_FMT, datefmt=_LOG_DATEFMT)
        )
        q: queue.Queue[logging.LogRecord] = queue.Queue(maxsize=_QUEUE_MAX)
        queue_handler = _DropQueueHandler(q)
        listener = logging.handlers.QueueListener(
            q, file_handler, respect_handler_level=True,
        )
        listener.start()
        _listeners.append(listener)
        logger.addHandler(queue_handler)
    return logger
