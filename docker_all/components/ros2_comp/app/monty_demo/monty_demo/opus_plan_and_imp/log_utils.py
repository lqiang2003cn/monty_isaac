"""
Shared logging utilities for the X3plus robot stack.

Provides rotating file loggers that:
  - Rotate at 5 MB, keeping 3 backups (~20 MB max per log)
  - Write to /tmp (tmpfs on Linux → RAM-backed, minimal latency)
  - Never propagate to the root logger (no accidental terminal spam)

Usage in a ROS node::

    from monty_demo.opus_plan_and_imp.log_utils import make_file_logger
    self._flog = make_file_logger("x3plus_planner", "/tmp/x3plus_planner.log")
    self._flog.info("detailed message that only goes to the file")
"""

import logging
from logging.handlers import RotatingFileHandler

_DEFAULT_MAX_BYTES = 5 * 1024 * 1024   # 5 MB per file
_DEFAULT_BACKUP_COUNT = 3               # keep 3 rotated copies
_LOG_FMT = "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def make_file_logger(
    name: str,
    path: str,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    backup_count: int = _DEFAULT_BACKUP_COUNT,
) -> logging.Logger:
    """Return a logger that writes ONLY to a rotating file.

    Safe to call multiple times with the same *name*: the handler is
    added only once.  Writes are synchronous but target /tmp (tmpfs),
    so latency is sub-millisecond and will not stall robot callbacks.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if not logger.handlers:
        handler = RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backup_count,
        )
        handler.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_LOG_DATEFMT))
        logger.addHandler(handler)
    return logger
