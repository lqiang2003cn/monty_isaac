"""DEPRECATED — the vision server now runs in the vision_comp container.

See ``docker_all/components/vision_comp/app/vision_server/server.py``.

This file is no longer used.  The ``_bridge.py`` module connects to the
server over TCP instead of spawning a subprocess.
"""

raise ImportError(
    "monty_ext.vision._server has been moved to the vision_comp container. "
    "See docker_all/components/vision_comp/app/vision_server/server.py"
)
