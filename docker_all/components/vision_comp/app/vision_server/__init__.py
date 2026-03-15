"""Vision model server — SAM2, VGGT, Grounding DINO.

Runs as a TCP JSON-lines server in its own container (vision_comp).
Clients (monty_comp) connect via ``_bridge.py``.
"""
