"""CameraSM subclass for virtual patches cropped from a physical sensor.

In the CNN-style multi-patch setup, 100 virtual patches are cropped from
a single ``view_finder`` sensor.  Each ``PatchCameraSM`` uses the
``view_finder`` sensor's physical state (position, rotation) rather than
trying to look up its own ``sensor_module_id`` (e.g. ``patch_42``) in the
agent's sensor registry.
"""

from __future__ import annotations

from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.frameworks.models.sensor_modules import CameraSM, SensorState

try:
    import quaternion as qt
except ImportError:
    qt = None


class PatchCameraSM(CameraSM):
    """CameraSM that derives its spatial state from a parent physical sensor."""

    def __init__(self, parent_sensor_id: str = "view_finder", **kwargs):
        super().__init__(**kwargs)
        self.parent_sensor_id = parent_sensor_id

    def update_state(self, agent):
        sensor = agent.sensors[SensorID(self.parent_sensor_id)]
        self.state = SensorState(
            position=agent.position
            + qt.rotate_vectors(agent.rotation, sensor.position),
            rotation=agent.rotation * sensor.rotation,
        )
