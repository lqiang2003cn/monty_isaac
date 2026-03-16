"""Transform world 3D coordinates to object-relative (canonical) frame.

Runs after DepthTo3DLocations and CropPatchesFromViewFinder. Converts semantic_3d
and sensor_frame_data from world coordinates to the primary target object's frame
so that learned graphs form a consistent object shape instead of a mixed shell
from many poses.

Requires ctx.primary_target (position, rotation, semantic_id) to be set by the
environment interface (e.g. EnvironmentInterfaceWithObjectFrame).
"""

from __future__ import annotations

import numpy as np
import quaternion as qt

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.abstract_monty_classes import Observations


class WorldToObjectFrame:
    """Convert world 3D points to primary target object frame.

    Modifies semantic_3d and sensor_frame_data in place for view_finder and all
    patches. Expects ctx.primary_target with keys: position, rotation, semantic_id.
    If primary_target is None or missing, no-op.
    """

    def __init__(self, agent_id: AgentID):
        self.agent_id = agent_id

    def __call__(self, observations: Observations, ctx) -> Observations:
        primary = getattr(ctx, "primary_target", None)
        if primary is None:
            return observations
        position = primary.get("position")
        rotation = primary.get("rotation")
        semantic_id = primary.get("semantic_id")
        if position is None or rotation is None or semantic_id is None:
            return observations
        position = np.asarray(position, dtype=np.float64).reshape(3)
        R = qt.as_rotation_matrix(rotation)

        for _sensor_id, obs in observations[self.agent_id].items():
            for key in ("semantic_3d", "sensor_frame_data"):
                if key not in obs:
                    continue
                arr = obs[key]
                if arr.size == 0:
                    continue
                on_obj = arr[:, 3] == semantic_id
                if not np.any(on_obj):
                    continue
                pts = np.asarray(arr[on_obj, :3].copy(), dtype=np.float64)
                transformed = (R.T @ (pts - position).T).T
                arr[on_obj, :3] = transformed
        return observations
