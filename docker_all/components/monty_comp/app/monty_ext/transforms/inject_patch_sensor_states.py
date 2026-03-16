"""Inject synthetic sensor states for patch_0..patch_N so agent state matches observations.

When using CropPatchesFromViewFinder, observations have patch_0..patch_N but the
environment only provides view_finder in the agent state. Sensor modules (CameraSM)
look up agent.sensors[SensorID(sensor_module_id)], so we must add patch_i entries
to the proprioceptive state. This transform duplicates view_finder's sensor state
for each patch (same physical camera).
"""

from __future__ import annotations

import copy

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_system_state import SensorState
from tbp.monty.frameworks.sensors import SensorID


class InjectPatchSensorStates:
    """Add patch_0..patch_{n_patches-1} to ctx.state[agent_id].sensors (copy of view_finder)."""

    def __init__(self, agent_id: AgentID, n_patches: int):
        self.agent_id = agent_id
        self.n_patches = n_patches

    def __call__(self, observations: Observations, ctx) -> Observations:
        if ctx.state is None:
            return observations
        agent_state = ctx.state.get(self.agent_id)
        if agent_state is None:
            return observations
        vf_id = SensorID("view_finder")
        if vf_id not in agent_state.sensors:
            return observations
        vf_state = agent_state.sensors[vf_id]
        for i in range(self.n_patches):
            pid = SensorID(f"patch_{i}")
            if pid not in agent_state.sensors:
                agent_state.sensors[pid] = SensorState(
                    position=copy.copy(vf_state.position),
                    rotation=vf_state.rotation,
                )
        return observations
