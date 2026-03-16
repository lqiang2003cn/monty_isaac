"""Environment interface that injects primary_target into the transform context.

Used with WorldToObjectFrame so 3D points are converted to object-relative coordinates
and learned graphs form a consistent object shape (e.g. a proper mug) instead of
a mixed shell from many world poses.
"""

from __future__ import annotations

from typing import Iterable

from tbp.monty.frameworks.environment_utils.transforms import TransformContext
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_system_state import (
    MotorSystemState,
    ProprioceptiveState,
)

from tbp.monty.frameworks.environments.embodied_data import EnvironmentInterfacePerObject


class EnvironmentInterfaceWithObjectFrame(EnvironmentInterfacePerObject):
    """Like EnvironmentInterfacePerObject but adds primary_target to transform context.

    Transforms (e.g. WorldToObjectFrame) can then convert world 3D to object-relative
    frame so graph memory stores a canonical object shape.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.change_object_by_idx(0)
        self._observations, self._proprioceptive_state = self.reset(self.rng)
        self.motor_system._state = MotorSystemState(self._proprioceptive_state)

    def apply_transform(
        self, transform, observations: Observations, state: ProprioceptiveState
    ) -> Observations:
        ctx = TransformContext(rng=self.rng, state=state)
        ctx.primary_target = self.primary_target
        if isinstance(transform, Iterable):
            for t in transform:
                observations = t(observations, ctx)
        else:
            observations = transform(observations, ctx)
        return observations
