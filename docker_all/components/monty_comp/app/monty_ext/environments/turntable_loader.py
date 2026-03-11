"""TurntableDataLoader — drives Monty's sensorimotor loop for turntable scanning.

Mirrors the EnvironmentDataLoader from tbp.monty but adapted for the turntable
use case where:
- The camera is static; the turntable rotates the object.
- Actions are just "next frame" signals (no motor commands).
- The training policy is a simple deterministic sweep.

Reference: tbp.monty.frameworks.environments.embodied_data.EnvironmentDataLoader
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple

from monty_ext.environments.turntable_env import (
    TurntableConfig,
    TurntableEnvironment,
    TurntableObservations,
)

logger = logging.getLogger(__name__)


class TurntableAction:
    """Trivial action: signal the environment to capture the next frame."""

    def __init__(self, agent_id: str = "agent_id_0"):
        self.agent_id = agent_id

    def act(self, actuator=None):
        pass


class TurntableTrainingPolicy:
    """Deterministic scan policy for turntable learning.

    Since the turntable handles object rotation, the policy simply issues
    "next frame" actions until ``max_steps`` is reached, then raises
    ``StopIteration``.

    Modelled after EverythingIsAwesomeTrainingPolicy.
    """

    def __init__(
        self,
        agent_id: str = "agent_id_0",
        max_steps: int = 1000,
    ):
        self.agent_id = agent_id
        self.max_steps = max_steps
        self.use_goal_state_driven_actions = False
        self._step = 0

    def __call__(self, state=None) -> TurntableAction:
        return self.dynamic_call(state)

    def dynamic_call(self, state=None) -> TurntableAction:
        if self._step >= self.max_steps:
            raise StopIteration()
        self._step += 1
        return TurntableAction(agent_id=self.agent_id)

    def pre_episode(self):
        self._step = 0

    def post_episode(self):
        pass

    def set_experiment_mode(self, mode: str):
        pass

    @property
    def last_action(self):
        return TurntableAction(agent_id=self.agent_id)


class TurntableDataLoader:
    """DataLoader that drives the turntable scan loop.

    Works as an iterator: each ``next()`` call captures, segments, and
    (when batch is ready) runs VGGT, then returns the observation in
    Monty's expected format.

    Usage::

        loader = TurntableDataLoader(env_config=TurntableConfig(), ...)
        for observation in loader:
            model.step(observation)
    """

    def __init__(
        self,
        env_config: Optional[TurntableConfig] = None,
        max_train_steps: int = 1000,
        object_name: str = "unknown",
        transform=None,
    ):
        self._env_config = env_config or TurntableConfig()
        self._max_steps = max_train_steps
        self._object_name = object_name
        self._transform = transform if transform is not None else []

        self._env: Optional[TurntableEnvironment] = None
        self._policy: Optional[TurntableTrainingPolicy] = None
        self._is_first: bool = True
        self._initial_obs = None

    @property
    def primary_target(self):
        return self._object_name

    @primary_target.setter
    def primary_target(self, value):
        self._object_name = value

    # ------------------------------------------------------------------
    # Iterator protocol
    # ------------------------------------------------------------------

    def __iter__(self):
        self._env = TurntableEnvironment(self._env_config)
        self._policy = TurntableTrainingPolicy(
            agent_id=self._env_config.agent_id,
            max_steps=self._max_steps,
        )
        self._policy.pre_episode()

        obs = self._env.reset()
        state = self._env.get_state()
        obs = self._apply_transforms(obs, state)
        self._initial_obs = obs
        self._is_first = True
        return self

    def __next__(self):
        if self._is_first:
            self._is_first = False
            return self._initial_obs

        try:
            action = self._policy()
        except StopIteration:
            raise

        obs = self._env.step(action)
        state = self._env.get_state()
        obs = self._apply_transforms(obs, state)
        return obs

    def __len__(self):
        return math.inf

    # ------------------------------------------------------------------
    # Hooks called by MontyExperiment
    # ------------------------------------------------------------------

    def pre_episode(self):
        if self._policy is not None:
            self._policy.pre_episode()

    def post_episode(self):
        if self._policy is not None:
            self._policy.post_episode()

    def pre_epoch(self):
        pass

    def post_epoch(self):
        pass

    def finish(self):
        if self._env is not None:
            self._env.close()

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def _apply_transforms(self, obs, state):
        for t in self._transform:
            obs = t(obs, state)
        return obs
