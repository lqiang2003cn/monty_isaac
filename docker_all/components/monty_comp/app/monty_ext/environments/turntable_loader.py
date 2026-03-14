"""Motor policy and action classes for turntable-based learning.

Provides a ``TurntableTrainingPolicy`` that extends tbp.monty's
``BasePolicy``, along with a trivial ``NextFrame`` action.  The policy
simply advances one frame per step and raises ``StopIteration`` when the
video is exhausted.

These classes follow the same pattern as the everything_is_awesome
``EverythingIsAwesomeTrainingPolicy`` and ``OrbitRight`` action.
"""

from __future__ import annotations

import logging
from typing import Any

from numpy.random import RandomState

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.actions.action_samplers import ActionSampler
from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_policies import (
    BasePolicy,
    MotorPolicyResult,
)
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    MotorSystemState,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class NextFrameActionSampler:
    """Protocol-style sampler for NextFrame actions."""

    def sample_next_frame(self, agent_id: AgentID, rng: RandomState) -> "NextFrame":
        return NextFrame(agent_id=agent_id)


class NextFrame:
    """Trivial action: advance to the next video frame.

    The ``VideoTurntableEnvironment`` ignores actions entirely (the video
    determines the trajectory), but the upstream ``MotorSystem`` /
    ``EnvironmentInterface`` pipeline requires an ``Action``-compatible
    object.
    """

    agent_id: AgentID

    def __init__(self, agent_id: AgentID) -> None:
        self.agent_id = agent_id

    @classmethod
    def action_name(cls) -> str:
        return "next_frame"

    @classmethod
    def sample(
        cls,
        agent_id: AgentID,
        sampler: NextFrameActionSampler,
        rng: RandomState,
    ) -> "NextFrame":
        return sampler.sample_next_frame(agent_id, rng)

    def act(self, actuator: Any = None) -> None:
        pass


# ---------------------------------------------------------------------------
# Action sampler
# ---------------------------------------------------------------------------

class TurntableActionSampler(ActionSampler):
    """ActionSampler that only produces NextFrame actions."""

    def __init__(self):
        super().__init__(actions=[])

    def sample(self, agent_id: AgentID, rng: RandomState) -> NextFrame:
        return NextFrame(agent_id=agent_id)

    def sample_next_frame(self, agent_id: AgentID, rng: RandomState) -> NextFrame:
        return NextFrame(agent_id=agent_id)


# ---------------------------------------------------------------------------
# Motor policy
# ---------------------------------------------------------------------------

class TurntableTrainingPolicy(BasePolicy):
    """Deterministic training policy for turntable scans.

    Advances one frame per step.  Raises ``StopIteration`` after
    ``max_steps`` frames, signalling the experiment to end the episode.
    This is exactly how EIA's ``EverythingIsAwesomeTrainingPolicy`` works.
    """

    def __init__(
        self,
        action_sampler: ActionSampler | None = None,
        agent_id: AgentID = "agent_id_0",
        max_steps: int = 1000,
    ):
        if action_sampler is None:
            action_sampler = TurntableActionSampler()
        super().__init__(action_sampler=action_sampler, agent_id=agent_id)
        self.use_goal_state_driven_actions = False
        self._max_steps = max_steps
        self._step = 0

    def __call__(
        self,
        ctx: RuntimeContext,
        observations: Observations,
        state: MotorSystemState | None = None,
    ) -> MotorPolicyResult:
        if self._step >= self._max_steps:
            raise StopIteration("TurntableTrainingPolicy: max_steps reached")

        self._step += 1
        action = NextFrame(agent_id=self.agent_id)
        return MotorPolicyResult(actions=[action])

    def pre_episode(self) -> None:
        self._step = 0

    def get_agent_state(self, state: MotorSystemState) -> AgentState:
        return state[self.agent_id]

    def state_dict(self) -> dict[str, Any]:
        return {"step": self._step, "max_steps": self._max_steps}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._step = state_dict.get("step", 0)
        self._max_steps = state_dict.get("max_steps", self._max_steps)
