"""Motor policies for full-sphere camera sweep (elevation × azimuth grid).

Provides FullSphereSweepPolicy: a deterministic policy that sweeps elevation and
azimuth so the viewfinder visits the full sphere around the object (no 90° cap).
Used by cnn_like_monty_100_test for full surface coverage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tbp.monty.frameworks.actions.actions import (
    LookDown,
    LookUp,
    TurnLeft,
    TurnRight,
)
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.motor_policies import (
    BasePolicy,
    MotorPolicyResult,
)
from tbp.monty.frameworks.models.motor_system_state import AgentState, MotorSystemState

if TYPE_CHECKING:
    from tbp.monty.context import RuntimeContext
    from tbp.monty.frameworks.models.abstract_monty_classes import Observations


def _build_full_sphere_actions(
    agent_id: AgentID,
    elevation_step_deg: float,
    azimuth_step_deg: float,
) -> list:
    """Build a list of LookUp/LookDown/TurnLeft/TurnRight that sweep full sphere.

    Order: start (0°, 0°). Sweep azimuth 0→360 at elevation 0; step elevation up;
    sweep 360→0; repeat until elevation 90°. Then step elevation down to -90°,
    sweep azimuth at -90°; step elevation back up to 0°, sweeping azimuth
    alternating direction (boustrophedon). Every step uses the given degrees.
    """
    assert elevation_step_deg > 0 and azimuth_step_deg > 0
    actions = []

    def add_look_up(n: int) -> None:
        for _ in range(n):
            actions.append(LookUp(agent_id=agent_id, rotation_degrees=elevation_step_deg))

    def add_look_down(n: int) -> None:
        for _ in range(n):
            actions.append(
                LookDown(agent_id=agent_id, rotation_degrees=elevation_step_deg)
            )

    def add_turn_left(n: int) -> None:
        for _ in range(n):
            actions.append(
                TurnLeft(agent_id=agent_id, rotation_degrees=azimuth_step_deg)
            )

    def add_turn_right(n: int) -> None:
        for _ in range(n):
            actions.append(
                TurnRight(agent_id=agent_id, rotation_degrees=azimuth_step_deg)
            )

    n_az = max(1, int(360.0 / azimuth_step_deg))
    n_elev_up = max(1, int(90.0 / elevation_step_deg))
    n_elev_down = max(1, int(180.0 / elevation_step_deg))

    # 0° elevation: sweep azimuth 0 → 360
    add_turn_left(n_az)

    # Elevation 0 → 90°: at each level sweep azimuth, then one step up
    for _ in range(n_elev_up - 1):
        add_look_up(1)
        add_turn_right(n_az)
        add_look_up(1)
        add_turn_left(n_az)

    add_look_up(1)
    # Now at elevation 90°. Sweep azimuth 0 → 360
    add_turn_left(n_az)

    # Elevation 90° → -90° (down)
    add_look_down(n_elev_down)

    # At -90°. Sweep azimuth 0 → 360
    add_turn_left(n_az)

    # Elevation -90° → 0°: at each level sweep azimuth (reverse), then one step up
    for _ in range(n_elev_up - 1):
        add_look_up(1)
        add_turn_right(n_az)
        add_look_up(1)
        add_turn_left(n_az)

    add_look_up(1)
    # Back at elevation 0. One more reverse azimuth sweep to return to 0°
    add_turn_right(n_az)

    return actions


class FullSphereSweepPolicy(BasePolicy):
    """Policy that sweeps elevation and azimuth so the viewfinder covers the full sphere.

    Uses a boustrophedon grid: sweep azimuth 0→360 at elevation 0, step up,
    sweep 360→0, etc., up to 90°; then step down to -90°, sweep; then step
    back up to 0°. No 90° cap — full latitude/longitude coverage.

    Requires an action_sampler for BasePolicy; it is not used (actions come from
    the precomputed sequence). Pass a dummy sampler, e.g. ConstantSampler with
    any actions.
    """

    def __init__(
        self,
        agent_id: AgentID,
        action_sampler: Any,
        elevation_step_deg: float = 5.0,
        azimuth_step_deg: float | None = None,
    ) -> None:
        """Initialize the policy.

        Args:
            agent_id: Agent to control.
            action_sampler: Required by BasePolicy; not used (sequence is fixed).
            elevation_step_deg: Step size in degrees for LookUp/LookDown.
            azimuth_step_deg: Step size in degrees for TurnLeft/TurnRight.
                Defaults to elevation_step_deg.
        """
        super().__init__(action_sampler=action_sampler, agent_id=agent_id)
        if azimuth_step_deg is None:
            azimuth_step_deg = elevation_step_deg
        self._elevation_step_deg = elevation_step_deg
        self._azimuth_step_deg = azimuth_step_deg
        self._action_list = _build_full_sphere_actions(
            agent_id=agent_id,
            elevation_step_deg=elevation_step_deg,
            azimuth_step_deg=azimuth_step_deg,
        )
        self._index = 0

    def __call__(
        self,
        ctx: "RuntimeContext",
        observations: "Observations",
        state: MotorSystemState | None = None,
    ) -> MotorPolicyResult:
        if self._index >= len(self._action_list):
            raise StopIteration(
                "FullSphereSweepPolicy: full sphere sweep complete "
                f"({len(self._action_list)} actions)"
            )
        action = self._action_list[self._index]
        self._index += 1
        return MotorPolicyResult([action])

    def pre_episode(self) -> None:
        self._index = 0

    def get_agent_state(self, state: MotorSystemState) -> AgentState:
        return state[self.agent_id]

    def state_dict(self) -> dict[str, Any]:
        return {"index": self._index}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._index = state_dict.get("index", 0)

    @property
    def num_actions(self) -> int:
        """Total number of actions in one full sweep (for config max_train_steps)."""
        return len(self._action_list)
