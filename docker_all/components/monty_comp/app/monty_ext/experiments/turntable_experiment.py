"""Custom MontyExperiment subclass for turntable-based object learning.

Follows the same pattern as everything_is_awesome's
``EverythingIsAwesomeTrainExperiment``:

- Overrides ``load_environment_interfaces`` to wire our
  ``VideoTurntableEnvironment`` into the upstream ``EnvironmentInterface``
  (bypassing the Habitat-centric default setup).
- Overrides ``pre_episode`` to pass ``primary_target`` and switch the model
  into exploratory training mode.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import quaternion as qt

from tbp.monty.frameworks.environment_utils.transforms import DepthTo3DLocations
from tbp.monty.frameworks.environments.embodied_data import EnvironmentInterface
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.experiments.monty_experiment import MontyExperiment

from monty_ext.environments.video_turntable_env import (
    VideoTurntableConfig,
    VideoTurntableEnvironment,
)

logger = logging.getLogger(__name__)


class TurntableTrainExperiment(MontyExperiment):
    """Experiment for training Monty from pre-recorded turntable scan videos."""

    def __init__(self, config):
        super().__init__(config)
        self._turntable_config = config.get("turntable_config", {})
        self._primary_target = {
            "object": self._turntable_config.get("object_name", "unknown"),
            "rotation": qt.from_float_array([1.0, 0.0, 0.0, 0.0]),
            "euler_rotation": np.array([0, 0, 0]),
            "quat_rotation": [0, 0, 0, 1],
            "position": np.array([0, 0, 0]),
            "scale": [1.0, 1.0, 1.0],
        }

    @property
    def logger_args(self):
        args = super().logger_args
        args["target"] = self._primary_target
        return args

    def load_environment_interfaces(self, config):
        """Create VideoTurntableEnvironment and wire it into EnvironmentInterface.

        Bypasses the standard Habitat-based setup entirely.
        """
        tc = self._turntable_config
        object_name = tc.get("object_name", "unknown")
        scan_dir = tc.get("scan_dir", "/data/scans")
        patch_size = tc.get("patch_size", 70)
        frame_subsample = tc.get("frame_subsample", 5)
        description = tc.get("description", "")
        debug_vis_dir = tc.get("debug_vis_dir", "")

        full_scan_dir = f"{scan_dir}/{object_name}"

        env_config = VideoTurntableConfig(
            scan_dir=full_scan_dir,
            object_name=object_name,
            description=description,
            patch_size=patch_size,
            frame_subsample=frame_subsample,
            debug_vis_dir=debug_vis_dir,
        )
        self.env = VideoTurntableEnvironment(config=env_config)

        hfov = self.env.get_hfov()

        transform = DepthTo3DLocations(
            agent_id="agent_id_0",
            sensor_ids=["patch"],
            resolutions=[(patch_size, patch_size)],
            hfov=hfov,
            get_all_points=True,
            use_semantic_sensor=False,
            world_coord=True,
        )

        if config.get("do_train", True):
            self.train_env_interface = EnvironmentInterface(
                env=self.env,
                motor_system=self.model.motor_system,
                rng=self.rng,
                seed=self.config["seed"],
                experiment_mode=ExperimentMode.TRAIN,
                transform=[transform],
            )
        else:
            self.train_env_interface = None

        if config.get("do_eval", False):
            self.eval_env_interface = EnvironmentInterface(
                env=self.env,
                motor_system=self.model.motor_system,
                rng=self.rng,
                seed=self.config["seed"],
                experiment_mode=ExperimentMode.EVAL,
                transform=[transform],
            )
        else:
            self.eval_env_interface = None

    def pre_episode(self):
        """Pass primary_target and switch to exploratory mode for training.

        This mirrors the EverythingIsAwesomeTrainExperiment pattern.
        """
        if self.experiment_mode is ExperimentMode.TRAIN:
            logger.info(
                "running train epoch %d, train episode %d",
                self.train_epochs,
                self.train_episodes,
            )
        else:
            logger.info(
                "running eval epoch %d, eval episode %d",
                self.eval_epochs,
                self.eval_episodes,
            )

        self.reset_episode_rng()

        self.model.pre_episode(self._primary_target)
        self.model.switch_to_exploratory_step()

        object_name = self._primary_target["object"]
        self.model.detected_object = object_name
        for lm in self.model.learning_modules:
            lm.detected_object = object_name

        self.env_interface.pre_episode(self.rng)

        self.max_steps = self.max_train_steps
        if self.experiment_mode is not ExperimentMode.TRAIN:
            self.max_steps = self.max_eval_steps

        self.logger_handler.pre_episode(self.logger_args)
