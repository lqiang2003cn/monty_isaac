"""Monty experiment configuration for turntable-based object learning.

Uses the upstream tbp.monty class hierarchy directly — no fork-specific
classes.  The experiment class is ``TurntableTrainExperiment`` which
overrides ``load_environment_interfaces`` (so we don't need
``env_interface_config`` here).

Usage::

    from monty_ext.configs.turntable_train import CONFIGS
    config = CONFIGS["turntable_pretrain_base"]
"""

from __future__ import annotations

from tbp.monty.frameworks.loggers.monty_handlers import BasicCSVStatsHandler
from tbp.monty.frameworks.models.feature_location_matching import FeatureGraphLM
from tbp.monty.frameworks.models.graph_matching import MontyForGraphMatching
from tbp.monty.frameworks.models.motor_system import MotorSystem
from tbp.monty.frameworks.models.sensor_modules import CameraSM

from monty_ext.environments.turntable_loader import (
    TurntableActionSampler,
    TurntableTrainingPolicy,
)
from monty_ext.experiments.turntable_experiment import TurntableTrainExperiment

AGENT_ID = "agent_id_0"
SENSOR_ID = "patch"
PATCH_SIZE = 70
MAX_TRAIN_STEPS = 2000


turntable_pretrain_base = dict(
    experiment_class=TurntableTrainExperiment,

    do_train=True,
    do_eval=False,
    max_train_steps=MAX_TRAIN_STEPS,
    max_eval_steps=500,
    max_total_steps=MAX_TRAIN_STEPS + 200,
    n_train_epochs=1,
    n_eval_epochs=0,
    min_lms_match=1,
    model_name_or_path="",
    seed=42,
    show_sensor_output=False,
    supervised_lm_ids=[],

    logging=dict(
        output_dir="/results/turntable",
        run_name="turntable_pretrain",
        python_log_level="INFO",
        python_log_to_file=True,
        python_log_to_stderr=True,
        monty_log_level="SILENT",
        monty_handlers=[BasicCSVStatsHandler],
        wandb_handlers=[],
    ),

    monty_config=dict(
        monty_class=MontyForGraphMatching,
        monty_args=dict(
            min_train_steps=3,
            min_eval_steps=3,
            num_exploratory_steps=MAX_TRAIN_STEPS,
            max_total_steps=MAX_TRAIN_STEPS + 200,
        ),

        learning_module_configs={
            "learning_module_0": dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances={
                        SENSOR_ID: dict(
                            hsv=[10, 20, 20],
                            principal_curvatures_log=[1, 1],
                        ),
                    },
                    graph_delta_thresholds={
                        SENSOR_ID: dict(
                            distance=0.001,
                        ),
                    },
                ),
            ),
        },

        sensor_module_configs={
            "sensor_module_0": dict(
                sensor_module_class=CameraSM,
                sensor_module_args=dict(
                    sensor_module_id=SENSOR_ID,
                    features=[
                        "pose_vectors",
                        "pose_fully_defined",
                        "on_object",
                        "hsv",
                        "principal_curvatures",
                        "principal_curvatures_log",
                    ],
                    save_raw_obs=False,
                    delta_thresholds=dict(
                        on_object=0,
                        distance=0.001,
                    ),
                ),
            ),
        },

        motor_system_config=dict(
            motor_system_class=MotorSystem,
            motor_system_args=dict(
                policy=TurntableTrainingPolicy(
                    action_sampler=TurntableActionSampler(),
                    agent_id=AGENT_ID,
                    max_steps=MAX_TRAIN_STEPS,
                ),
            ),
        ),

        sm_to_agent_dict={
            SENSOR_ID: AGENT_ID,
        },
        sm_to_lm_matrix=[[0]],
        lm_to_lm_matrix=None,
        lm_to_lm_vote_matrix=None,
    ),

    turntable_config=dict(
        scan_dir="/data/scans",
        object_name="unknown",
        description="",
        patch_size=PATCH_SIZE,
        frame_subsample=3,
        debug_vis_dir="",
    ),
)


CONFIGS = {
    "turntable_pretrain_base": turntable_pretrain_base,
}
