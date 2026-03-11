"""Monty experiment configuration for turntable supervised pretraining.

Mirrors the structure of everything_is_awesome/benchmarks/configs/my_experiments.py.
Each config is a plain dict that can be passed to Monty's experiment runner.

Usage (inside monty_comp container)::

    conda activate tbp.monty
    python -m tbp.monty.run -e turntable_pretrain_base

Or programmatically::

    from monty_ext.configs.turntable_train import CONFIGS
    config = CONFIGS["turntable_pretrain_base"]
"""

from __future__ import annotations

from monty_ext.environments.turntable_env import TurntableConfig
from monty_ext.environments.turntable_loader import TurntableDataLoader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENT_ID = "agent_id_0"
SENSOR_ID = "patch"
PATCH_SIZE = 64

MAX_TRAIN_STEPS = 1000

# ---------------------------------------------------------------------------
# Reusable sub-configs
# ---------------------------------------------------------------------------


def make_turntable_env_config(**overrides) -> TurntableConfig:
    defaults = dict(
        realsense_width=1280,
        realsense_height=720,
        realsense_fps=30,
        vggt_batch_size=5,
        vggt_resolution=518,
        vggt_device="cuda",
        vggt_max_batch=8,
        sam2_device="cuda",
        sam2_points_per_side=32,
        patch_size=PATCH_SIZE,
        sensor_id=SENSOR_ID,
        agent_id=AGENT_ID,
    )
    defaults.update(overrides)
    return TurntableConfig(**defaults)


def make_train_dataloader_config(
    object_name: str = "unknown",
    max_steps: int = MAX_TRAIN_STEPS,
    env_overrides: dict | None = None,
) -> dict:
    return dict(
        env_config=make_turntable_env_config(**(env_overrides or {})),
        max_train_steps=max_steps,
        object_name=object_name,
    )


# ---------------------------------------------------------------------------
# Experiment configs
# ---------------------------------------------------------------------------

turntable_pretrain_base = dict(
    experiment_class="tbp.monty.frameworks.experiments.monty_experiment.MontyExperiment",
    experiment_args=dict(
        do_eval=False,
        do_train=True,
        n_train_epochs=1,
        max_train_steps=MAX_TRAIN_STEPS,
        max_total_steps=MAX_TRAIN_STEPS,
    ),
    logging_config=dict(
        output_dir="/results/turntable",
        run_name="turntable_pretrain",
    ),
    # Monty model config — uses default surface agent with one patch sensor.
    monty_config=dict(
        monty_class="tbp.monty.frameworks.models.monty_base.MontyBase",
        monty_args=dict(),
        learning_module_configs={
            "learning_module_0": dict(
                learning_module_class=(
                    "tbp.monty.frameworks.models.graph_matching.GraphLM"
                ),
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances={"patch.rgba": 20.0},
                ),
            ),
        },
        sensor_module_configs={
            "sensor_module_0": dict(
                sensor_module_class=(
                    "tbp.monty.frameworks.models.sensor_modules.HabitatSurfacePatchSM"
                ),
                sensor_module_args=dict(
                    sensor_module_id=SENSOR_ID,
                    features=["pose_vectors", "pose_fully_defined", "on_object"],
                    save_raw_obs=False,
                ),
            ),
        },
        motor_system_config=dict(
            motor_system_class=(
                "tbp.monty.frameworks.models.motor_system.MotorSystem"
            ),
        ),
    ),
    train_dataloader_class=f"{TurntableDataLoader.__module__}.{TurntableDataLoader.__qualname__}",
    train_dataloader_args=make_train_dataloader_config(),
)


CONFIGS = {
    "turntable_pretrain_base": turntable_pretrain_base,
}
