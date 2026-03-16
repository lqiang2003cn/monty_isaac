"""Tutorial dist_agent_5lm_2obj_train replicated exactly, with one difference: 5 patches
come from cropping a single view_finder image instead of 5 separate Habitat sensors.

Same as: tbp.monty conf/experiment/tutorial/dist_agent_5lm_2obj_train.yaml
Difference: env renders one view_finder (64x320); transform crops into patch_0..patch_4 (64x64 each).

Usage::
  python -m monty_ext.configs.dist_agent_5lm_2obj_crop
  python -m monty_ext.configs.dist_agent_5lm_2obj_crop --output-dir /results/5lm_crop
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Same rotations as benchmarks.defaults.yaml rotations_all (14 rotations)
ROTATIONS_14 = [
    (0, 0, 0),
    (0, 90, 0),
    (0, 180, 0),
    (0, 270, 0),
    (90, 0, 0),
    (90, 180, 0),
    (35, 45, 0),
    (325, 45, 0),
    (35, 315, 0),
    (325, 315, 0),
    (35, 135, 0),
    (325, 135, 0),
    (35, 225, 0),
    (325, 225, 0),
]

AGENT_ID = "agent_id_0"
N_PATCHES = 5
PATCH_H, PATCH_W = 64, 64
# One row of 5 patches
VF_H, VF_W = PATCH_H, N_PATCHES * PATCH_W  # 64 x 320

# Same features as multi_lm_to_extract (tutorial)
FEATURES = [
    "on_object",
    "rgba",
    "hsv",
    "pose_vectors",
    "pose_fully_defined",
    "principal_curvatures",
    "principal_curvatures_log",
    "gaussian_curvature",
    "mean_curvature",
    "gaussian_curvature_sc",
    "mean_curvature_sc",
]

RESULTS_ROOT = os.environ.get("MONTY_RESULTS_ROOT", "/results")
OUTPUT_DIR_DEFAULT = os.path.join(RESULTS_ROOT, "dist_agent_5lm_2obj_crop")
MONTY_DATA = os.environ.get("MONTY_DATA", os.path.expanduser("~/tbp/data"))
HABITAT_DATA_PATH = os.path.join(MONTY_DATA, "habitat", "objects", "ycb")

_HabitatEnvironment = None
_MultiSensorAgent = None


def _import_habitat():
    global _HabitatEnvironment, _MultiSensorAgent
    if _HabitatEnvironment is not None:
        return
    try:
        from tbp.monty.simulators.habitat.environment import HabitatEnvironment as _HE
        from tbp.monty.simulators.habitat import MultiSensorAgent as _MSA
        _HabitatEnvironment = _HE
        _MultiSensorAgent = _MSA
    except ImportError as e:
        raise ImportError("Habitat required. Run inside monty_comp.") from e


def build_config(
    output_dir: str = OUTPUT_DIR_DEFAULT,
    object_names: list[str] | None = None,
    rotations: list[tuple[float, float, float]] | None = None,
    n_train_epochs: int | None = None,
    num_exploratory_steps: int = 500,
) -> dict:
    """Build config identical to tutorial YAML except view_finder + crop for 5 patches."""
    _import_habitat()

    if object_names is None:
        object_names = ["mug", "banana"]
    if rotations is None:
        rotations = list(ROTATIONS_14)
    if n_train_epochs is None:
        n_train_epochs = len(rotations)

    from tbp.monty.frameworks.environment_utils.transforms import (
        DepthTo3DLocations,
        MissingToMaxDepth,
    )
    from tbp.monty.frameworks.actions.actions import LookDown, LookUp, TurnLeft, TurnRight
    from tbp.monty.frameworks.actions.action_samplers import ConstantSampler
    from tbp.monty.frameworks.models.motor_policies import NaiveScanPolicy
    from tbp.monty.frameworks.models.motor_system import MotorSystem
    from tbp.monty.frameworks.models.sensor_modules import CameraSM, Probe
    from tbp.monty.frameworks.models.displacement_matching import DisplacementGraphLM
    from tbp.monty.frameworks.models.graph_matching import MontyForGraphMatching
    from tbp.monty.frameworks.loggers.monty_handlers import BasicCSVStatsHandler
    from tbp.monty.frameworks.environments.embodied_data import EnvironmentInterfacePerObject
    from tbp.monty.frameworks.environments.object_init_samplers import Predefined
    from monty_ext.transforms.crop_patches import CropPatchesFromViewFinder
    from monty_ext.transforms.inject_patch_sensor_states import InjectPatchSensorStates

    # Transform: same as tutorial but view_finder only then crop to 5 patches.
    # InjectPatchSensorStates adds patch_0..4 to agent state so CameraSM can look them up.
    transform = [
        MissingToMaxDepth(agent_id=AGENT_ID, max_depth=1.0),
        DepthTo3DLocations(
            agent_id=AGENT_ID,
            sensor_ids=["view_finder"],
            resolutions=[(VF_H, VF_W)],
            zooms=[1.0],
            world_coord=True,
            get_all_points=True,
        ),
        CropPatchesFromViewFinder(
            agent_id=AGENT_ID,
            n_rows=1,
            n_cols=N_PATCHES,
            patch_h=PATCH_H,
            patch_w=PATCH_W,
        ),
        InjectPatchSensorStates(agent_id=AGENT_ID, n_patches=N_PATCHES),
    ]

    # Motor: NaiveScanPolicy fixed_amount=5 (same as naive_scan_spiral.yaml)
    motor_system_config = {
        "motor_system_class": MotorSystem,
        "motor_system_args": {
            "policy": NaiveScanPolicy(
                fixed_amount=5.0,
                action_sampler=ConstantSampler(
                    actions=[LookUp, LookDown, TurnLeft, TurnRight],
                    rotation_degrees=5.0,
                ),
                agent_id=AGENT_ID,
            ),
        },
    }

    # Sensor modules: 5 CameraSM (patch_0..4) + Probe (view_finder) — same as five_lm.yaml
    sensor_module_configs = {}
    for i in range(N_PATCHES):
        sensor_module_configs[f"sensor_module_{i}"] = {
            "sensor_module_class": CameraSM,
            "sensor_module_args": {
                "sensor_module_id": f"patch_{i}",
                "features": list(FEATURES),
                "save_raw_obs": False,
            },
        }
    sensor_module_configs[f"sensor_module_{N_PATCHES}"] = {
        "sensor_module_class": Probe,
        "sensor_module_args": {
            "sensor_module_id": "view_finder",
            "save_raw_obs": False,
        },
    }

    # Learning modules: 5 DisplacementGraphLM k=5 match_attribute=displacement
    learning_module_configs = {
        f"learning_module_{i}": {
            "learning_module_class": DisplacementGraphLM,
            "learning_module_args": {"k": 5, "match_attribute": "displacement"},
        }
        for i in range(N_PATCHES)
    }

    sm_to_agent_dict = {"view_finder": AGENT_ID}
    for i in range(N_PATCHES):
        sm_to_agent_dict[f"patch_{i}"] = AGENT_ID

    sm_to_lm_matrix = [[i] for i in range(N_PATCHES)]
    lm_to_lm_vote_matrix = [
        [j for j in range(N_PATCHES) if j != i] for i in range(N_PATCHES)
    ]

    return {
        "experiment_class": TutorialPretrainingExperiment,
        "do_train": True,
        "do_eval": False,
        "max_train_steps": num_exploratory_steps,
        "max_eval_steps": 1,
        "max_total_steps": num_exploratory_steps,
        "n_train_epochs": n_train_epochs,
        "n_eval_epochs": 0,
        "min_lms_match": 1,
        "model_name_or_path": "",
        "seed": 42,
        "show_sensor_output": False,
        "supervised_lm_ids": "all",
        "logging": {
            "output_dir": output_dir,
            "run_name": "dist_agent_5lm_2obj_crop",
            "python_log_level": "INFO",
            "python_log_to_file": True,
            "python_log_to_stderr": True,
            "monty_log_level": "SILENT",
            "monty_handlers": [BasicCSVStatsHandler],
            "wandb_handlers": [],
        },
        "monty_config": {
            "monty_class": MontyForGraphMatching,
            "monty_args": {
                "num_exploratory_steps": num_exploratory_steps,
                "min_train_steps": 3,
                "min_eval_steps": 3,
                "max_total_steps": num_exploratory_steps,
            },
            "sensor_module_configs": sensor_module_configs,
            "learning_module_configs": learning_module_configs,
            "motor_system_config": motor_system_config,
            "sm_to_agent_dict": sm_to_agent_dict,
            "sm_to_lm_matrix": sm_to_lm_matrix,
            "lm_to_lm_matrix": None,
            "lm_to_lm_vote_matrix": lm_to_lm_vote_matrix,
        },
        "env_interface_config": {
            "env_init_func": _HabitatEnvironment,
            "env_init_args": {
                "agents": {
                    "agent_type": _MultiSensorAgent,
                    "agent_args": {
                        "agent_id": AGENT_ID,
                        "sensor_ids": ["view_finder"],
                        "resolutions": [(VF_H, VF_W)],
                        "zooms": [1.0],
                        "position": [0.0, 1.5, 0.2],
                        "height": 0.0,
                        "positions": [(0.0, 0.0, 0.0)],
                        "rotations": [(1.0, 0.0, 0.0, 0.0)],
                        "semantics": [False],
                    },
                },
                "data_path": HABITAT_DATA_PATH,
            },
            "transform": transform,
        },
        "train_env_interface_class": EnvironmentInterfacePerObject,
        "train_env_interface_args": {
            "object_names": object_names,
            "object_init_sampler": Predefined(rotations=rotations),
        },
    }


def _get_pretraining_base():
    from tbp.monty.frameworks.experiments.pretraining_experiments import (
        MontySupervisedObjectPretrainingExperiment,
    )
    return MontySupervisedObjectPretrainingExperiment


# Subclass that accepts plain-dict config (no OmegaConf.to_object)
class TutorialPretrainingExperiment(_get_pretraining_base()):
    """Same as MontySupervisedObjectPretrainingExperiment but __init__ skips OmegaConf.to_object.
    Overrides setup_experiment so sensor_pos has one entry per LM (5) when env only has view_finder.
    """

    def __init__(self, config):
        from pathlib import Path
        output_dir = Path(config["logging"]["output_dir"])
        config["logging"]["output_dir"] = output_dir / "pretrained"
        self.first_epoch_object_location = {}
        from tbp.monty.frameworks.experiments.monty_experiment import MontyExperiment
        MontyExperiment.__init__(self, config)

    def setup_experiment(self, config):
        super().setup_experiment(config)
        # Crop setup: env has 1 sensor (view_finder) but we have 5 LMs; pretraining expects
        # sensor_pos[i] per LM. Use one position repeated 5 times so lm_offset is zero.
        import numpy as np
        n_lms = len(config["monty_config"]["learning_module_configs"])
        self.sensor_pos = np.zeros((n_lms, 3))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run tutorial-equivalent 5LM pretraining with view_finder crop.",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--epochs", type=int, default=None, help="n_train_epochs")
    parser.add_argument("--steps", type=int, default=500, help="num_exploratory_steps")
    args = parser.parse_args()

    kwargs = {}
    if args.output_dir is not None:
        kwargs["output_dir"] = args.output_dir
    if args.epochs is not None:
        kwargs["n_train_epochs"] = args.epochs
    kwargs["num_exploratory_steps"] = args.steps

    config = build_config(**kwargs)
    experiment_class = config.pop("experiment_class")
    experiment = experiment_class(config)
    try:
        experiment.setup_experiment(config)
        experiment.run()
        if hasattr(experiment, "save_state_dict"):
            experiment.save_state_dict()
    finally:
        if hasattr(experiment, "logger_handler"):
            experiment.close()
        elif getattr(experiment, "env", None) is not None:
            experiment.env.close()

    print(f"Done. Output: {config['logging']['output_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
