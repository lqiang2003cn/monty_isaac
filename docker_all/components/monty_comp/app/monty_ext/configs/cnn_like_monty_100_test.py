"""CNN-style multi-patch Monty experiment configuration.

Single source of truth: this module. All large fields (sensor_module_configs,
learning_module_configs, sm_to_lm_matrix, sm_to_agent_dict, rotations) are
computed dynamically from the constants below.

Renders a single 1280x800 view_finder in Habitat, crops into 100 patches,
trains 100 DisplacementGraphLMs over 72 Y-axis rotations.

Usage (import)::

    from monty_ext.configs.cnn_like_monty_100_test import build_config, CONFIGS
    config = build_config()  # or CONFIGS["cnn_like_monty_100_test"]

Usage (standalone run)::

    python -m monty_ext.configs.cnn_like_monty_100_test
    python -m monty_ext.configs.cnn_like_monty_100_test --objects mug banana --rotations 36
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants (faithful map from YAML; large fields derived below)
# ---------------------------------------------------------------------------

AGENT_ID = "agent_id_0"
N_PATCHES = 100
N_ROWS, N_COLS = 10, 10
PATCH_H, PATCH_W = 80, 80
VF_H, VF_W = N_ROWS * PATCH_H, N_COLS * PATCH_W  # 800, 800

# More rotations = denser Y-axis coverage (object turntable).
N_ROTATIONS = 144
# Full-sphere sweep: elevation × azimuth grid so viewfinder visits entire surface.
# Step size (degrees). Finer = better surface sampling, slower run.
# FullSphereSweepPolicy generates one action per (elevation, azimuth) step; total steps
# depend on step size (e.g. 10° → ~1300 steps, 5° → ~5300 steps).
MOTOR_DEGREES = 10.0
# Keep more 3D points: only merge points within this distance (m). Smaller = denser graphs.
GRAPH_DISTANCE_THRESHOLD = 0.0005
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
DEFAULT_OBJECTS = ["mug", "banana"]

# Learned models are saved under OUTPUT_DIR_DEFAULT/pretrained/ (see save_state_dict in
# tbp.monty.frameworks.experiments.monty_experiment). Files written:
#   - model.pt       : Monty model state (lm_dict with graph_memory per LM)
#   - exp_state_dict.pt, config.pt
# Must be under the compose bind mount so they persist on the host:
# monty_comp compose mounts ${MONTY_RESULTS:-../../data/monty_results}:/results
RESULTS_ROOT = os.environ.get("MONTY_RESULTS_ROOT", "/results")
OUTPUT_DIR_DEFAULT = os.path.join(RESULTS_ROOT, "cnn_monty")
RUN_NAME = "cnn_like_monty_100"

# Habitat object dataset (YCB). Resolved via MONTY_DATA env var or default.
MONTY_DATA = os.environ.get("MONTY_DATA", os.path.expanduser("~/tbp/data"))
HABITAT_DATA_PATH = os.path.join(MONTY_DATA, "habitat", "objects", "ycb")

# Habitat imports deferred (OpenGL at import time)
_HabitatEnvironment = None
_MultiSensorAgent = None


def _import_habitat():
    global _HabitatEnvironment, _MultiSensorAgent
    if _HabitatEnvironment is not None:
        return
    try:
        from tbp.monty.simulators.habitat.environment import (
            HabitatEnvironment as _HE,
        )
        from tbp.monty.simulators.habitat import MultiSensorAgent as _MSA
        _HabitatEnvironment = _HE
        _MultiSensorAgent = _MSA
    except ImportError as e:
        raise ImportError(
            "Habitat simulator required. Run inside monty_comp or install habitat_sim."
            f" Original: {e}"
        ) from e


# ---------------------------------------------------------------------------
# build_config() — full experiment config
# ---------------------------------------------------------------------------

def build_config(
    object_names: list[str] | None = None,
    n_rotations: int = N_ROTATIONS,
    output_dir: str = OUTPUT_DIR_DEFAULT,
) -> dict:
    """Build full experiment config. Large fields computed from constants."""
    _import_habitat()

    if object_names is None:
        object_names = list(DEFAULT_OBJECTS)

    # rotations: n_rotations × [x, y_deg, z] (Y-axis turntable)
    rotations = [
        (0.0, float(y), 0.0)
        for y in range(0, 360, 360 // n_rotations)
    ]

    # sensor_module_configs: sensor_module_0..99 (PatchCameraSM), sensor_module_100 (Probe)
    from tbp.monty.frameworks.models.sensor_modules import Probe
    from monty_ext.sensor_modules.patch_camera_sm import PatchCameraSM

    sensor_module_configs = {}
    for i in range(N_PATCHES):
        sensor_module_configs[f"sensor_module_{i}"] = {
            "sensor_module_class": PatchCameraSM,
            "sensor_module_args": {
                "sensor_module_id": f"patch_{i}",
                "parent_sensor_id": "view_finder",
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

    # learning_module_configs: learning_module_0..99 (DisplacementGraphLM)
    from tbp.monty.frameworks.models.displacement_matching import DisplacementGraphLM

    # Tighter distance threshold keeps more points (less aggressive merge in remove_close_points).
    graph_delta_thresholds = {
        f"patch_{i}": {"distance": GRAPH_DISTANCE_THRESHOLD} for i in range(N_PATCHES)
    }
    learning_module_configs = {
        f"learning_module_{i}": {
            "learning_module_class": DisplacementGraphLM,
            "learning_module_args": {
                "k": 5,
                "match_attribute": "displacement",
                "graph_delta_thresholds": graph_delta_thresholds,
            },
        }
        for i in range(N_PATCHES)
    }

    # sm_to_agent_dict: view_finder + patch_0..patch_99 -> agent_id_0
    sm_to_agent_dict = {"view_finder": AGENT_ID}
    for i in range(N_PATCHES):
        sm_to_agent_dict[f"patch_{i}"] = AGENT_ID

    # sm_to_lm_matrix: row i = [i] (patch_i -> LM i)
    sm_to_lm_matrix = [[i] for i in range(N_PATCHES)]

    # transform pipeline (after env render)
    from tbp.monty.frameworks.environment_utils.transforms import (
        DepthTo3DLocations,
        MissingToMaxDepth,
    )
    from monty_ext.transforms.crop_patches import CropPatchesFromViewFinder
    from monty_ext.transforms.world_to_object_frame import WorldToObjectFrame

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
            n_rows=N_ROWS,
            n_cols=N_COLS,
            patch_h=PATCH_H,
            patch_w=PATCH_W,
        ),
        WorldToObjectFrame(agent_id=AGENT_ID),
    ]

    # motor_system_config: FullSphereSweepPolicy sweeps elevation × azimuth so the
    # viewfinder visits the full sphere (no 90° cap). Step size MOTOR_DEGREES.
    from tbp.monty.frameworks.actions.action_samplers import ConstantSampler
    from tbp.monty.frameworks.actions.actions import (
        LookDown,
        LookUp,
        TurnLeft,
        TurnRight,
    )
    from tbp.monty.frameworks.models.motor_system import MotorSystem
    from monty_ext.motor_policies import FullSphereSweepPolicy

    _scan_policy = FullSphereSweepPolicy(
        agent_id=AGENT_ID,
        action_sampler=ConstantSampler(
            actions=[LookUp, LookDown, TurnLeft, TurnRight],
            rotation_degrees=MOTOR_DEGREES,
        ),
        elevation_step_deg=MOTOR_DEGREES,
        azimuth_step_deg=MOTOR_DEGREES,
    )
    scan_steps_per_episode = _scan_policy.num_actions

    motor_system_config = {
        "motor_system_class": MotorSystem,
        "motor_system_args": {
            "policy": _scan_policy,
        },
    }

    from tbp.monty.frameworks.models.graph_matching import MontyForGraphMatching
    from tbp.monty.frameworks.loggers.monty_handlers import BasicCSVStatsHandler
    from tbp.monty.frameworks.environments.object_init_samplers import Predefined
    from monty_ext.environments.object_frame_interface import EnvironmentInterfaceWithObjectFrame
    from monty_ext.experiments.cnn_experiment import CNNStylePretrainingExperiment

    # Full config — key order matches YAML
    return {
        # Top-level (YAML root)
        "experiment_class": CNNStylePretrainingExperiment,
        "do_train": True,
        "do_eval": False,
        "max_train_steps": scan_steps_per_episode,
        "max_eval_steps": 1,
        "max_total_steps": scan_steps_per_episode,
        "n_train_epochs": n_rotations,
        "n_eval_epochs": 0,
        "min_lms_match": 1,
        "model_name_or_path": "",
        "seed": 42,
        "show_sensor_output": False,
        "supervised_lm_ids": "all",
        "logging": {
            "output_dir": output_dir,
            "run_name": RUN_NAME,
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
                "num_exploratory_steps": 1,
                "min_train_steps": 1,
                "min_eval_steps": 1,
                "max_total_steps": scan_steps_per_episode,
            },
            "sensor_module_configs": sensor_module_configs,
            "learning_module_configs": learning_module_configs,
            "motor_system_config": motor_system_config,
            "sm_to_agent_dict": sm_to_agent_dict,
            "sm_to_lm_matrix": sm_to_lm_matrix,
            "lm_to_lm_matrix": None,
            "lm_to_lm_vote_matrix": None,
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
                        "positions": [(0.0, 0.0, 0.0)],
                        "rotations": [(1.0, 0.0, 0.0, 0.0)],
                        "semantics": [False],
                    },
                },
                "data_path": HABITAT_DATA_PATH,
            },
            "transform": transform,
        },
        "train_env_interface_class": EnvironmentInterfaceWithObjectFrame,
        "train_env_interface_args": {
            "object_names": object_names,
            "object_init_sampler": Predefined(rotations=rotations),
        },
    }


# ---------------------------------------------------------------------------
# CONFIGS registry (lazy; for backward compatibility)
# ---------------------------------------------------------------------------

class _LazyConfigs(dict):
    _built = False

    def __getitem__(self, key):
        if not self._built:
            self._built = True
            super().__setitem__("cnn_like_monty_100_test", build_config())
        return super().__getitem__(key)

    def __contains__(self, key):
        if not self._built:
            self._built = True
            super().__setitem__("cnn_like_monty_100_test", build_config())
        return super().__contains__(key)


CONFIGS = _LazyConfigs()


# ---------------------------------------------------------------------------
# Standalone run (python -m monty_ext.configs.cnn_like_monty_100_test)
# ---------------------------------------------------------------------------

def _run_experiment(config: dict) -> int:
    """Run experiment, save learned model state, and print summary; return exit code."""
    experiment_class = config.pop("experiment_class")
    experiment = experiment_class(config)
    try:
        experiment.setup_experiment(config)
        experiment.run()
        # Persist learned model (graph memory, etc.) to output_dir so it survives the process
        if hasattr(experiment, "save_state_dict"):
            experiment.save_state_dict()
    finally:
        if hasattr(experiment, "logger_handler"):
            experiment.close()
        else:
            env = getattr(experiment, "env", None)
            if env is not None:
                env.close()

    output_dir = Path(config["logging"]["output_dir"])
    n_graphs = 0
    n_nodes_total = 0
    for lm in experiment.model.learning_modules:
        if hasattr(lm, "graph_memory") and hasattr(lm.graph_memory, "models_in_memory"):
            for obj_id, channels in lm.graph_memory.models_in_memory.items():
                n_graphs += 1
                for ch_name, graph in channels.items():
                    if hasattr(graph, "pos"):
                        n_nodes_total += len(graph.pos)
    n_lms = len(experiment.model.learning_modules)
    objects = config.get("train_env_interface_args", {}).get("object_names", [])
    rotations = config.get("n_train_epochs", 72)

    print(f"\n{'=' * 60}")
    print("CNN-style multi-patch Monty pretraining complete")
    print(f"  Objects:           {objects}")
    print(f"  Rotations:         {rotations}")
    print(f"  LMs:               {n_lms}")
    print(f"  Graph objects:     {n_graphs}")
    print(f"  Total graph nodes: {n_nodes_total}")
    print(f"  Output (in container): {output_dir}")
    print("  Learned models:    written under output_dir (bind-mounted, persists on host)")
    print("  On host:           ${MONTY_RESULTS:-../../data/monty_results}/cnn_monty (monty_comp compose)")
    print(f"{'=' * 60}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run CNN-style multi-patch Monty pretraining (standalone).",
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        default=None,
        help=f"Object names (default: {DEFAULT_OBJECTS})",
    )
    parser.add_argument(
        "--rotations",
        type=int,
        default=None,
        help=f"Y-axis rotation count (default: {N_ROTATIONS})",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Output directory (default: {OUTPUT_DIR_DEFAULT})",
    )
    args = parser.parse_args()

    kwargs = {}
    if args.objects is not None:
        kwargs["object_names"] = args.objects
    if args.rotations is not None:
        kwargs["n_rotations"] = args.rotations
    if args.output_dir is not None:
        kwargs["output_dir"] = args.output_dir

    config = build_config(**kwargs)
    return _run_experiment(config)


if __name__ == "__main__":
    sys.exit(main())
