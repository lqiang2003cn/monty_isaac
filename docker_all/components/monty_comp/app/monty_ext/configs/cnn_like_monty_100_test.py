"""CNN-style multi-patch Monty experiment configuration.

Renders a single 1280x800 ``view_finder`` in Habitat, crops it into 100
non-overlapping 128x80 patches via :class:`CropPatchesFromViewFinder`,
and feeds each to its own CameraSM + DisplacementGraphLM.

Object training uses 72 Y-axis rotations (0-355 deg, 5 deg steps) via
the ``Predefined`` sampler with ``n_train_epochs=72``.  Each epoch
presents every object at a new rotation; each episode takes a single
observation (the initial view).

Usage::

    from monty_ext.configs.cnn_like_monty_100_test import CONFIGS
    config = CONFIGS["cnn_like_monty_100_test"]
"""

from __future__ import annotations

from tbp.monty.frameworks.actions.action_samplers import ConstantSampler
from tbp.monty.frameworks.actions.actions import (
    LookDown,
    LookUp,
    TurnLeft,
    TurnRight,
)
from tbp.monty.frameworks.environment_utils.transforms import (
    DepthTo3DLocations,
    MissingToMaxDepth,
)
from tbp.monty.frameworks.environments.embodied_data import (
    EnvironmentInterfacePerObject,
)
from tbp.monty.frameworks.environments.object_init_samplers import Predefined
from tbp.monty.frameworks.loggers.monty_handlers import BasicCSVStatsHandler
from tbp.monty.frameworks.models.displacement_matching import DisplacementGraphLM
from tbp.monty.frameworks.models.graph_matching import MontyForGraphMatching
from tbp.monty.frameworks.models.motor_policies import NaiveScanPolicy
from tbp.monty.frameworks.models.motor_system import MotorSystem
from tbp.monty.frameworks.models.sensor_modules import CameraSM, Probe

from monty_ext.experiments.cnn_experiment import CNNStylePretrainingExperiment
from monty_ext.transforms.crop_patches import CropPatchesFromViewFinder

# Habitat imports are deferred because habitat_sim requires OpenGL at import
# time.  They are only needed when build_config() is called and the experiment
# is actually run (with a GPU + display available).
HabitatEnvironment = None
MultiSensorAgent = None


def _import_habitat():
    """Lazily import Habitat classes, raising a clear error if unavailable."""
    global HabitatEnvironment, MultiSensorAgent
    if HabitatEnvironment is not None:
        return
    try:
        from tbp.monty.simulators.habitat.environment import (
            HabitatEnvironment as _HE,
        )
        from tbp.monty.simulators.habitat import MultiSensorAgent as _MSA
        HabitatEnvironment = _HE
        MultiSensorAgent = _MSA
    except ImportError as e:
        raise ImportError(
            "Habitat simulator is required for CNN-style Monty experiments. "
            "Ensure habitat_sim is installed and OpenGL libraries are available. "
            f"Original error: {e}"
        ) from e

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENT_ID = "agent_id_0"
N_PATCHES = 100
N_ROWS, N_COLS = 10, 10
PATCH_H, PATCH_W = 80, 128
VF_H, VF_W = N_ROWS * PATCH_H, N_COLS * PATCH_W  # 800, 1280

N_ROTATIONS = 72
TURNTABLE_ROTATIONS = [(0.0, float(y), 0.0) for y in range(0, 360, 360 // N_ROTATIONS)]

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


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_config(
    object_names: list[str] | None = None,
    n_rotations: int = N_ROTATIONS,
    output_dir: str = "/results/cnn_monty",
) -> dict:
    """Build the full experiment config dict.

    Args:
        object_names: Habitat object names to train on.
        n_rotations: Number of evenly-spaced Y-axis rotations.
        output_dir: Directory for model checkpoints and logs.

    Returns:
        Config dict consumable by ``CNNStylePretrainingExperiment``.
    """
    _import_habitat()

    if object_names is None:
        object_names = list(DEFAULT_OBJECTS)

    rotations = [
        (0.0, float(y), 0.0)
        for y in range(0, 360, 360 // n_rotations)
    ]

    # -- sensor modules: 100 CameraSMs + 1 Probe (view_finder) ---------------
    sensor_module_configs = {}
    for i in range(N_PATCHES):
        sensor_module_configs[f"sensor_module_{i}"] = dict(
            sensor_module_class=CameraSM,
            sensor_module_args=dict(
                sensor_module_id=f"patch_{i}",
                features=list(FEATURES),
                save_raw_obs=False,
            ),
        )
    sensor_module_configs[f"sensor_module_{N_PATCHES}"] = dict(
        sensor_module_class=Probe,
        sensor_module_args=dict(
            sensor_module_id="view_finder",
            save_raw_obs=False,
        ),
    )

    # -- learning modules: 100 DisplacementGraphLMs ---------------------------
    learning_module_configs = {}
    for i in range(N_PATCHES):
        learning_module_configs[f"learning_module_{i}"] = dict(
            learning_module_class=DisplacementGraphLM,
            learning_module_args=dict(
                k=5,
                match_attribute="displacement",
            ),
        )

    # -- mapping matrices -----------------------------------------------------
    sm_to_agent_dict = {"view_finder": AGENT_ID}
    sm_to_lm_matrix = []
    for i in range(N_PATCHES):
        sm_to_agent_dict[f"patch_{i}"] = AGENT_ID
        sm_to_lm_matrix.append([i])

    # -- transform pipeline ---------------------------------------------------
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
    ]

    # -- motor policy (never called with max_total_steps=1, but required) -----
    motor_system_config = dict(
        motor_system_class=MotorSystem,
        motor_system_args=dict(
            policy=NaiveScanPolicy(
                fixed_amount=5.0,
                action_sampler=ConstantSampler(
                    actions=[LookUp, LookDown, TurnLeft, TurnRight],
                    rotation_degrees=5.0,
                ),
                agent_id=AGENT_ID,
            ),
        ),
    )

    # -- full config ----------------------------------------------------------
    config = dict(
        experiment_class=CNNStylePretrainingExperiment,

        do_train=True,
        do_eval=False,
        max_train_steps=1,
        max_eval_steps=1,
        max_total_steps=1,
        n_train_epochs=n_rotations,
        n_eval_epochs=0,
        min_lms_match=1,
        model_name_or_path="",
        seed=42,
        show_sensor_output=False,
        supervised_lm_ids="all",

        logging=dict(
            output_dir=output_dir,
            run_name="cnn_like_monty_100",
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
                num_exploratory_steps=1,
                min_train_steps=1,
                min_eval_steps=1,
                max_total_steps=1,
            ),
            sensor_module_configs=sensor_module_configs,
            learning_module_configs=learning_module_configs,
            motor_system_config=motor_system_config,
            sm_to_agent_dict=sm_to_agent_dict,
            sm_to_lm_matrix=sm_to_lm_matrix,
            lm_to_lm_matrix=None,
            lm_to_lm_vote_matrix=None,
        ),

        env_interface_config=dict(
            env_init_func=HabitatEnvironment,
            env_init_args=dict(
                agents=dict(
                    agent_type=MultiSensorAgent,
                    agent_args=dict(
                        agent_id=AGENT_ID,
                        sensor_ids=["view_finder"],
                        resolutions=[(VF_H, VF_W)],
                        zooms=[1.0],
                        positions=[(0.0, 0.0, 0.0)],
                        rotations=[(1.0, 0.0, 0.0, 0.0)],
                        semantics=[False],
                    ),
                ),
            ),
            transform=transform,
        ),

        train_env_interface_class=EnvironmentInterfacePerObject,
        train_env_interface_args=dict(
            object_names=object_names,
            object_init_sampler=Predefined(
                rotations=rotations,
            ),
        ),
    )

    return config


# ---------------------------------------------------------------------------
# CONFIGS registry — lazy to avoid importing habitat_sim at module load.
# Access via CONFIGS["cnn_like_monty_100_test"] which triggers build_config().
# ---------------------------------------------------------------------------


class _LazyConfigs(dict):
    """Build the config on first access so habitat_sim isn't imported at load."""

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
