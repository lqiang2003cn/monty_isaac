"""Doc-only config builder for cnn_like_monty_100_test (no tbp/Habitat imports).

Used by generate_cnn_config_yaml to emit conf/cnn_like_monty_100_test.yaml
with all 100 sensor/learning modules, sm_to_lm_matrix, sm_to_agent_dict, and
rotations expanded. Keep constants in sync with cnn_like_monty_100_test.py.
"""

from __future__ import annotations

# Mirrored from cnn_like_monty_100_test.py (no tbp import)
AGENT_ID = "agent_id_0"
N_PATCHES = 100
N_ROWS, N_COLS = 10, 10
PATCH_H, PATCH_W = 80, 128
VF_H, VF_W = N_ROWS * PATCH_H, N_COLS * PATCH_W  # 800, 1280
N_ROTATIONS = 72
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

_CAMERA_SM = "tbp.monty.frameworks.models.sensor_modules.CameraSM"
_PROBE = "tbp.monty.frameworks.models.sensor_modules.Probe"
_DISPLACEMENT_LM = "tbp.monty.frameworks.models.displacement_matching.DisplacementGraphLM"
_MONTY = "tbp.monty.frameworks.models.graph_matching.MontyForGraphMatching"
_MOTOR_SYSTEM = "tbp.monty.frameworks.models.motor_system.MotorSystem"
_NAIVE_SCAN = "tbp.monty.frameworks.models.motor_policies.NaiveScanPolicy"
_CONSTANT_SAMPLER = "tbp.monty.frameworks.actions.action_samplers.ConstantSampler"
_MISSING_MAX_DEPTH = "tbp.monty.frameworks.environment_utils.transforms.MissingToMaxDepth"
_DEPTH_TO_3D = "tbp.monty.frameworks.environment_utils.transforms.DepthTo3DLocations"
_CROP_PATCHES = "monty_ext.transforms.crop_patches.CropPatchesFromViewFinder"
_HABITAT_ENV = "tbp.monty.simulators.habitat.environment.HabitatEnvironment"
_MULTI_SENSOR_AGENT = "tbp.monty.simulators.habitat.MultiSensorAgent"
_ENV_INTERFACE = "tbp.monty.frameworks.environments.embodied_data.EnvironmentInterfacePerObject"
_PREDEFINED = "tbp.monty.frameworks.environments.object_init_samplers.Predefined"
_BASIC_CSV = "tbp.monty.frameworks.loggers.monty_handlers.BasicCSVStatsHandler"
_CNN_EXPERIMENT = "monty_ext.experiments.cnn_experiment.CNNStylePretrainingExperiment"


def build_doc_config(
    object_names: list[str] | None = None,
    n_rotations: int = N_ROTATIONS,
    output_dir: str = "/results/cnn_monty",
) -> dict:
    """Build a YAML-serializable config with the same structure as build_config()."""
    if object_names is None:
        object_names = list(DEFAULT_OBJECTS)
    rotations = [
        [0.0, float(y), 0.0]
        for y in range(0, 360, 360 // n_rotations)
    ]
    sensor_module_configs = {}
    for i in range(N_PATCHES):
        sensor_module_configs[f"sensor_module_{i}"] = dict(
            sensor_module_class=_CAMERA_SM,
            sensor_module_args=dict(
                sensor_module_id=f"patch_{i}",
                features=list(FEATURES),
                save_raw_obs=False,
            ),
        )
    sensor_module_configs[f"sensor_module_{N_PATCHES}"] = dict(
        sensor_module_class=_PROBE,
        sensor_module_args=dict(
            sensor_module_id="view_finder",
            save_raw_obs=False,
        ),
    )
    lm_template = dict(
        learning_module_class=_DISPLACEMENT_LM,
        learning_module_args=dict(k=5, match_attribute="displacement"),
    )
    learning_module_configs = {
        f"learning_module_{i}": lm_template for i in range(N_PATCHES)
    }
    sm_to_agent_dict = {"view_finder": AGENT_ID}
    sm_to_lm_matrix = [[i] for i in range(N_PATCHES)]
    for i in range(N_PATCHES):
        sm_to_agent_dict[f"patch_{i}"] = AGENT_ID
    transform = [
        dict(**{"class": _MISSING_MAX_DEPTH}, agent_id=AGENT_ID, max_depth=1.0),
        dict(
            **{"class": _DEPTH_TO_3D},
            agent_id=AGENT_ID,
            sensor_ids=["view_finder"],
            resolutions=[[VF_H, VF_W]],
            zooms=[1.0],
            world_coord=True,
            get_all_points=True,
        ),
        dict(
            **{"class": _CROP_PATCHES},
            agent_id=AGENT_ID,
            n_rows=N_ROWS,
            n_cols=N_COLS,
            patch_h=PATCH_H,
            patch_w=PATCH_W,
        ),
    ]
    motor_system_config = dict(
        motor_system_class=_MOTOR_SYSTEM,
        motor_system_args=dict(
            policy=dict(
                **{"class": _NAIVE_SCAN},
                fixed_amount=5.0,
                agent_id=AGENT_ID,
                action_sampler=dict(
                    **{"class": _CONSTANT_SAMPLER},
                    actions=[
                        "tbp.monty.frameworks.actions.actions.LookUp",
                        "tbp.monty.frameworks.actions.actions.LookDown",
                        "tbp.monty.frameworks.actions.actions.TurnLeft",
                        "tbp.monty.frameworks.actions.actions.TurnRight",
                    ],
                    rotation_degrees=5.0,
                ),
            ),
        ),
    )
    return dict(
        experiment_class=_CNN_EXPERIMENT,
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
            monty_handlers=[_BASIC_CSV],
            wandb_handlers=[],
        ),
        monty_config=dict(
            monty_class=_MONTY,
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
            env_init_func=_HABITAT_ENV,
            env_init_args=dict(
                agents=dict(
                    agent_type=_MULTI_SENSOR_AGENT,
                    agent_args=dict(
                        agent_id=AGENT_ID,
                        sensor_ids=["view_finder"],
                        resolutions=[[VF_H, VF_W]],
                        zooms=[1.0],
                        positions=[[0.0, 0.0, 0.0]],
                        rotations=[[1.0, 0.0, 0.0, 0.0]],
                        semantics=[False],
                    ),
                ),
            ),
            transform=transform,
        ),
        train_env_interface_class=_ENV_INTERFACE,
        train_env_interface_args=dict(
            object_names=object_names,
            object_init_sampler=dict(**{"class": _PREDEFINED}, rotations=rotations),
        ),
    )
