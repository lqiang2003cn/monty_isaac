"""Generate conf/cnn_like_monty_100_test.yaml from build_config().

Expands all repetitive structures (100 sensor modules, 100 learning modules,
sm_to_lm_matrix, sm_to_agent_dict, rotations) so the YAML is complete and
never hand-edited for those sections.

Usage (from repo root or with PYTHONPATH including monty_ext)::

    python -m monty_ext.scripts.generate_cnn_config_yaml
    generate-cnn-config-yaml

Output: docker_all/components/monty_comp/app/monty_ext/conf/cnn_like_monty_100_test.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

# conf/ is next to configs/ under monty_ext (scripts -> parent = monty_ext)
CONF_DIR = Path(__file__).resolve().parent.parent / "conf"
OUTPUT_FILE = CONF_DIR / "cnn_like_monty_100_test.yaml"


def main() -> int:
    try:
        import yaml
    except ImportError:
        print("PyYAML is required: pip install pyyaml", file=sys.stderr)
        return 1

    try:
        from monty_ext.configs.cnn_like_monty_100_test import build_config
        from monty_ext.configs.config_to_yaml import config_to_yaml_safe
        config = build_config()
        data = config_to_yaml_safe(config)
    except ImportError:
        # No tbp/Habitat: use doc-only builder (same structure, string class paths)
        from monty_ext.configs.cnn_like_monty_100_test_doc import build_doc_config
        data = build_doc_config()

    out_path = OUTPUT_FILE
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = """# CNN-style Multi-Patch Monty — Experiment Configuration
#
# AUTO-GENERATED from monty_ext.configs.cnn_like_monty_100_test.build_config().
# Do not edit the repetitive sections (sensor_module_*, learning_module_*,
# sm_to_lm_matrix, sm_to_agent_dict, rotations); re-run generate-cnn-config-yaml.
#
# To run:
#   run-cnn-experiment
#   run-cnn-experiment --objects mug banana --rotations 72

"""

    with open(out_path, "w") as f:
        f.write(header)
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
