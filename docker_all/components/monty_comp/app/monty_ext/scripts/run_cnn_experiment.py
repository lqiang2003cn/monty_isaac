"""Run the CNN-style multi-patch Monty pretraining experiment.

Renders a single 1280x800 view_finder in Habitat, crops into 100 patches,
and trains 100 DisplacementGraphLMs via supervised pretraining over 72
Y-axis turntable rotations.

Usage (inside monty_comp container)::

    run-cnn-experiment
    run-cnn-experiment --objects mug banana
    run-cnn-experiment --objects mug --rotations 36 --output-dir /results/quick
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run CNN-style multi-patch Monty pretraining.",
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        default=None,
        help="Habitat object names to train on (default: mug banana)",
    )
    parser.add_argument(
        "--rotations",
        type=int,
        default=None,
        help="Number of Y-axis rotations (default: 72)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for model and logs (default: /results/cnn_monty)",
    )
    args = parser.parse_args()

    from monty_ext.configs.cnn_like_monty_100_test import build_config

    kwargs = {}
    if args.objects is not None:
        kwargs["object_names"] = args.objects
    if args.rotations is not None:
        kwargs["n_rotations"] = args.rotations
    if args.output_dir is not None:
        kwargs["output_dir"] = args.output_dir

    config = build_config(**kwargs)

    experiment_class = config.pop("experiment_class")
    experiment = experiment_class(config)

    try:
        experiment.setup_experiment(config)
        experiment.run()
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
    print(f"  Output:            {output_dir}")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
