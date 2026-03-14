"""Learn an object model from a pre-recorded turntable scan video.

Reads a scan directory produced by ``record-scan``, extracts frames, runs
the SAM2 + VGGT vision pipeline, and feeds the resulting observations to
Monty's graph-matching learning module.

Usage (inside monty_comp container)::

    turntable-learn --object cup
    turntable-learn --object cup --scan-dir /data/scans/cup
    turntable-learn --object cup --max-steps 500
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _build_config(
    object_name: str,
    scan_dir: str,
    max_steps: int | None = None,
    output_dir: str | None = None,
    frame_subsample: int | None = None,
    description: str | None = None,
) -> dict:
    """Load the base config and apply CLI overrides."""
    from monty_ext.configs.turntable_train import CONFIGS

    config = copy.deepcopy(CONFIGS["turntable_pretrain_base"])

    config["turntable_config"]["object_name"] = object_name
    config["turntable_config"]["scan_dir"] = scan_dir

    if description:
        config["turntable_config"]["description"] = description

    if frame_subsample is not None:
        config["turntable_config"]["frame_subsample"] = frame_subsample

    if max_steps is not None:
        config["max_train_steps"] = max_steps
        config["max_total_steps"] = max_steps + 200
        config["monty_config"]["monty_args"]["num_exploratory_steps"] = max_steps
        config["monty_config"]["monty_args"]["max_total_steps"] = max_steps + 200

    if output_dir is not None:
        config["logging"]["output_dir"] = output_dir
    else:
        config["logging"]["output_dir"] = f"/results/turntable/{object_name}"

    out_dir = config["logging"]["output_dir"]
    config["turntable_config"]["debug_vis_dir"] = f"{out_dir}/debug_vis"

    config["logging"]["run_name"] = f"turntable_{object_name}"

    return config


def learn(config: dict) -> Path:
    """Run the turntable learning experiment.

    Returns:
        Path to the output directory containing model.pt.
    """
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
    object_name = config["turntable_config"]["object_name"]

    # Print summary
    n_graphs = 0
    n_nodes_total = 0
    for lm in experiment.model.learning_modules:
        if hasattr(lm, "graph_memory") and hasattr(lm.graph_memory, "models_in_memory"):
            n_graphs = len(lm.graph_memory.models_in_memory)
            for obj_id, channels in lm.graph_memory.models_in_memory.items():
                for ch_name, graph in channels.items():
                    if hasattr(graph, "pos"):
                        n_nodes_total += len(graph.pos)
            break

    print(f"\n{'='*60}")
    print(f"Learning complete: {object_name}")
    print(f"  Graph objects stored: {n_graphs}")
    print(f"  Total graph nodes:    {n_nodes_total}")
    print(f"  Training steps:       {experiment.total_train_steps}")
    print(f"  Output:               {output_dir}")
    print(f"{'='*60}")

    # Auto-generate visualizations
    model_epoch_dir = output_dir / "0"
    model_pt = model_epoch_dir / "model.pt"
    if not model_pt.exists():
        model_pt = output_dir / "model.pt"
        model_epoch_dir = output_dir

    if model_pt.exists():
        try:
            from monty_ext.scripts.visualize_model import visualize, visualize_3d_interactive
            print("\nGenerating visualizations...")
            visualize(model_dir=str(model_epoch_dir))
            visualize_3d_interactive(model_dir=str(model_epoch_dir))
        except Exception as e:
            print(f"Visualization generation failed: {e}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Learn an object model from a pre-recorded turntable scan."
    )
    parser.add_argument(
        "--object",
        required=True,
        help="Name of the object to learn (must match a scan directory)",
    )
    parser.add_argument(
        "--scan-dir",
        default=os.environ.get("SCAN_DIR", "/data/scans"),
        help="Root directory containing scan subdirectories",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps (default: from config)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for model and logs",
    )
    parser.add_argument(
        "--frame-subsample",
        type=int,
        default=None,
        help="Frame subsampling factor (default: 5 = every 5th frame)",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help='Text description of the object for SAM2 segmentation '
             '(e.g. "red box on black turntable")',
    )
    args = parser.parse_args()

    object_scan_dir = Path(args.scan_dir) / args.object
    if not object_scan_dir.exists():
        print(f"Error: scan directory not found: {object_scan_dir}", file=sys.stderr)
        print(f"Run 'record-scan --object {args.object}' first.", file=sys.stderr)
        sys.exit(1)

    config = _build_config(
        object_name=args.object,
        scan_dir=args.scan_dir,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        frame_subsample=args.frame_subsample,
        description=args.description,
    )

    learn(config)


if __name__ == "__main__":
    main()
