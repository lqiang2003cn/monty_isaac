"""Visualize learned graph models saved by CNN-style Monty pretraining.

Where learned models are saved
------------------------------
The experiment writes to <output_dir>/pretrained/ (see
tbp.monty.frameworks.experiments.monty_experiment.MontyExperiment.save_state_dict).
For the CNN config, output_dir is RESULTS_ROOT/cnn_monty, so:
  - In container: /results/cnn_monty/pretrained/
  - On host:      ${MONTY_RESULTS:-../../data/monty_results}/cnn_monty/pretrained/

Files in that directory:
  - model.pt          : Full Monty model state (lm_dict, sm_dict, etc.).
  - exp_state_dict.pt : Experiment counters and timestamps.
  - config.pt         : Experiment config.

model.pt structure (see tbp.monty.frameworks.models.monty_base.MontyBase.state_dict):
  state["lm_dict"][lm_id] = LM.state_dict() with key "graph_memory"
  graph_memory = models_in_memory = { object_id: { input_channel: GraphObjectModel } }
  GraphObjectModel has .pos (node positions), .edge_index, .edge_attr (see
  tbp.monty.frameworks.models.object_model.GraphObjectModel and
  docs/how-monty-works/learning-module/object-models.md).

Usage
-----
  # Inside monty_comp container (after a CNN pretraining run has produced model.pt):
  conda run -n tbp.monty visualize-learned-models \\
    --model-dir /results/cnn_monty/pretrained --out-dir /results/cnn_monty/viz

  # Or as module:
  conda run -n tbp.monty python -m monty_ext.scripts.visualize_learned_models \\
    --model-dir /results/cnn_monty/pretrained --out-dir /results/cnn_monty/viz

  # From host (output appears under ${MONTY_RESULTS}/cnn_monty/viz):
  docker compose -f docker_all/compose.yml --profile monty run --rm monty_comp \\
    bash -c "conda run -n tbp.monty visualize-learned-models \\
      --model-dir /results/cnn_monty/pretrained --out-dir /results/cnn_monty/viz"

  # Select specific LMs (default is all LMs) and draw edges:
  ... visualize-learned-models --lm-indices 0 10 50 99 --edges

  # Interactive 3D in browser (no display needed; open the HTML file). All LMs by default:
  ... visualize-learned-models --html /results/cnn_monty/viz/explore.html --edges
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Optional matplotlib; fail with a clear message if not available
try:
    import matplotlib
    # Backend is set in main(): "Agg" when saving to file, interactive when showing
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ImportError as e:
    raise ImportError("visualize_learned_models requires matplotlib") from e


def _to_numpy(x):
    """Convert tensor or array to numpy."""
    if x is None:
        return None
    if hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_pretrained_state(model_dir: Path):
    """Load model.pt from a pretrained directory. Returns state dict."""
    import torch
    model_pt = model_dir / "model.pt"
    if not model_pt.is_file():
        raise FileNotFoundError(f"No model.pt in {model_dir}")
    try:
        return torch.load(model_pt, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(model_pt, map_location="cpu")


def get_graph_models(state):
    """
    Yield (lm_id, object_id, channel, model) for each graph in the saved state.
    state["lm_dict"][lm_id]["graph_memory"] = models_in_memory = { obj_id: { ch: model } }.
    """
    lm_dict = state.get("lm_dict") or {}
    for lm_id, lm_state in lm_dict.items():
        graph_memory = lm_state.get("graph_memory")
        if not isinstance(graph_memory, dict):
            continue
        for object_id, channels in graph_memory.items():
            if not isinstance(channels, dict):
                continue
            for ch, model in channels.items():
                if hasattr(model, "pos") and model.pos is not None:
                    yield lm_id, object_id, ch, model


def plot_one_graph(
    model,
    ax,
    *,
    show_edges: bool = True,
    color_by_z: bool = True,
):
    """Draw one graph (nodes and optionally edges) into a 3D axes."""
    pos = _to_numpy(model.pos)
    if pos is None or pos.size == 0:
        return
    if pos.ndim == 1:
        pos = pos.reshape(1, -1)
    n = pos.shape[0]
    if color_by_z:
        c = pos[:, 2]
    else:
        c = np.arange(n)
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=c, alpha=0.6, s=8)

    if show_edges:
        edge_index = getattr(model, "edge_index", None)
        if edge_index is not None:
            edge_index = _to_numpy(edge_index)
            if edge_index is not None and edge_index.size > 0:
                for e in range(edge_index.shape[1]):
                    i, j = edge_index[0, e], edge_index[1, e]
                    if i < n and j < n:
                        ax.plot(
                            [pos[i, 0], pos[j, 0]],
                            [pos[i, 1], pos[j, 1]],
                            [pos[i, 2], pos[j, 2]],
                            color="gray",
                            alpha=0.4,
                            linewidth=0.5,
                        )
    ax.set_aspect("equal")


def _write_interactive_html(unique, html_path: Path, show_edges: bool):
    """Write an interactive 3D Plotly figure to an HTML file (browser can rotate/zoom).
    When there are multiple LMs, adds a dropdown to show one LM at a time (default) or All.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly is required for --html. Install with: pip install plotly", file=sys.stderr)
        return False

    fig = go.Figure()
    step = 2 if show_edges else 1  # traces per LM: markers + optional edges
    for lm_id, obj_id, model in unique:
        pos = _to_numpy(model.pos)
        if pos is None or pos.size == 0:
            continue
        if pos.ndim == 1:
            pos = pos.reshape(1, -1)
        n = pos.shape[0]
        name = f"LM {lm_id} / {obj_id} (n={n})"
        fig.add_trace(go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
            mode="markers",
            marker=dict(size=4, color=pos[:, 2], colorscale="Viridis", opacity=0.8),
            name=name,
        ))
        if show_edges:
            edge_index = getattr(model, "edge_index", None)
            if edge_index is not None:
                edge_index = _to_numpy(edge_index)
                if edge_index is not None and edge_index.size > 0:
                    edge_x, edge_y, edge_z = [], [], []
                    for e in range(edge_index.shape[1]):
                        i, j = edge_index[0, e], edge_index[1, e]
                        if i < n and j < n:
                            edge_x += [pos[i, 0], pos[j, 0], None]
                            edge_y += [pos[i, 1], pos[j, 1], None]
                            edge_z += [pos[i, 2], pos[j, 2], None]
                    fig.add_trace(go.Scatter3d(
                        x=edge_x, y=edge_y, z=edge_z,
                        mode="lines",
                        line=dict(color="gray", width=2),
                        opacity=0.5,
                        name=f"{name} edges",
                        showlegend=False,
                    ))
    num_traces = len(fig.data)
    num_lms = num_traces // step

    # Dropdown: one LM at a time (default first) or All, so multiple LMs don't overdraw into a cloud
    if num_lms > 1:
        # Default: show only first LM (clean single shape)
        for i in range(num_traces):
            fig.data[i].visible = i < step
        visibility_all = [True] * num_traces
        buttons = [
            dict(
                label="All",
                method="update",
                args=[{"visible": visibility_all}, {"title": "Learned graph models — All LMs"}],
            )
        ]
        for lm_idx in range(num_lms):
            visible = [False] * num_traces
            for t in range(step):
                visible[lm_idx * step + t] = True
            lm_id, obj_id = unique[lm_idx][0], unique[lm_idx][1]
            buttons.append(
                dict(
                    label=f"LM {lm_id}",
                    method="update",
                    args=[{"visible": visible}, {"title": f"Learned graph — LM {lm_id} / {obj_id}"}],
                )
            )
        fig.update_layout(
            updatemenus=[
                dict(
                    active=1,
                    buttons=buttons,
                    direction="down",
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=1.08,
                    yanchor="top",
                )
            ],
            title="Learned graph — LM {} / {}".format(unique[0][0], unique[0][1]),
            scene=dict(aspectmode="data"),
            margin=dict(l=0, r=0, b=0, t=40),
        )
    else:
        fig.update_layout(
            title="Learned graph models (drag to rotate, scroll to zoom)",
            scene=dict(aspectmode="data"),
            margin=dict(l=0, r=0, b=0, t=40),
        )
    html_path = Path(html_path)
    if html_path.suffix.lower() != ".html":
        html_path = html_path / "learned_models_interactive.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(html_path))
    print(f"Interactive 3D plot saved to: {html_path}")
    print("Open this file in a browser to rotate and zoom.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Visualize learned graph models from CNN-style Monty pretraining.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/results/cnn_monty/pretrained"),
        help="Directory containing model.pt (default: /results/cnn_monty/pretrained)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to save figures; if not set, figures are not saved",
    )
    parser.add_argument(
        "--lm-indices",
        type=int,
        nargs="*",
        default=None,
        help="Only plot these LM indices (e.g. 0 10 50 99). Default: plot all LMs",
    )
    parser.add_argument(
        "--object",
        type=str,
        default=None,
        help="Only plot this object id (default: first object found)",
    )
    parser.add_argument(
        "--edges",
        action="store_true",
        help="Draw edges between nodes (k-NN graph)",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Plot one LM per figure instead of a grid",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=None,
        metavar="PATH",
        help="Save interactive 3D plot as HTML (open in browser to rotate/zoom). No display needed.",
    )
    args = parser.parse_args()

    # Use non-interactive backend when saving to file or HTML (no window needed)
    if args.out_dir is not None or args.html is not None:
        matplotlib.use("Agg")

    state = load_pretrained_state(args.model_dir)
    graphs = list(get_graph_models(state))
    if not graphs:
        print("No graph models found in model.pt (lm_dict[][graph_memory]).", file=sys.stderr)
        return 1

    # Filter by object if requested
    if args.object is not None:
        graphs = [(a, b, c, d) for a, b, c, d in graphs if b == args.object]
        if not graphs:
            print(f"No graphs for object '{args.object}'.", file=sys.stderr)
            return 1

    # Unique (lm_id, object_id) and pick first channel per pair
    seen = set()
    unique = []
    for lm_id, obj_id, ch, model in graphs:
        key = (lm_id, obj_id)
        if key not in seen:
            seen.add(key)
            unique.append((lm_id, obj_id, model))

    if args.lm_indices is not None:
        idx_set = set(args.lm_indices)
        unique = [(a, b, m) for a, b, m in unique if a in idx_set]
        if not unique:
            print("No LMs left after --lm-indices filter.", file=sys.stderr)
            return 1

    if args.html is not None:
        if _write_interactive_html(unique, args.html, args.edges):
            return 0
        return 1

    n_plots = len(unique)
    if args.single:
        for i, (lm_id, obj_id, model) in enumerate(unique):
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(1, 1, 1, projection="3d")
            plot_one_graph(model, ax, show_edges=args.edges)
            ax.set_title(f"LM {lm_id} / {obj_id} (n={_to_numpy(model.pos).shape[0]})")
            if args.out_dir:
                args.out_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(args.out_dir / f"lm_{lm_id}_{obj_id}.png", dpi=120, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()
        if args.out_dir and unique:
            print(f"Saved {len(unique)} figures to {args.out_dir}")
        return 0

    # Grid: plot all LMs (4 columns, rows as needed)
    n_cols = 4
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, subplot_kw={"projection": "3d"}, figsize=(4 * n_cols, 4 * n_rows)
    )
    axes = np.atleast_2d(axes)
    for idx, (lm_id, obj_id, model) in enumerate(unique):
        r, c = idx // n_cols, idx % n_cols
        ax = axes[r, c]
        plot_one_graph(model, ax, show_edges=args.edges)
        n_nodes = _to_numpy(model.pos).shape[0]
        ax.set_title(f"LM{lm_id} {obj_id} (n={n_nodes})", fontsize=8)
    for idx in range(len(unique), axes.size):
        r, c = idx // n_cols, idx % n_cols
        axes[r, c].set_visible(False)
    fig.suptitle("Learned graph models (node point clouds)")
    plt.tight_layout()
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.out_dir / "learned_models_grid.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved grid to {out_path}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
