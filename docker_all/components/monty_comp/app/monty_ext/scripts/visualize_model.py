"""Visualize a learned Monty graph object model from a turntable training run.

Loads model.pt, extracts the GraphObjectModel, and produces multiple views:
  1. 3D point cloud with edges (colored by height)
  2. 3D point cloud colored by HSV hue
  3. Surface normals as quiver arrows
  4. Feature distributions (curvatures, HSV)

Usage::

    # From host (with tbp.monty conda env)
    conda run -n tbp.monty visualize-model --model-dir data/monty_results/turntable/yellow_cube/0

    # Inside monty_comp container
    conda run -n tbp.monty visualize-model --model-dir /results/turntable/yellow_cube/0
"""

from __future__ import annotations

import argparse
import colorsys
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D


def load_graph(model_dir: str):
    """Load the first GraphObjectModel from model.pt.

    Returns:
        (object_name, channel_name, graph) tuple.
    """
    model_path = Path(model_dir) / "model.pt"
    if not model_path.exists():
        print(f"Error: {model_path} not found", file=sys.stderr)
        sys.exit(1)

    state_dict = torch.load(str(model_path), map_location="cpu")
    lm_dict = state_dict["lm_dict"]

    for lm_id in sorted(lm_dict.keys()):
        gm = lm_dict[lm_id].get("graph_memory", {})
        for obj_id, channels in gm.items():
            for channel, graph in channels.items():
                return obj_id, channel, graph

    print("Error: no graph found in model.pt", file=sys.stderr)
    sys.exit(1)


def get_feature_values(graph, feature_name: str) -> np.ndarray | None:
    """Extract feature values from graph.x using feature_mapping."""
    if not hasattr(graph, "feature_mapping") or graph.feature_mapping is None:
        return None
    if feature_name not in graph.feature_mapping:
        return None
    start, end = graph.feature_mapping[feature_name]
    return np.array(graph.x[:, start:end])


def hsv_to_rgb_array(hsv_values: np.ndarray) -> np.ndarray:
    """Convert Monty HSV values to RGB for matplotlib.

    Monty stores H in [0,360), S in [0,100], V in [0,100].
    """
    rgb = np.zeros((len(hsv_values), 3))
    for i, (h, s, v) in enumerate(hsv_values):
        rgb[i] = colorsys.hsv_to_rgb(h / 360.0, s / 100.0, v / 100.0)
    return rgb


def plot_3d_graph(graph, obj_name: str, ax: Axes3D, color_by="height"):
    """Plot nodes and edges in 3D."""
    pos = np.array(graph.pos)

    if color_by == "height":
        colors = pos[:, 2]
        cmap = "viridis"
        scatter = ax.scatter(
            pos[:, 0], pos[:, 1], pos[:, 2],
            c=colors, cmap=cmap, s=80, edgecolors="k", linewidth=0.5,
            depthshade=True,
        )
    elif color_by == "hsv":
        hsv = get_feature_values(graph, "hsv")
        if hsv is not None:
            colors = hsv_to_rgb_array(hsv)
        else:
            colors = "tab:blue"
        scatter = ax.scatter(
            pos[:, 0], pos[:, 1], pos[:, 2],
            c=colors, s=80, edgecolors="k", linewidth=0.5,
            depthshade=True,
        )
    else:
        scatter = ax.scatter(
            pos[:, 0], pos[:, 1], pos[:, 2],
            c="tab:blue", s=80, edgecolors="k", linewidth=0.5,
        )

    if hasattr(graph, "edge_index") and graph.edge_index is not None:
        ei = np.array(graph.edge_index)
        for i in range(ei.shape[1]):
            e1, e2 = ei[0, i], ei[1, i]
            ax.plot(
                [pos[e1, 0], pos[e2, 0]],
                [pos[e1, 1], pos[e2, 1]],
                [pos[e1, 2], pos[e2, 2]],
                color="gray", alpha=0.3, linewidth=0.5,
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_aspect("equal")
    return scatter


def plot_normals(graph, ax: Axes3D, scale: float = 0.02):
    """Plot surface normal vectors as arrows."""
    pos = np.array(graph.pos)
    norm = np.array(graph.norm) if graph.norm is not None else None

    ax.scatter(
        pos[:, 0], pos[:, 1], pos[:, 2],
        c="tab:blue", s=60, edgecolors="k", linewidth=0.5,
    )

    if norm is not None:
        ax.quiver(
            pos[:, 0], pos[:, 1], pos[:, 2],
            norm[:, 0] * scale, norm[:, 1] * scale, norm[:, 2] * scale,
            color="red", arrow_length_ratio=0.3, linewidth=1.5,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_aspect("equal")


def plot_feature_distributions(graph, axes):
    """Plot histograms of key features."""
    hsv = get_feature_values(graph, "hsv")
    curv = get_feature_values(graph, "principal_curvatures_log")

    ax_idx = 0
    if hsv is not None and len(axes) > ax_idx:
        ax = axes[ax_idx]
        labels = ["Hue", "Sat", "Val"]
        for j in range(min(3, hsv.shape[1])):
            ax.hist(hsv[:, j], bins=max(3, len(hsv) // 2), alpha=0.7, label=labels[j])
        ax.set_title("HSV Distribution")
        ax.legend(fontsize=8)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax_idx += 1

    if curv is not None and len(axes) > ax_idx:
        ax = axes[ax_idx]
        labels = ["k1 (log)", "k2 (log)"]
        for j in range(min(2, curv.shape[1])):
            ax.hist(curv[:, j], bins=max(3, len(curv) // 2), alpha=0.7, label=labels[j])
        ax.set_title("Principal Curvatures (log)")
        ax.legend(fontsize=8)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")


def visualize(model_dir: str, save_path: str | None = None):
    """Generate the full visualization figure."""
    obj_name, channel, graph = load_graph(model_dir)

    pos = np.array(graph.pos)
    n_nodes = pos.shape[0]
    n_edges = graph.edge_index.shape[1] if graph.edge_index is not None else 0
    features = list(graph.feature_mapping.keys()) if graph.feature_mapping else []

    print(f"Object: {obj_name}")
    print(f"Channel: {channel}")
    print(f"Nodes: {n_nodes}")
    print(f"Edges: {n_edges}")
    print(f"Features: {features}")
    print(f"Position range: {pos.min(axis=0)} to {pos.max(axis=0)}")

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"Learned Model: {obj_name}\n"
        f"{n_nodes} nodes, {n_edges} edges, features: {', '.join(features)}",
        fontsize=13,
        y=0.98,
    )

    # Panel 1: 3D graph colored by height
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    sc = plot_3d_graph(graph, obj_name, ax1, color_by="height")
    ax1.set_title(f"Graph (colored by Z)")
    fig.colorbar(sc, ax=ax1, shrink=0.5, label="Z position")

    # Panel 2: 3D graph colored by HSV hue
    ax2 = fig.add_subplot(2, 3, 2, projection="3d")
    plot_3d_graph(graph, obj_name, ax2, color_by="hsv")
    ax2.set_title("Graph (colored by object HSV)")

    # Panel 3: surface normals
    ax3 = fig.add_subplot(2, 3, 3, projection="3d")
    plot_normals(graph, ax3)
    ax3.set_title("Surface Normals")

    # Panel 4: top-down view (XY plane)
    ax4 = fig.add_subplot(2, 3, 4)
    hsv = get_feature_values(graph, "hsv")
    if hsv is not None:
        colors = hsv_to_rgb_array(hsv)
    else:
        colors = "tab:blue"
    ax4.scatter(pos[:, 0], pos[:, 1], c=colors, s=80, edgecolors="k", linewidth=0.5)
    if hasattr(graph, "edge_index") and graph.edge_index is not None:
        ei = np.array(graph.edge_index)
        for i in range(ei.shape[1]):
            e1, e2 = ei[0, i], ei[1, i]
            ax4.plot(
                [pos[e1, 0], pos[e2, 0]],
                [pos[e1, 1], pos[e2, 1]],
                color="gray", alpha=0.3, linewidth=0.5,
            )
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    ax4.set_title("Top-down View (XY)")
    ax4.set_aspect("equal")
    ax4.grid(True, alpha=0.3)

    # Panels 5-6: feature histograms
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)
    plot_feature_distributions(graph, [ax5, ax6])

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        out = Path(save_path)
    else:
        out = Path(model_dir) / f"{obj_name}_graph.png"

    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"\nSaved visualization to {out}")

    try:
        plt.show()
    except Exception:
        pass

    return str(out)


def visualize_3d_interactive(model_dir: str, save_path: str | None = None):
    """Generate an interactive 3D HTML visualization using plotly.

    This creates an explorable 3D point cloud of the learned graph that
    can be rotated, zoomed, and inspected in a web browser.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly not installed. Install with: pip install plotly", file=sys.stderr)
        sys.exit(1)

    obj_name, channel, graph = load_graph(model_dir)
    pos = np.array(graph.pos)
    n_nodes = pos.shape[0]
    n_edges = graph.edge_index.shape[1] if graph.edge_index is not None else 0
    features = list(graph.feature_mapping.keys()) if graph.feature_mapping else []

    print(f"Object: {obj_name}")
    print(f"Nodes: {n_nodes}, Edges: {n_edges}")
    print(f"Features: {features}")
    print(f"Position range: {pos.min(axis=0)} to {pos.max(axis=0)}")

    hsv = get_feature_values(graph, "hsv")
    if hsv is not None:
        colors = hsv_to_rgb_array(hsv)
        color_strs = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
                       for r, g, b in colors]
    else:
        color_strs = "blue"

    norm = np.array(graph.norm) if graph.norm is not None else None
    hover_text = []
    for i in range(n_nodes):
        parts = [f"Node {i}", f"pos=({pos[i,0]:.4f}, {pos[i,1]:.4f}, {pos[i,2]:.4f})"]
        if norm is not None:
            parts.append(f"normal=({norm[i,0]:.3f}, {norm[i,1]:.3f}, {norm[i,2]:.3f})")
        if hsv is not None:
            parts.append(f"HSV=({hsv[i,0]:.0f}, {hsv[i,1]:.0f}, {hsv[i,2]:.0f})")
        curv = get_feature_values(graph, "principal_curvatures_log")
        if curv is not None:
            parts.append(f"curv=({curv[i,0]:.2f}, {curv[i,1]:.2f})")
        hover_text.append("<br>".join(parts))

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        mode="markers",
        marker=dict(
            size=4,
            color=color_strs if isinstance(color_strs, list) else color_strs,
            line=dict(width=0.5, color="black"),
        ),
        text=hover_text,
        hoverinfo="text",
        name=f"{obj_name} ({n_nodes} nodes)",
    ))

    if hasattr(graph, "edge_index") and graph.edge_index is not None:
        ei = np.array(graph.edge_index)
        edge_x, edge_y, edge_z = [], [], []
        for i in range(ei.shape[1]):
            e1, e2 = ei[0, i], ei[1, i]
            edge_x += [pos[e1, 0], pos[e2, 0], None]
            edge_y += [pos[e1, 1], pos[e2, 1], None]
            edge_z += [pos[e1, 2], pos[e2, 2], None]
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode="lines",
            line=dict(color="gray", width=1),
            opacity=0.3,
            name="edges",
            hoverinfo="skip",
        ))

    if norm is not None:
        scale = (pos.max() - pos.min()) * 0.05
        cone_x = pos[:, 0]
        cone_y = pos[:, 1]
        cone_z = pos[:, 2]
        fig.add_trace(go.Cone(
            x=cone_x, y=cone_y, z=cone_z,
            u=norm[:, 0], v=norm[:, 1], w=norm[:, 2],
            sizemode="absolute",
            sizeref=scale,
            colorscale="Blues",
            showscale=False,
            opacity=0.4,
            name="normals",
            visible="legendonly",
        ))

    fig.update_layout(
        title=f"Learned Model: {obj_name} — {n_nodes} nodes, {n_edges} edges",
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="data",
        ),
        width=1200,
        height=800,
        legend=dict(x=0, y=1),
    )

    if save_path:
        out = Path(save_path)
    else:
        out = Path(model_dir) / f"{obj_name}_graph_3d.html"

    fig.write_html(str(out))
    print(f"\nInteractive 3D visualization saved to {out}")
    print("Open in browser to explore.")

    return str(out)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a learned Monty graph object model."
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory containing model.pt (e.g. /results/turntable/yellow_cube/0)",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Output PNG path (default: <model-dir>/<object>_graph.png)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Generate interactive 3D HTML visualization (plotly)",
    )
    args = parser.parse_args()

    visualize(model_dir=args.model_dir, save_path=args.save)
    if args.interactive:
        visualize_3d_interactive(model_dir=args.model_dir)


if __name__ == "__main__":
    main()
