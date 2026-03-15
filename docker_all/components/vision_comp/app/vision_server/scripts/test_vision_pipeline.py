"""Standalone test for the SAM2 + Grounding DINO + depth vision pipeline.

Runs each vision component on a pre-recorded turntable scan and produces
comprehensive debug output WITHOUT involving the Monty learning module.
Use this to verify the vision pipeline before running ``turntable-learn``.

Depth can come from either pre-recorded RealSense D455 frames (default)
or VGGT neural-net inference, selected with ``--depth-source``.

This script runs inside the vision_comp container where the models are
available locally — no TCP bridge needed.

Usage (inside vision_comp container)::

    test-vision-pipeline --object red_box
    test-vision-pipeline --object red_box --depth-source vggt
    test-vision-pipeline --object red_box --description "red box on turntable"
    test-vision-pipeline --object red_box --depth-source vggt --run-vggt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SCAN_DIR = os.environ.get("SCAN_DIR", "/data/scans")
DEFAULT_OUTPUT_DIR = os.environ.get("VISION_TEST_DIR", "/results/vision_test")


# ----------------------------------------------------------------------
# Frame loading
# ----------------------------------------------------------------------

def _extract_frames(video_path: Path, frames_dir: Path) -> int:
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(frames_dir / f"{idx:06d}.jpg"), frame)
        idx += 1
    cap.release()
    return idx


def load_frames(
    scan_dir: Path,
    subsample: int = 5,
    max_frames: Optional[int] = None,
) -> tuple:
    """Load frames from a scan directory.

    Returns:
        (frames_rgb, frames_dir, frame_indices, all_frame_count)
    """
    frames_dir = scan_dir / "frames"
    video_path = scan_dir / "video.mp4"

    if not frames_dir.exists() or len(list(frames_dir.glob("*.jpg"))) == 0:
        if not video_path.exists():
            raise FileNotFoundError(
                f"Neither frames/ dir nor video.mp4 found in {scan_dir}"
            )
        n = _extract_frames(video_path, frames_dir)
        print(f"  Extracted {n} frames from {video_path}")

    all_jpgs = sorted(frames_dir.glob("*.jpg"))
    total = len(all_jpgs)
    indices = list(range(0, total, subsample))

    if max_frames is not None and len(indices) > max_frames:
        indices = indices[:max_frames]

    selected = [all_jpgs[i] for i in indices]
    frames = []
    for p in selected:
        img = cv2.imread(str(p))
        if img is not None:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return frames, str(frames_dir), indices, total


def load_depth_frames(
    scan_dir: Path,
    frame_indices: List[int],
) -> Optional[List[np.ndarray]]:
    """Load pre-recorded depth frames as float32 metres.

    Prefers ``depth_float/`` (npy, already in metres) over ``depth/``
    (uint16 PNG, needs scale conversion).

    Args:
        scan_dir: Root directory of the scan.
        frame_indices: Which frame indices to load (must match the RGB
            subsampling used by :func:`load_frames`).

    Returns:
        List of (H, W) float32 arrays in metres, or None if no depth
        data is available.
    """
    depth_float_dir = scan_dir / "depth_float"
    depth_dir = scan_dir / "depth"

    if depth_float_dir.exists():
        depths = []
        for idx in frame_indices:
            p = depth_float_dir / f"{idx:06d}.npy"
            if not p.exists():
                logger.warning("Missing depth_float frame %s", p)
                break
            arr = np.load(str(p))
            depths.append(arr.astype(np.float32))
        if len(depths) == len(frame_indices):
            return depths
        logger.info("depth_float incomplete, falling back to depth/ PNGs")

    if not depth_dir.exists():
        return None

    meta_path = scan_dir / "metadata.json"
    depth_scale_to_m = 0.001  # default: uint16 mm -> metres
    if meta_path.exists():
        with open(str(meta_path)) as f:
            meta = json.load(f)
        hw_scale = meta.get("depth_scale", 1.0)
        if hw_scale >= 0.1:
            depth_scale_to_m = 0.001
        else:
            depth_scale_to_m = hw_scale

    depths = []
    for idx in frame_indices:
        p = depth_dir / f"{idx:06d}.png"
        if not p.exists():
            logger.warning("Missing depth frame %s", p)
            return None
        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if raw is None:
            logger.warning("Failed to read depth frame %s", p)
            return None
        depths.append(raw.astype(np.float32) * depth_scale_to_m)

    return depths


# ----------------------------------------------------------------------
# GDINO test
# ----------------------------------------------------------------------

def test_gdino(
    frames: List[np.ndarray],
    text_prompt: str,
    output_dir: Path,
) -> tuple:
    """Run Grounding DINO on every frame and save debug images.

    Returns:
        (summary_dict, detections_list) where each detection entry has
        keys ``frame``, ``box`` (or None), and ``confidence``.
    """
    from vision_server import local_api as vision

    vision.load_sam2()

    out = output_dir / "gdino"
    out.mkdir(parents=True, exist_ok=True)

    detections = []
    detected_count = 0
    confidences = []

    for i, rgb in enumerate(frames):
        result = vision.detect_gdino(rgb, text_prompt)
        box = result.get("box")
        conf = result.get("confidence", 0.0)

        entry = {"frame": i, "box": box, "confidence": conf}
        detections.append(entry)

        if box is not None:
            detected_count += 1
            confidences.append(conf)

        overlay = rgb.copy()
        if box is not None:
            x0, y0, x1, y1 = [int(v) for v in box]
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)
            label = f"conf: {conf:.2f}"
            cv2.putText(overlay, label, (x0, max(y0 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(overlay, "no detection", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(overlay, f"frame {i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(
            str(out / f"frame_{i:04d}.jpg"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )

    with open(str(out / "detections.json"), "w") as f:
        json.dump(detections, f, indent=2)

    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    missed = [d["frame"] for d in detections if d["box"] is None]

    summary = {
        "total_frames": len(frames),
        "detected": detected_count,
        "detection_rate": detected_count / max(len(frames), 1),
        "avg_confidence": round(avg_conf, 3),
        "missed_frames": missed,
        "pass": detected_count / max(len(frames), 1) >= 0.8,
    }
    return summary, detections


# ----------------------------------------------------------------------
# SAM2 video tracking test
# ----------------------------------------------------------------------

def test_sam2_tracking(
    frames: List[np.ndarray],
    frames_dir: str,
    frame_indices: List[int],
    text_prompt: str,
    output_dir: Path,
    reanchor_interval: int = 30,
) -> dict:
    """Run SAM2 VideoPredictor and save mask overlays.

    Returns summary dict with mask consistency stats.
    """
    from vision_server import local_api as vision

    out = output_dir / "sam2_tracked"
    out.mkdir(parents=True, exist_ok=True)

    masks = vision.segment_video(
        frames_dir=frames_dir,
        frame_indices=frame_indices,
        text_prompt=text_prompt,
        debug_vis_dir=str(output_dir),
        reanchor_interval=reanchor_interval,
    )

    areas = []
    for i, (rgb, mask) in enumerate(zip(frames, masks)):
        pct = mask.sum() / mask.size * 100
        areas.append(pct)

        overlay = rgb.copy()
        overlay[mask] = (
            overlay[mask].astype(np.float32) * 0.5
            + np.array([0, 255, 0], dtype=np.float32) * 0.5
        ).astype(np.uint8)
        overlay[~mask] = (overlay[~mask].astype(np.float32) * 0.3).astype(np.uint8)

        cv2.putText(
            overlay, f"frame {i}  mask {pct:.1f}%",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )
        cv2.imwrite(
            str(out / f"frame_{i:04d}.jpg"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )

    # Save masks for programmatic access
    np.savez_compressed(str(out / "masks.npz"), masks=np.stack(masks))

    areas_arr = np.array(areas)
    median_area = float(np.median(areas_arr))
    drift_threshold = 2.0
    drift_flags = [
        i for i, a in enumerate(areas)
        if median_area > 0 and (a > median_area * drift_threshold
                                or a < median_area / drift_threshold)
    ]

    # Consecutive-frame IoU
    ious = []
    for i in range(1, len(masks)):
        inter = np.logical_and(masks[i], masks[i - 1]).sum()
        union = np.logical_or(masks[i], masks[i - 1]).sum()
        ious.append(inter / union if union > 0 else 0.0)
    avg_iou = float(np.mean(ious)) if ious else 0.0

    summary = {
        "total_frames": len(frames),
        "median_area_pct": round(median_area, 2),
        "min_area_pct": round(float(areas_arr.min()), 2) if len(areas_arr) else 0,
        "max_area_pct": round(float(areas_arr.max()), 2) if len(areas_arr) else 0,
        "drift_flags": drift_flags,
        "avg_iou": round(avg_iou, 3),
        "drift_ratio": round(len(drift_flags) / max(len(frames), 1), 3),
        "pass": len(drift_flags) / max(len(frames), 1) <= 0.05 and avg_iou >= 0.5,
    }
    return summary


# ----------------------------------------------------------------------
# RealSense depth test
# ----------------------------------------------------------------------

def test_depth_realsense(
    frames: List[np.ndarray],
    depths: List[np.ndarray],
    masks: List[np.ndarray],
    output_dir: Path,
) -> dict:
    """Validate pre-recorded RealSense depth and save visualisations.

    Args:
        frames: RGB frames (used for overlay context in debug images).
        depths: Per-frame (H, W) float32 depth maps in metres.
        masks: Per-frame (H, W) bool SAM2 masks.
        output_dir: Root output directory; writes into ``realsense_depth/``.

    Returns:
        Summary dict with depth quality stats.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as _cm

    out = output_dir / "realsense_depth"
    out.mkdir(parents=True, exist_ok=True)

    on_object_mins = []
    on_object_maxs = []
    on_object_means = []
    valid_ratios = []

    for i, (depth, mask) in enumerate(zip(depths, masks)):
        h_d, w_d = depth.shape[:2]
        h_m, w_m = mask.shape[:2]
        if (h_d, w_d) != (h_m, w_m):
            mask_resized = cv2.resize(
                mask.astype(np.uint8), (w_d, h_d),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            mask_resized = mask

        on_obj = depth[mask_resized]
        valid = on_obj[on_obj > 0]
        valid_ratio = len(valid) / max(len(on_obj), 1)
        valid_ratios.append(valid_ratio)

        if len(valid) > 0:
            on_object_mins.append(float(valid.min()))
            on_object_maxs.append(float(valid.max()))
            on_object_means.append(float(valid.mean()))
        else:
            on_object_mins.append(0.0)
            on_object_maxs.append(0.0)
            on_object_means.append(0.0)

        d = depth.squeeze()
        d_valid = d[d > 0]
        if len(d_valid) > 0:
            vmin, vmax = np.percentile(d_valid, [2, 98])
        else:
            vmin, vmax = 0, 1
        d_norm = np.clip((d - vmin) / max(vmax - vmin, 1e-6), 0, 1)
        d_color = (_cm.viridis(d_norm)[:, :, :3] * 255).astype(np.uint8)
        obj_min = on_object_mins[-1]
        obj_max = on_object_maxs[-1]
        cv2.putText(
            d_color,
            f"frame {i}  depth [{obj_min:.3f}, {obj_max:.3f}]m",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
        cv2.imwrite(
            str(out / f"depth_{i:04d}.jpg"),
            cv2.cvtColor(d_color, cv2.COLOR_RGB2BGR),
        )

    np.savez(
        str(out / "depth_data.npz"),
        depths=np.stack(depths),
    )

    obj_depth_min = min(on_object_mins) if on_object_mins else 0.0
    obj_depth_max = max(on_object_maxs) if on_object_maxs else 0.0
    obj_depth_mean = float(np.mean(on_object_means)) if on_object_means else 0.0
    avg_valid_ratio = float(np.mean(valid_ratios)) if valid_ratios else 0.0

    depth_ok = 0.01 < obj_depth_min and obj_depth_max < 10.0
    coverage_ok = avg_valid_ratio >= 0.8

    summary = {
        "total_frames": len(depths),
        "depth_range": [round(obj_depth_min, 4), round(obj_depth_max, 4)],
        "mean_depth": round(obj_depth_mean, 4),
        "avg_valid_ratio": round(avg_valid_ratio, 3),
        "depth_ok": depth_ok,
        "coverage_ok": coverage_ok,
        "pass": depth_ok and coverage_ok,
    }
    return summary


# ----------------------------------------------------------------------
# Combined annotated video
# ----------------------------------------------------------------------

def generate_combined_video(
    frames: List[np.ndarray],
    detections: List[dict],
    masks: List[np.ndarray],
    depths: Optional[List[np.ndarray]],
    output_dir: Path,
    fps: float = 5.0,
) -> Path:
    """Render fully annotated frames and encode them as an MP4.

    Each frame composites:
      - Depth-colormap inside the SAM2 mask blended onto RGB
        (falls back to green overlay if *depths* is None)
      - Dimmed background outside the mask
      - GDINO detection rectangle with confidence
      - On-object depth range text
      - Frame index

    Returns:
        Path to the output MP4 file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as _cm

    out = output_dir / "combined"
    out.mkdir(parents=True, exist_ok=True)

    h, w = frames[0].shape[:2]
    video_path = out / "combined_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))

    for i, rgb in enumerate(frames):
        mask = masks[i] if i < len(masks) else np.zeros((h, w), dtype=bool)

        h_m, w_m = mask.shape[:2]
        if (h_m, w_m) != (h, w):
            mask = cv2.resize(
                mask.astype(np.uint8), (w, h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        canvas = rgb.copy().astype(np.float32)
        canvas[~mask] *= 0.3

        depth_min_m = 0.0
        depth_max_m = 0.0
        has_depth = depths is not None and i < len(depths)

        if has_depth:
            depth = depths[i]
            h_d, w_d = depth.shape[:2]
            if (h_d, w_d) != (h, w):
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

            on_obj = depth[mask]
            valid = on_obj[on_obj > 0]
            if len(valid) > 0:
                depth_min_m = float(valid.min())
                depth_max_m = float(valid.max())
                vmin, vmax = np.percentile(valid, [2, 98])
            else:
                vmin, vmax = 0.0, 1.0

            d_norm = np.clip(
                (depth - vmin) / max(vmax - vmin, 1e-6), 0, 1,
            )
            d_color = (_cm.viridis(d_norm)[:, :, :3] * 255).astype(np.float32)
            canvas[mask] = canvas[mask] * 0.4 + d_color[mask] * 0.6
        else:
            green = np.array([0, 255, 0], dtype=np.float32)
            canvas[mask] = canvas[mask] * 0.5 + green * 0.5

        overlay = canvas.astype(np.uint8)

        det = detections[i] if i < len(detections) else {}
        box = det.get("box")
        conf = det.get("confidence", 0.0)
        if box is not None:
            x0, y0, x1, y1 = [int(v) for v in box]
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(
                overlay, f"conf: {conf:.2f}", (x0, max(y0 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )

        cv2.putText(
            overlay, f"frame {i}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )

        if has_depth:
            depth_label = f"depth: [{depth_min_m:.3f}, {depth_max_m:.3f}] m"
            cv2.putText(
                overlay, depth_label, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
            )

        cv2.imwrite(
            str(out / f"frame_{i:04d}.jpg"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )
        writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"   Saved {len(frames)} combined frames + video to {out}")
    return video_path


# ----------------------------------------------------------------------
# VGGT test
# ----------------------------------------------------------------------

def test_vggt(
    frames: List[np.ndarray],
    masks: List[np.ndarray],
    output_dir: Path,
    batch_size: int = 5,
) -> dict:
    """Run VGGT on frames and save depth maps + camera trajectory.

    Returns summary dict with pose/depth quality stats.
    """
    from vision_server import local_api as vision

    out = output_dir / "vggt"
    out.mkdir(parents=True, exist_ok=True)

    overlap = 1
    all_extrinsics = np.zeros((len(frames), 3, 4), dtype=np.float64)
    all_intrinsics = np.zeros((len(frames), 3, 3), dtype=np.float64)
    all_depths = []

    start = 0
    batch_idx = 0
    while start < len(frames):
        end = min(start + batch_size, len(frames))
        batch_frames = frames[start:end]
        batch_masks = masks[start:end]

        result = vision.vggt_batch(batch_frames, masks=batch_masks)

        if batch_idx == 0:
            all_extrinsics[start:end] = result.extrinsics
            all_intrinsics[start:end] = result.intrinsics
            all_depths.extend(
                [result.depths[i] for i in range(len(batch_frames))]
            )
        else:
            prev_ext = np.eye(4)
            prev_ext[:3, :] = all_extrinsics[start]
            curr_ext = np.eye(4)
            curr_ext[:3, :] = result.extrinsics[0]
            alignment = prev_ext @ np.linalg.inv(curr_ext)

            for j in range(len(batch_frames)):
                local_ext = np.eye(4)
                local_ext[:3, :] = result.extrinsics[j]
                aligned = alignment @ local_ext
                idx = start + j
                if idx < len(frames):
                    all_extrinsics[idx] = aligned[:3, :]
                    all_intrinsics[idx] = result.intrinsics[j]

            for j in range(overlap, len(batch_frames)):
                idx = start + j
                if idx < len(frames):
                    all_depths.append(result.depths[j])

        batch_idx += 1
        start = end - overlap
        if start >= len(frames) - 1:
            break

    depths = np.stack(all_depths[:len(frames)])

    # Save depth visualizations
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as _cm
    import matplotlib.pyplot as mplt

    for i in range(len(frames)):
        d = depths[i].squeeze()
        d_valid = d[d > 0]
        if len(d_valid) > 0:
            vmin, vmax = np.percentile(d_valid, [2, 98])
        else:
            vmin, vmax = 0, 1
        d_norm = np.clip((d - vmin) / max(vmax - vmin, 1e-6), 0, 1)
        d_color = (_cm.viridis(d_norm)[:, :, :3] * 255).astype(np.uint8)
        cv2.putText(
            d_color, f"frame {i}  depth [{vmin:.3f}, {vmax:.3f}]m",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
        cv2.imwrite(
            str(out / f"depth_{i:04d}.jpg"),
            cv2.cvtColor(d_color, cv2.COLOR_RGB2BGR),
        )

    # Camera trajectory plot
    positions = all_extrinsics[:, :3, 3]
    fig = mplt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], "b.-", alpha=0.5)
    ax.scatter(*positions[0], c="green", s=100, label="start")
    ax.scatter(*positions[-1], c="red", s=100, label="end")
    scale = 0.1
    step = max(1, len(all_extrinsics) // 15)
    for i in range(0, len(all_extrinsics), step):
        R = all_extrinsics[i][:3, :3]
        t = all_extrinsics[i][:3, 3]
        look_dir = R[:, 2]
        ax.quiver(*t, *(-look_dir * scale), color="red", alpha=0.5,
                  arrow_length_ratio=0.3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title(f"Camera Trajectory ({len(all_extrinsics)} frames)")
    fig.savefig(str(out / "camera_trajectory.png"), dpi=150, bbox_inches="tight")
    mplt.close(fig)

    # Save pose data
    poses = {
        "extrinsics": all_extrinsics.tolist(),
        "intrinsics": all_intrinsics.tolist(),
    }
    with open(str(out / "poses.json"), "w") as f:
        json.dump(poses, f)

    np.savez(
        str(out / "vggt_data.npz"),
        extrinsics=all_extrinsics,
        intrinsics=all_intrinsics,
        depths=depths,
    )

    # Sanity checks
    all_valid_depths = depths[depths > 0]
    depth_min = float(all_valid_depths.min()) if len(all_valid_depths) else 0.0
    depth_max = float(all_valid_depths.max()) if len(all_valid_depths) else 0.0
    depth_ok = 0.01 < depth_min and depth_max < 50.0

    # Check for sudden position jumps
    diffs = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    median_diff = float(np.median(diffs)) if len(diffs) else 0.0
    pose_jumps = int(np.sum(diffs > median_diff * 5)) if median_diff > 0 else 0

    # Check if trajectory is roughly circular (endpoint close to start)
    if len(positions) >= 2:
        closure = float(np.linalg.norm(positions[-1] - positions[0]))
        path_len = float(np.sum(diffs))
        is_circular = closure < path_len * 0.5 if path_len > 0 else False
    else:
        closure = 0.0
        is_circular = False

    summary = {
        "total_frames": len(frames),
        "depth_range": [round(depth_min, 3), round(depth_max, 3)],
        "depth_ok": depth_ok,
        "trajectory_circular": is_circular,
        "pose_jumps": pose_jumps,
        "pass": depth_ok and pose_jumps <= 2,
    }
    return summary


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def run_test(
    object_name: str,
    scan_dir: str = DEFAULT_SCAN_DIR,
    output_dir: Optional[str] = None,
    description: Optional[str] = None,
    frame_subsample: int = 5,
    max_frames: Optional[int] = None,
    depth_source: str = "realsense",
    skip_vggt: bool = True,
    reanchor_interval: int = 30,
) -> dict:
    """Run the full vision pipeline test and return the report.

    Args:
        depth_source: ``"realsense"`` (default) to load pre-recorded D455
            depth frames, or ``"vggt"`` to infer depth from RGB via VGGT.
        skip_vggt: When True (default), skips the VGGT step.  Only
            relevant when ``depth_source="vggt"``.
    """

    scan_path = Path(scan_dir) / object_name
    if not scan_path.exists():
        print(f"Error: scan directory not found: {scan_path}", file=sys.stderr)
        sys.exit(1)

    if output_dir is None:
        output_dir = os.path.join(DEFAULT_OUTPUT_DIR, object_name)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    text_prompt = description or object_name

    print("=" * 60)
    print(f"Vision Pipeline Test: {object_name}  (depth: {depth_source})")
    print("=" * 60)

    # 1. Load frames
    print("\n1. Loading frames...")
    t0 = time.time()
    frames, frames_dir, frame_indices, total_frames = load_frames(
        scan_path, subsample=frame_subsample, max_frames=max_frames,
    )
    print(f"   Loaded {len(frames)} frames (subsample={frame_subsample} "
          f"from {total_frames} total) in {time.time() - t0:.1f}s")

    report = {
        "object_name": object_name,
        "text_prompt": text_prompt,
        "depth_source": depth_source,
        "frames_processed": len(frames),
        "frames_total": total_frames,
        "frame_subsample": frame_subsample,
    }

    # 2. GDINO detection
    print("\n2. Running Grounding DINO detection on all frames...")
    t0 = time.time()
    gdino_summary, gdino_detections = test_gdino(frames, text_prompt, out_path)
    gdino_time = time.time() - t0
    report["gdino"] = gdino_summary
    report["gdino"]["time_sec"] = round(gdino_time, 1)

    print(f"   Detection rate:      {gdino_summary['detected']}/"
          f"{gdino_summary['total_frames']} "
          f"({gdino_summary['detection_rate']*100:.1f}%)"
          f"  {'PASS' if gdino_summary['pass'] else 'FAIL'}")
    print(f"   Avg confidence:      {gdino_summary['avg_confidence']:.3f}")
    if gdino_summary['missed_frames']:
        print(f"   Missed frames:       {gdino_summary['missed_frames']}")
    print(f"   Time:                {gdino_time:.1f}s")

    # 3. SAM2 video tracking
    print("\n3. Running SAM2 video tracking...")
    t0 = time.time()
    sam2_summary = test_sam2_tracking(
        frames, frames_dir, frame_indices, text_prompt, out_path,
        reanchor_interval=reanchor_interval,
    )
    sam2_time = time.time() - t0
    report["sam2"] = sam2_summary
    report["sam2"]["time_sec"] = round(sam2_time, 1)

    print(f"   Median mask area:    {sam2_summary['median_area_pct']:.1f}%")
    print(f"   Area range:          [{sam2_summary['min_area_pct']:.1f}%, "
          f"{sam2_summary['max_area_pct']:.1f}%]")
    print(f"   Drift flags:         {len(sam2_summary['drift_flags'])}/{len(frames)}"
          f"  {'PASS' if not sam2_summary['drift_flags'] else 'WARN'}")
    print(f"   Avg IoU:             {sam2_summary['avg_iou']:.3f}"
          f"  {'PASS' if sam2_summary['avg_iou'] >= 0.5 else 'FAIL'}")
    print(f"   Time:                {sam2_time:.1f}s")

    # Load masks for depth testing
    masks_npz = np.load(str(out_path / "sam2_tracked" / "masks.npz"))
    masks_list = [masks_npz["masks"][i] for i in range(masks_npz["masks"].shape[0])]

    # 4. Depth
    depth_frames_for_combined: Optional[List[np.ndarray]] = None

    if depth_source == "realsense":
        print("\n4. Testing RealSense depth...")
        t0 = time.time()
        rs_depths = load_depth_frames(scan_path, frame_indices)
        if rs_depths is None:
            print("   ERROR: No depth/ directory in scan — "
                  "re-record with depth or use --depth-source vggt",
                  file=sys.stderr)
            report["depth"] = {"error": "no depth data", "pass": False}
        else:
            depth_frames_for_combined = rs_depths
            print(f"   Loaded {len(rs_depths)} depth frames")
            depth_summary = test_depth_realsense(
                frames, rs_depths, masks_list, out_path,
            )
            depth_time = time.time() - t0
            report["depth"] = depth_summary
            report["depth"]["time_sec"] = round(depth_time, 1)

            print(f"   On-object depth:     "
                  f"[{depth_summary['depth_range'][0]:.4f}m, "
                  f"{depth_summary['depth_range'][1]:.4f}m]"
                  f"  {'PASS' if depth_summary['depth_ok'] else 'FAIL'}")
            print(f"   Mean on-object:      "
                  f"{depth_summary['mean_depth']:.4f}m")
            print(f"   Valid pixel ratio:   "
                  f"{depth_summary['avg_valid_ratio']:.1%}"
                  f"  {'PASS' if depth_summary['coverage_ok'] else 'FAIL'}")
            print(f"   Time:                {depth_time:.1f}s")

    elif depth_source == "vggt":
        if not skip_vggt:
            print("\n4. Running VGGT depth/pose estimation...")
            t0 = time.time()
            vggt_summary = test_vggt(frames, masks_list, out_path)
            vggt_time = time.time() - t0
            report["depth"] = vggt_summary
            report["depth"]["time_sec"] = round(vggt_time, 1)

            print(f"   Depth range:         "
                  f"[{vggt_summary['depth_range'][0]:.3f}m, "
                  f"{vggt_summary['depth_range'][1]:.3f}m]"
                  f"  {'PASS' if vggt_summary['depth_ok'] else 'FAIL'}")
            print(f"   Trajectory circular: "
                  f"{'yes' if vggt_summary['trajectory_circular'] else 'no'}"
                  f"  {'PASS' if vggt_summary['trajectory_circular'] else 'WARN'}")
            print(f"   Pose jumps:          {vggt_summary['pose_jumps']}"
                  f"  {'PASS' if vggt_summary['pose_jumps'] <= 2 else 'FAIL'}")
            print(f"   Time:                {vggt_time:.1f}s")
        else:
            print("\n4. VGGT skipped (--skip-vggt)")
            report["depth"] = {"skipped": True}

    # 5. Combined annotated video
    print("\n5. Generating combined annotated video...")
    t0 = time.time()

    meta_path = scan_path / "metadata.json"
    video_fps = 5.0
    if meta_path.exists():
        with open(str(meta_path)) as f:
            scan_meta = json.load(f)
        video_fps = scan_meta.get("fps", 30) / max(frame_subsample, 1)

    combined_path = generate_combined_video(
        frames, gdino_detections, masks_list,
        depth_frames_for_combined, out_path, fps=video_fps,
    )
    combined_time = time.time() - t0
    report["combined_video"] = str(combined_path)
    print(f"   Video:               {combined_path}")
    print(f"   Time:                {combined_time:.1f}s")

    # 6. Summary
    overall = (
        gdino_summary.get("pass", False)
        and sam2_summary.get("pass", False)
        and report.get("depth", {}).get("pass", False)
    )
    report["overall_pass"] = overall

    print(f"\n{'=' * 60}")
    print(f"Vision Pipeline Test Report: {object_name}  (depth: {depth_source})")
    print(f"{'=' * 60}")
    print(f"  Frames processed:     {len(frames)} "
          f"(subsampled from {total_frames})")
    print()
    print(f"  GDINO:")
    print(f"    Detection rate:     {gdino_summary['detected']}/"
          f"{gdino_summary['total_frames']} "
          f"({gdino_summary['detection_rate']*100:.1f}%)"
          f"  {'PASS' if gdino_summary['pass'] else 'FAIL'}")
    print(f"    Avg confidence:     {gdino_summary['avg_confidence']:.3f}")
    if gdino_summary['missed_frames']:
        print(f"    Missed frames:      {gdino_summary['missed_frames']}")
    print()
    print(f"  SAM2 Video Tracking:")
    print(f"    Median mask area:   {sam2_summary['median_area_pct']:.1f}%")
    print(f"    Area range:         [{sam2_summary['min_area_pct']:.1f}%, "
          f"{sam2_summary['max_area_pct']:.1f}%]")
    print(f"    Drift flags:        "
          f"{len(sam2_summary['drift_flags'])}/{len(frames)}"
          f"           {'PASS' if not sam2_summary['drift_flags'] else 'WARN'}")
    print(f"    Mask IoU:           {sam2_summary['avg_iou']:.3f} avg"
          f"     {'PASS' if sam2_summary['avg_iou'] >= 0.5 else 'FAIL'}")

    ds = report.get("depth", {})
    if not ds.get("skipped"):
        print()
        if depth_source == "realsense" and "error" not in ds:
            print(f"  RealSense Depth:")
            print(f"    Depth range:        "
                  f"[{ds['depth_range'][0]:.4f}m, {ds['depth_range'][1]:.4f}m]"
                  f"  {'PASS' if ds['depth_ok'] else 'FAIL'}")
            print(f"    Mean on-object:     {ds['mean_depth']:.4f}m")
            print(f"    Valid pixel ratio:  {ds['avg_valid_ratio']:.1%}"
                  f"       {'PASS' if ds['coverage_ok'] else 'FAIL'}")
        elif depth_source == "realsense" and "error" in ds:
            print(f"  RealSense Depth:      ERROR — {ds['error']}")
        elif depth_source == "vggt":
            print(f"  VGGT:")
            print(f"    Depth range:        "
                  f"[{ds['depth_range'][0]:.3f}m, {ds['depth_range'][1]:.3f}m]"
                  f" {'PASS' if ds['depth_ok'] else 'FAIL'}")
            print(f"    Trajectory shape:   "
                  f"{'circular' if ds.get('trajectory_circular') else 'open'}"
                  f"       {'PASS' if ds.get('trajectory_circular') else 'WARN'}")
            print(f"    Pose jumps:         {ds.get('pose_jumps', '?')}"
                  f"              {'PASS' if ds.get('pose_jumps', 99) <= 2 else 'FAIL'}")

    print()
    print(f"  Overall:              {'PASS' if overall else 'FAIL'}")
    print(f"  Output:               {out_path}")
    print(f"{'=' * 60}")

    # Save report
    with open(str(out_path / "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Test the SAM2 + GDINO + depth vision pipeline on a scan."
    )
    parser.add_argument(
        "--object", required=True,
        help="Name of the object (scan subdirectory)",
    )
    parser.add_argument(
        "--scan-dir", default=DEFAULT_SCAN_DIR,
        help="Root directory containing scan subdirectories",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Where to save debug output",
    )
    parser.add_argument(
        "--description", default=None,
        help="Text description for GDINO (default: object name)",
    )
    parser.add_argument(
        "--frame-subsample", type=int, default=5,
        help="Take every Nth frame (default: 5)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Cap total frames processed",
    )
    parser.add_argument(
        "--depth-source",
        choices=["realsense", "vggt"], default="realsense",
        help="Depth source: pre-recorded D455 frames or VGGT inference "
             "(default: realsense)",
    )
    vggt_group = parser.add_mutually_exclusive_group()
    vggt_group.add_argument(
        "--skip-vggt", action="store_true", default=True,
        help="Skip VGGT step (default; only relevant with --depth-source vggt)",
    )
    vggt_group.add_argument(
        "--run-vggt", action="store_true",
        help="Run VGGT step (only relevant with --depth-source vggt)",
    )
    parser.add_argument(
        "--reanchor-interval", type=int, default=30,
        help="GDINO re-anchor check interval (default: 30 frames)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    report = run_test(
        object_name=args.object,
        scan_dir=args.scan_dir,
        output_dir=args.output_dir,
        description=args.description,
        frame_subsample=args.frame_subsample,
        max_frames=args.max_frames,
        depth_source=args.depth_source,
        skip_vggt=not args.run_vggt,
        reanchor_interval=args.reanchor_interval,
    )

    sys.exit(0 if report.get("overall_pass", False) else 1)


if __name__ == "__main__":
    main()
