"""Record a turntable scan video using an RGBD camera.

Supports **Orbbec Gemini 2** (default) and **RealSense D455** cameras.
Saves an annotated MP4 video, aligned depth frames (16-bit PNG, millimetres),
camera intrinsics, and a metadata JSON file that the ``turntable-learn``
pipeline reads.

Usage::

    record-scan --object cup --description "white ceramic cup"
    record-scan --object cup --camera realsense
    record-scan --object toy --no-preview --duration 10
    record-scan --object toy --no-depth

Press Enter or Ctrl+C to stop recording.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

DEFAULT_SCAN_DIR = os.environ.get("SCAN_DIR", "./data/monty/scans")
_WARMUP_FRAMES = 30


# ---------------------------------------------------------------------------
# Depth utilities
# ---------------------------------------------------------------------------

def _colorize_depth(
    depth_uint16: np.ndarray,
    min_m: float,
    max_m: float,
) -> np.ndarray:
    """uint16 depth (mm) → BGR TURBO colormap.  Zero stays black."""
    min_mm, max_mm = min_m * 1000.0, max_m * 1000.0
    depth_f = np.clip(depth_uint16.astype(np.float32), min_mm, max_mm)
    mask = depth_uint16 > 0
    norm = ((depth_f - min_mm) / (max_mm - min_mm) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    colored[~mask] = 0
    return colored


def _depth_to_pointcloud(
    depth_m: np.ndarray,
    color_bgr: np.ndarray,
    fx: float, fy: float,
    ppx: float, ppy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Deproject depth → coloured 3-D point cloud (xyz float32, rgb uint8)."""
    mask = depth_m > 0
    v, u = np.where(mask)
    z = depth_m[v, u]
    x = (u - ppx) * z / fx
    y = (v - ppy) * z / fy
    xyz = np.stack([x, y, z], axis=-1).astype(np.float32)
    rgb = color_bgr[v, u, ::-1].copy()
    return xyz, rgb


def _write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Write a coloured point cloud to a binary-little-endian PLY file."""
    n = len(xyz)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    dt = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ])
    data = np.empty(n, dtype=dt)
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["r"], data["g"], data["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def _visualize_pointcloud(
    xyz: np.ndarray,
    rgb: np.ndarray,
    title: str,
    save_path: Path,
    max_points: int = 80_000,
) -> None:
    """Render a 3-D scatter plot of the point cloud and save as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(xyz) > max_points:
        idx = np.random.default_rng(42).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]

    colors = rgb.astype(np.float32) / 255.0
    fig = plt.figure(figsize=(16, 7))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(xyz[:, 0], xyz[:, 2], -xyz[:, 1], c=colors, s=0.3, marker=".")
    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Z (m)"); ax1.set_zlabel("-Y (m)")
    ax1.set_title(f"{title} — RGB"); ax1.view_init(elev=-75, azim=-90)

    ax2 = fig.add_subplot(122, projection="3d")
    sc = ax2.scatter(xyz[:, 0], xyz[:, 2], -xyz[:, 1],
                     c=xyz[:, 2], cmap="turbo", s=0.3, marker=".")
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Z (m)"); ax2.set_zlabel("-Y (m)")
    ax2.set_title(f"{title} — depth"); ax2.view_init(elev=-75, azim=-90)
    fig.colorbar(sc, ax=ax2, shrink=0.6, label="depth (m)")

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  pointcloud viz: {save_path}")


# ---------------------------------------------------------------------------
# Annotated video overlay
# ---------------------------------------------------------------------------

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SM = cv2.FONT_HERSHEY_PLAIN


def _draw_text_bg(img, text, org, font=_FONT, scale=0.55, color=(255, 255, 255),
                  bg=(0, 0, 0), thickness=1, pad=4):
    """Draw text with a semi-transparent dark background rectangle."""
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    overlay = img.copy()
    cv2.rectangle(overlay, (x - pad, y - th - pad),
                  (x + tw + pad, y + baseline + pad), bg, -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _annotate_frame(
    img: np.ndarray,
    object_name: str,
    frame_idx: int,
    elapsed: float,
    camera_tag: str,
    res_w: int, res_h: int, fps: int,
    depth_mm: np.ndarray | None = None,
    depth_color: np.ndarray | None = None,
    depth_min_m: float = 0.2,
    depth_max_m: float = 3.0,
) -> np.ndarray:
    """Burn full annotation overlay onto a frame (returns a copy)."""
    out = img.copy()
    h, w = out.shape[:2]

    _draw_text_bg(out, f"Object: {object_name}", (10, 28), scale=0.65,
                  color=(0, 255, 200))

    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    _draw_text_bg(out, f"Frame {frame_idx:06d}  |  {elapsed:.1f}s  |  {ts}",
                  (10, 56), scale=0.5, color=(200, 255, 200))

    _draw_text_bg(out, f"{camera_tag}  {res_w}x{res_h} @ {fps}fps",
                  (10, h - 14), scale=0.45, color=(180, 180, 255))

    if depth_mm is not None:
        valid = depth_mm[depth_mm > 0]
        total = depth_mm.size
        coverage = 100.0 * len(valid) / total if total else 0
        if len(valid) > 0:
            med_m = float(np.median(valid)) / 1000.0
            mn_m = float(valid.min()) / 1000.0
            mx_m = float(valid.max()) / 1000.0
            dtxt = (f"Depth: {coverage:.0f}% cov  "
                    f"med={med_m:.3f}m  min={mn_m:.3f}m  max={mx_m:.3f}m")
        else:
            dtxt = f"Depth: {coverage:.0f}% coverage (no data)"
        _draw_text_bg(out, dtxt, (10, 82), scale=0.5, color=(100, 200, 255))

        if depth_color is not None:
            iw, ih = 200, 120
            inset = cv2.resize(depth_color, (iw, ih), interpolation=cv2.INTER_AREA)
            x0, y0 = w - iw - 10, 10
            out[y0:y0 + ih, x0:x0 + iw] = inset
            cv2.rectangle(out, (x0 - 1, y0 - 1), (x0 + iw, y0 + ih),
                          (255, 255, 255), 1)
            _draw_text_bg(out, "depth", (x0 + 2, y0 + ih - 4),
                          scale=0.4, color=(255, 255, 255))

    return out


# ---------------------------------------------------------------------------
# GUI helper
# ---------------------------------------------------------------------------

def _gui_available() -> bool:
    try:
        cv2.namedWindow("__probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__probe__")
        return True
    except cv2.error:
        return False


# ---------------------------------------------------------------------------
# Camera backends
# ---------------------------------------------------------------------------

def _orbbec_frame_to_bgr(color_frame) -> np.ndarray | None:
    """Convert an Orbbec colour frame to a BGR numpy array."""
    from pyorbbecsdk import OBFormat, FormatConvertFilter, OBConvertFormat

    w = color_frame.get_width()
    h = color_frame.get_height()
    fmt = color_frame.get_format()
    data = np.asanyarray(color_frame.get_data())

    if fmt == OBFormat.RGB:
        return cv2.cvtColor(data.reshape((h, w, 3)), cv2.COLOR_RGB2BGR)
    elif fmt == OBFormat.BGR:
        return data.reshape((h, w, 3)).copy()
    elif fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif fmt == OBFormat.YUYV:
        return cv2.cvtColor(data.reshape((h, w, 2)), cv2.COLOR_YUV2BGR_YUY2)
    elif fmt == OBFormat.UYVY:
        return cv2.cvtColor(data.reshape((h, w, 2)), cv2.COLOR_YUV2BGR_UYVY)

    _CONV_MAP = {
        OBFormat.I420: OBConvertFormat.I420_TO_RGB888,
        OBFormat.NV21: OBConvertFormat.NV21_TO_RGB888,
        OBFormat.NV12: OBConvertFormat.NV12_TO_RGB888,
    }
    conv = _CONV_MAP.get(fmt)
    if conv is not None:
        flt = FormatConvertFilter()
        flt.set_format_convert_format(conv)
        rgb_frame = flt.process(color_frame)
        if rgb_frame is not None:
            d2 = np.asanyarray(rgb_frame.get_data())
            return cv2.cvtColor(d2.reshape((h, w, 3)), cv2.COLOR_RGB2BGR)
    return None


def _record_orbbec(
    object_name: str,
    out_dir: Path,
    description: str,
    width: int, height: int, fps: int,
    preview: bool,
    record_depth: bool,
    duration: float | None,
    depth_min_m: float,
    depth_max_m: float,
) -> tuple[int, float, dict]:
    """Capture loop for Orbbec Gemini cameras via pyorbbecsdk."""
    from pyorbbecsdk import (
        Pipeline, Config, OBSensorType, OBFormat,
        OBStreamType, OBFrameAggregateOutputMode, AlignFilter,
    )

    video_path = out_dir / "video.mp4"
    annotated_path = out_dir / "video_annotated.mp4"
    depth_dir = out_dir / "depth"
    depth_color_dir = out_dir / "depth_color"
    depth_float_dir = out_dir / "depth_float"
    pointcloud_dir = out_dir / "pointcloud"
    if record_depth:
        for d in (depth_dir, depth_color_dir, depth_float_dir, pointcloud_dir):
            d.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline()
    config = Config()

    profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    try:
        color_profile = profile_list.get_video_stream_profile(
            width, height, OBFormat.MJPG, fps)
    except Exception:
        color_profile = profile_list.get_default_video_stream_profile()
        print(f"WARNING: {width}x{height}@{fps} MJPG not available, "
              f"using {color_profile.get_width()}x{color_profile.get_height()}"
              f"@{color_profile.get_fps()} {color_profile.get_format()}")
    config.enable_stream(color_profile)

    depth_profile = None
    if record_depth:
        dp_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        try:
            depth_profile = dp_list.get_video_stream_profile(
                0, 0, OBFormat.Y14, fps)
        except Exception:
            depth_profile = dp_list.get_default_video_stream_profile()
            print(f"WARNING: Y14 depth not available, using "
                  f"{depth_profile.get_format()}")
        config.enable_stream(depth_profile)
        config.set_frame_aggregate_output_mode(
            OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)

    pipeline.start(config)

    actual_w = color_profile.get_width()
    actual_h = color_profile.get_height()
    actual_fps = color_profile.get_fps()

    align_filter = None
    depth_scale = 0.0
    color_intrinsics: dict | None = None

    if record_depth:
        align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        print(f"Depth: {depth_profile.get_width()}x{depth_profile.get_height()} "
              f"→ aligned to {actual_w}x{actual_h}")

    if preview and not _gui_available():
        print("WARNING: no GUI backend — disabling preview")
        preview = False

    camera_tag = "Orbbec Gemini"
    depth_label = " + depth" if record_depth else ""
    print(f"Recording {object_name} at {actual_w}x{actual_h} "
          f"@ {actual_fps} fps{depth_label}  [{camera_tag}]")
    if duration is not None:
        print(f"Recording for {duration}s (Ctrl+C to stop early)...")
    else:
        print("Press Enter or Ctrl+C to stop recording...")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, actual_fps,
                             (actual_w, actual_h))
    ann_writer = cv2.VideoWriter(str(annotated_path), fourcc, actual_fps,
                                 (actual_w, actual_h))

    stop_event = threading.Event()
    frame_count = 0
    start_time = time.monotonic()

    def _on_signal(signum, _frame):
        stop_event.set()

    prev_sigint = signal.signal(signal.SIGINT, _on_signal)
    prev_sigterm = signal.signal(signal.SIGTERM, _on_signal)

    if duration is not None:
        threading.Thread(target=lambda: (stop_event.wait(timeout=duration),
                                         stop_event.set()),
                         daemon=True).start()

    if sys.stdin.isatty():
        def _wait_enter():
            try:
                input()
            except EOFError:
                pass
            stop_event.set()
        threading.Thread(target=_wait_enter, daemon=True).start()

    if record_depth:
        print(f"Warming up ({_WARMUP_FRAMES} frames)...", end="", flush=True)
        for _ in range(_WARMUP_FRAMES):
            pipeline.wait_for_frames(5000)
        print(" done")

    try:
        while not stop_event.is_set():
            frames = pipeline.wait_for_frames(5000)
            if frames is None:
                continue

            if record_depth and align_filter is not None:
                frames = align_filter.process(frames)
                if frames is None:
                    continue
                frames = frames.as_frame_set()

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            img = _orbbec_frame_to_bgr(color_frame)
            if img is None:
                continue

            depth_mm_img: np.ndarray | None = None
            depth_vis: np.ndarray | None = None

            if record_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    if depth_scale == 0.0:
                        depth_scale = depth_frame.get_depth_scale()
                        intr = (color_frame.get_stream_profile()
                                .as_video_stream_profile().get_intrinsic())
                        color_intrinsics = dict(
                            fx=intr.fx, fy=intr.fy,
                            ppx=intr.cx, ppy=intr.cy,
                            width=intr.width, height=intr.height,
                        )
                        print(f"Depth scale: {depth_scale}, "
                              f"intrinsics: fx={intr.fx:.1f} fy={intr.fy:.1f} "
                              f"cx={intr.cx:.1f} cy={intr.cy:.1f}")

                    dw = depth_frame.get_width()
                    dh = depth_frame.get_height()
                    raw = np.frombuffer(depth_frame.get_data(),
                                       dtype=np.uint16).reshape((dh, dw))
                    depth_mm_img = (raw.astype(np.float32)
                                    * depth_scale).astype(np.uint16)

                    min_mm = int(depth_min_m * 1000)
                    max_mm = int(depth_max_m * 1000)
                    depth_mm_img = np.where(
                        (depth_mm_img >= min_mm) & (depth_mm_img <= max_mm),
                        depth_mm_img, 0).astype(np.uint16)

                    fname = f"{frame_count:06d}.png"
                    cv2.imwrite(str(depth_dir / fname), depth_mm_img)
                    depth_vis = _colorize_depth(depth_mm_img,
                                                depth_min_m, depth_max_m)
                    cv2.imwrite(str(depth_color_dir / fname), depth_vis)
                    depth_metres = depth_mm_img.astype(np.float32) / 1000.0
                    np.save(str(depth_float_dir / f"{frame_count:06d}.npy"),
                            depth_metres)

                    if frame_count % max(actual_fps, 1) == 0:
                        xyz, rgb = _depth_to_pointcloud(
                            depth_metres, img,
                            color_intrinsics["fx"], color_intrinsics["fy"],
                            color_intrinsics["ppx"], color_intrinsics["ppy"],
                        )
                        _write_ply(pointcloud_dir / f"{frame_count:06d}.ply",
                                   xyz, rgb)

            writer.write(img)

            elapsed = time.monotonic() - start_time
            annotated = _annotate_frame(
                img, object_name, frame_count, elapsed,
                camera_tag, actual_w, actual_h, actual_fps,
                depth_mm=depth_mm_img, depth_color=depth_vis,
                depth_min_m=depth_min_m, depth_max_m=depth_max_m,
            )
            ann_writer.write(annotated)

            frame_count += 1

            if preview:
                display = cv2.resize(annotated, (640, 360))
                cv2.imshow("record-scan", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            elif frame_count % max(actual_fps, 1) == 0:
                el = time.monotonic() - start_time
                print(f"\r  {frame_count} frames  ({el:.1f}s)",
                      end="", flush=True)
    finally:
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)
        pipeline.stop()
        writer.release()
        ann_writer.release()
        if preview:
            cv2.destroyAllWindows()

    elapsed = time.monotonic() - start_time
    metadata = dict(
        object_name=object_name,
        description=description,
        camera="orbbec_gemini",
        width=actual_w, height=actual_h, fps=actual_fps,
        frame_count=frame_count,
        duration_sec=round(elapsed, 2),
        timestamp=datetime.now(timezone.utc).isoformat(),
        depth_enabled=record_depth,
    )
    if record_depth and color_intrinsics is not None:
        metadata["depth_scale"] = depth_scale
        metadata["color_intrinsics"] = color_intrinsics
        metadata["depth_range_m"] = [depth_min_m, depth_max_m]
    return frame_count, elapsed, metadata


def _record_realsense(
    object_name: str,
    out_dir: Path,
    description: str,
    width: int, height: int, fps: int,
    preview: bool,
    record_depth: bool,
    duration: float | None,
    depth_min_m: float,
    depth_max_m: float,
) -> tuple[int, float, dict]:
    """Capture loop for Intel RealSense D4xx cameras."""
    import pyrealsense2 as rs

    video_path = out_dir / "video.mp4"
    annotated_path = out_dir / "video_annotated.mp4"
    depth_dir = out_dir / "depth"
    depth_color_dir = out_dir / "depth_color"
    depth_float_dir = out_dir / "depth_float"
    pointcloud_dir = out_dir / "pointcloud"
    if record_depth:
        for d in (depth_dir, depth_color_dir, depth_float_dir, pointcloud_dir):
            d.mkdir(parents=True, exist_ok=True)

    rs_depth_w, rs_depth_h = 848, 480

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    if record_depth:
        config.enable_stream(rs.stream.depth, rs_depth_w, rs_depth_h,
                             rs.format.z16, fps)

    profile = pipeline.start(config)

    depth_sensor = None
    if record_depth:
        depth_sensor = profile.get_device().first_depth_sensor()
        if depth_sensor.supports(rs.option.visual_preset):
            depth_sensor.set_option(rs.option.visual_preset, 1)
        if depth_sensor.supports(rs.option.laser_power):
            mx = depth_sensor.get_option_range(rs.option.laser_power).max
            depth_sensor.set_option(rs.option.laser_power, mx)

    stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    actual_w, actual_h, actual_fps = stream.width(), stream.height(), stream.fps()

    depth_scale = 0.0
    color_intrinsics = None
    align = None
    depth_filters = []
    if record_depth:
        depth_scale = depth_sensor.get_depth_scale()
        intr = stream.get_intrinsics()
        color_intrinsics = dict(fx=intr.fx, fy=intr.fy, ppx=intr.ppx,
                                ppy=intr.ppy, width=intr.width, height=intr.height)
        align = rs.align(rs.stream.color)

        threshold = rs.threshold_filter()
        threshold.set_option(rs.option.min_distance, depth_min_m)
        threshold.set_option(rs.option.max_distance, depth_max_m)
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 2)
        spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        spatial.set_option(rs.option.filter_smooth_delta, 20)
        spatial.set_option(rs.option.holes_fill, 0)
        temporal = rs.temporal_filter()
        temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
        temporal.set_option(rs.option.filter_smooth_delta, 20)
        depth_filters = [threshold, spatial, temporal]

    if preview and not _gui_available():
        preview = False

    camera_tag = "RealSense D4xx"
    depth_label = " + depth" if record_depth else ""
    print(f"Recording {object_name} at {actual_w}x{actual_h} "
          f"@ {actual_fps} fps{depth_label}  [{camera_tag}]")
    if duration is not None:
        print(f"Recording for {duration}s ...")
    else:
        print("Press Enter or Ctrl+C to stop recording...")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, actual_fps,
                             (actual_w, actual_h))
    ann_writer = cv2.VideoWriter(str(annotated_path), fourcc, actual_fps,
                                 (actual_w, actual_h))

    stop_event = threading.Event()
    frame_count = 0
    start_time = time.monotonic()

    def _on_signal(signum, _frame):
        stop_event.set()
    prev_sigint = signal.signal(signal.SIGINT, _on_signal)
    prev_sigterm = signal.signal(signal.SIGTERM, _on_signal)

    if duration is not None:
        threading.Thread(target=lambda: (stop_event.wait(timeout=duration),
                                         stop_event.set()),
                         daemon=True).start()
    if sys.stdin.isatty():
        def _we():
            try:
                input()
            except EOFError:
                pass
            stop_event.set()
        threading.Thread(target=_we, daemon=True).start()

    if record_depth:
        print(f"Warming up ({_WARMUP_FRAMES} frames)...", end="", flush=True)
        for _ in range(_WARMUP_FRAMES):
            pipeline.wait_for_frames(timeout_ms=5000)
        print(" done")

    try:
        while not stop_event.is_set():
            frameset = pipeline.wait_for_frames(timeout_ms=5000)
            if record_depth:
                frameset = align.process(frameset)
            color_frame = frameset.get_color_frame()
            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())

            depth_mm_img = None
            depth_vis = None
            if record_depth:
                depth_frame = frameset.get_depth_frame()
                if depth_frame:
                    for flt in depth_filters:
                        depth_frame = flt.process(depth_frame)
                    depth_raw = np.asanyarray(depth_frame.get_data())
                    depth_mm_img = depth_raw.copy()
                    fname = f"{frame_count:06d}.png"
                    cv2.imwrite(str(depth_dir / fname), depth_mm_img)
                    depth_vis = _colorize_depth(depth_mm_img,
                                                depth_min_m, depth_max_m)
                    cv2.imwrite(str(depth_color_dir / fname), depth_vis)
                    depth_metres = depth_raw.astype(np.float32) * depth_scale
                    np.save(str(depth_float_dir / f"{frame_count:06d}.npy"),
                            depth_metres)
                    if frame_count % max(actual_fps, 1) == 0:
                        xyz, rgb = _depth_to_pointcloud(
                            depth_metres, img,
                            color_intrinsics["fx"], color_intrinsics["fy"],
                            color_intrinsics["ppx"], color_intrinsics["ppy"])
                        _write_ply(pointcloud_dir / f"{frame_count:06d}.ply",
                                   xyz, rgb)

            writer.write(img)

            elapsed = time.monotonic() - start_time
            annotated = _annotate_frame(
                img, object_name, frame_count, elapsed,
                camera_tag, actual_w, actual_h, actual_fps,
                depth_mm=depth_mm_img, depth_color=depth_vis,
                depth_min_m=depth_min_m, depth_max_m=depth_max_m)
            ann_writer.write(annotated)

            frame_count += 1

            if preview:
                display = cv2.resize(annotated, (640, 360))
                cv2.imshow("record-scan", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            elif frame_count % max(actual_fps, 1) == 0:
                el = time.monotonic() - start_time
                print(f"\r  {frame_count} frames  ({el:.1f}s)",
                      end="", flush=True)
    finally:
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)
        pipeline.stop()
        writer.release()
        ann_writer.release()
        if preview:
            cv2.destroyAllWindows()

    elapsed = time.monotonic() - start_time
    metadata = dict(
        object_name=object_name, description=description,
        camera="realsense_d4xx",
        width=actual_w, height=actual_h, fps=actual_fps,
        frame_count=frame_count,
        duration_sec=round(elapsed, 2),
        timestamp=datetime.now(timezone.utc).isoformat(),
        depth_enabled=record_depth,
    )
    if record_depth and color_intrinsics:
        metadata["depth_scale"] = depth_scale
        metadata["color_intrinsics"] = color_intrinsics
        metadata["depth_native_resolution"] = f"{rs_depth_w}x{rs_depth_h}"
        metadata["depth_range_m"] = [depth_min_m, depth_max_m]
        metadata["depth_filters"] = "threshold+spatial+temporal"
    return frame_count, elapsed, metadata


# ---------------------------------------------------------------------------
# Main record() entry point
# ---------------------------------------------------------------------------

def record(
    object_name: str,
    scan_dir: str,
    camera: str = "orbbec",
    description: str = "",
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    preview: bool = True,
    record_depth: bool = True,
    duration: float | None = None,
    depth_min_m: float = 0.2,
    depth_max_m: float = 3.0,
) -> Path:
    """Record video + depth from an RGBD camera.

    Returns the output directory.
    """
    out_dir = Path(scan_dir) / object_name
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.json"
    video_path = out_dir / "video.mp4"
    annotated_path = out_dir / "video_annotated.mp4"

    depth_dir = out_dir / "depth"
    depth_color_dir = out_dir / "depth_color"
    depth_float_dir = out_dir / "depth_float"
    pointcloud_dir = out_dir / "pointcloud"

    if camera == "orbbec":
        frame_count, elapsed, metadata = _record_orbbec(
            object_name, out_dir, description,
            width, height, fps, preview, record_depth,
            duration, depth_min_m, depth_max_m)
    elif camera == "realsense":
        frame_count, elapsed, metadata = _record_realsense(
            object_name, out_dir, description,
            width, height, fps, preview, record_depth,
            duration, depth_min_m, depth_max_m)
    else:
        raise ValueError(f"Unknown camera: {camera!r}  (use 'orbbec' or 'realsense')")

    meta_path.write_text(json.dumps(metadata, indent=2))

    color_intrinsics = metadata.get("color_intrinsics")

    print(f"\nSaved {frame_count} frames ({elapsed:.1f}s) to {out_dir}")
    print(f"  video (raw):      {video_path}")
    print(f"  video (annotated): {annotated_path}")
    if record_depth:
        ply_count = len(list(pointcloud_dir.glob("*.ply")))
        print(f"  depth:       {depth_dir}/ ({frame_count} × 16-bit PNG, mm)")
        print(f"  depth_float: {depth_float_dir}/ ({frame_count} × float32 .npy, metres)")
        print(f"  depth_color: {depth_color_dir}/ ({frame_count} × colormap)")
        print(f"  pointcloud:  {pointcloud_dir}/ ({ply_count} × PLY, RGB)")
    print(f"  metadata:    {meta_path}")

    if record_depth and frame_count > 0:
        _check_depth_quality(depth_dir, frame_count)

        actual_fps = metadata.get("fps", fps)
        mid_idx = (frame_count // 2 // max(actual_fps, 1)) * max(actual_fps, 1)
        npy_path = depth_float_dir / f"{mid_idx:06d}.npy"
        if npy_path.exists() and color_intrinsics is not None:
            depth_m = np.load(str(npy_path))
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
            ret, color_bgr = cap.read()
            cap.release()
            if ret and color_bgr is not None:
                xyz, rgb = _depth_to_pointcloud(
                    depth_m, color_bgr,
                    color_intrinsics["fx"], color_intrinsics["fy"],
                    color_intrinsics["ppx"], color_intrinsics["ppy"])
                _visualize_pointcloud(xyz, rgb,
                                      f"{object_name} frame {mid_idx}",
                                      out_dir / "pointcloud_viz.png")

    # save a sample depth image at the mid-point for quick review
    if record_depth and frame_count > 0:
        mid = frame_count // 2
        sample_src = depth_color_dir / f"{mid:06d}.png"
        if sample_src.exists():
            import shutil
            dst = out_dir / "depth_sample.png"
            shutil.copy2(sample_src, dst)
            print(f"  depth sample: {dst}")

    return out_dir


def _check_depth_quality(depth_dir: Path, frame_count: int):
    """Analyze a sample of depth frames and report coverage quality."""
    indices = [0, frame_count // 4, frame_count // 2,
               3 * frame_count // 4, frame_count - 1]
    indices = sorted(set(i for i in indices if i < frame_count))

    coverages = []
    print("\n  Depth quality check:")
    for idx in indices:
        fpath = depth_dir / f"{idx:06d}.png"
        if not fpath.exists():
            continue
        img = cv2.imread(str(fpath), cv2.IMREAD_UNCHANGED)
        total = img.size
        nonzero = np.count_nonzero(img)
        coverage = 100.0 * nonzero / total
        n_unique = len(np.unique(img))
        coverages.append(coverage)
        nz = img[img > 0]
        if len(nz) > 0:
            med_mm = int(np.median(nz))
            print(f"    frame {idx:06d}: {coverage:5.1f}% coverage, "
                  f"{n_unique} unique values, "
                  f"median {med_mm}mm ({med_mm * 0.001:.2f}m)")
        else:
            print(f"    frame {idx:06d}: {coverage:5.1f}% coverage (no depth)")

    if coverages:
        avg = sum(coverages) / len(coverages)
        status = "GOOD" if avg >= 50 else "POOR" if avg >= 20 else "BAD"
        print(f"  Average coverage: {avg:.1f}% — {status}")
        if status != "GOOD":
            print("  Hint: try adjusting --depth-min / --depth-max, "
                  "move object closer, or check lighting")


def main():
    parser = argparse.ArgumentParser(
        description="Record a turntable scan from an RGBD camera (Orbbec / RealSense)."
    )
    parser.add_argument("--object", required=True,
                        help="Name of the object being scanned")
    parser.add_argument("--description", default="",
                        help="Free-text description (saved in metadata.json)")
    parser.add_argument("--camera", default="orbbec",
                        choices=["orbbec", "realsense"],
                        help="Camera backend (default: orbbec)")
    parser.add_argument("--scan-dir", default=DEFAULT_SCAN_DIR,
                        help=f"Root scan directory (default: {DEFAULT_SCAN_DIR})")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-preview", action="store_true",
                        help="Disable live preview window")
    parser.add_argument("--no-depth", action="store_true",
                        help="Disable depth recording")
    parser.add_argument("--duration", type=float, default=None,
                        help="Auto-stop after N seconds")
    parser.add_argument("--depth-min", type=float, default=0.2,
                        help="Min depth range in metres (default: 0.2)")
    parser.add_argument("--depth-max", type=float, default=3.0,
                        help="Max depth range in metres (default: 3.0)")
    args = parser.parse_args()

    record(
        object_name=args.object,
        scan_dir=args.scan_dir,
        camera=args.camera,
        description=args.description,
        width=args.width,
        height=args.height,
        fps=args.fps,
        preview=not args.no_preview,
        record_depth=not args.no_depth,
        duration=args.duration,
        depth_min_m=args.depth_min,
        depth_max_m=args.depth_max,
    )


if __name__ == "__main__":
    main()
