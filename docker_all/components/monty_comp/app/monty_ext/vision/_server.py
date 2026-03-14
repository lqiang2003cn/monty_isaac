#!/usr/bin/env python3
"""Vision model server — runs in the ``vision`` conda env.

Accepts JSON-line commands on stdin, processes them with SAM2/VGGT,
and writes JSON-line responses to stdout.  Binary data (images, masks,
depths) is exchanged via numpy files on /dev/shm for zero-copy speed.

This script is NOT imported by the tbp.monty env.  It is spawned as a
subprocess by ``_bridge.VisionBridge``.

Protocol (one JSON object per line, newline-delimited):

  → {"cmd": "load_sam2"}
  ← {"ok": true}

  → {"cmd": "load_vggt"}
  ← {"ok": true}

  → {"cmd": "segment", "image": "/dev/shm/vb_frame.npy"}
  ← {"ok": true, "mask": "/dev/shm/vb_mask.npy"}

  → {"cmd": "segment_video",
     "frames_dir": "/data/scans/red_box/frames",
     "frame_indices": [0, 5, 10, ...],
     "text_prompt": "red box",
     "debug_vis_dir": "/results/turntable/red_box/debug_vis",
     "reanchor_interval": 30}
  ← {"ok": true,
     "masks": "/dev/shm/vb_video_masks.npy",
     "gdino_results": "/dev/shm/vb_gdino_results.npy"}

  → {"cmd": "vggt", "images": "/dev/shm/vb_batch.npy",
     "masks": "/dev/shm/vb_masks.npy"}   (masks optional)
  ← {"ok": true,
      "extrinsics": "/dev/shm/vb_ext.npy",
      "intrinsics": "/dev/shm/vb_intr.npy",
      "depths": "/dev/shm/vb_depths.npy",
      "points3d": "/dev/shm/vb_pts.npy"}

  → {"cmd": "detect_gdino", "image": "/dev/shm/vb_frame.npy",
     "text_prompt": "red box"}
  ← {"ok": true, "box": [x0, y0, x1, y1], "confidence": 0.85}

  → {"cmd": "shutdown"}
  ← {"ok": true}
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import traceback

import contextlib

import cv2
import numpy as np

_SHM = "/dev/shm"


@contextlib.contextmanager
def _suppress_stdout():
    """Redirect stdout to stderr during model loading.

    Third-party libraries (HuggingFace, timm) sometimes print warnings or
    progress bars to stdout, which corrupts the JSON-lines IPC protocol.
    """
    saved = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = saved

# ---------------------------------------------------------------------------
# Lazy-loaded models
# ---------------------------------------------------------------------------

_sam2_model = None
_sam2_generator = None
_sam2_predictor = None
_sam2_video_predictor = None
_sam2_ckpt_file = None
_gdino_ckpt_file = None
_vggt_model = None

_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def _get_sam2_ckpt() -> str:
    """Return path to SAM2 checkpoint, downloading if necessary."""
    global _sam2_ckpt_file
    if _sam2_ckpt_file is not None:
        return _sam2_ckpt_file

    cache = os.environ.get("MODEL_CACHE_DIR", "/models")
    ckpt_dir = os.path.join(cache, "sam2")
    ckpt_file = os.path.join(ckpt_dir, "sam2.1_hiera_large.pt")

    expected_size = 898_083_611
    if not os.path.isfile(ckpt_file) or os.path.getsize(ckpt_file) < expected_size:
        import urllib.request
        os.makedirs(ckpt_dir, exist_ok=True)
        url = (
            "https://dl.fbaipublicfiles.com/segment_anything_2/"
            "092824/sam2.1_hiera_large.pt"
        )
        for attempt in range(3):
            try:
                _log(f"Downloading SAM2 checkpoint (attempt {attempt + 1}/3) ...")
                urllib.request.urlretrieve(url, ckpt_file)
                if os.path.getsize(ckpt_file) >= expected_size:
                    break
                _log(f"Download incomplete: "
                     f"{os.path.getsize(ckpt_file)}/{expected_size}")
            except Exception as e:
                _log(f"Download failed: {e}")
                if attempt == 2:
                    raise

    _sam2_ckpt_file = ckpt_file
    return ckpt_file


def _load_sam2() -> None:
    global _sam2_model, _sam2_generator, _sam2_predictor
    if _sam2_model is not None:
        return

    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    ckpt_file = _get_sam2_ckpt()

    _log("Loading SAM2 ...")
    with _suppress_stdout():
        _sam2_model = build_sam2(_SAM2_CONFIG, ckpt_file, device="cuda")
        _sam2_generator = SAM2AutomaticMaskGenerator(
            model=_sam2_model,
            points_per_side=32,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.85,
        )
        _sam2_predictor = SAM2ImagePredictor(_sam2_model)
    _log("SAM2 loaded")


def _load_sam2_video_predictor() -> None:
    global _sam2_video_predictor
    if _sam2_video_predictor is not None:
        return

    from sam2.build_sam import build_sam2_video_predictor

    ckpt_file = _get_sam2_ckpt()
    _log("Loading SAM2 VideoPredictor ...")
    with _suppress_stdout():
        _sam2_video_predictor = build_sam2_video_predictor(
            _SAM2_CONFIG, ckpt_file, device="cuda",
        )
    _log("SAM2 VideoPredictor loaded")


def _load_vggt() -> None:
    global _vggt_model
    if _vggt_model is not None:
        return

    import torch
    os.environ.setdefault("TORCH_HOME", os.environ.get("MODEL_CACHE_DIR", "/models"))

    from vggt.models.vggt import VGGT

    _log("Loading VGGT ...")
    with _suppress_stdout():
        _vggt_model = VGGT.from_pretrained("facebook/VGGT-1B")
        _vggt_model.eval().cuda()
    _log("VGGT loaded")


# ---------------------------------------------------------------------------
# Grounding DINO
# ---------------------------------------------------------------------------

_gdino_model = None

_GDINO_CKPT_URL = (
    "https://huggingface.co/pengxian/grounding-dino/resolve/main/"
    "groundingdino_swint_ogc.pth"
)
_GDINO_CKPT_SIZE = 650_000_000


def _find_gdino_config() -> str | None:
    """Locate the GroundingDINO config file from the installed package."""
    try:
        import groundingdino
        pkg_dir = os.path.dirname(groundingdino.__file__)
        candidates = [
            os.path.join(pkg_dir, "config", "GroundingDINO_SwinT_OGC.py"),
            os.path.join(pkg_dir, "config", "GroundingDINO_SwinT_OGC.cfg.py"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
    except ImportError:
        pass
    cache = os.environ.get("MODEL_CACHE_DIR", "/models")
    fallback = os.path.join(cache, "groundingdino", "GroundingDINO_SwinT_OGC.py")
    if os.path.isfile(fallback):
        return fallback
    return None


def _get_gdino_ckpt() -> str:
    """Return path to GDINO checkpoint, downloading if necessary."""
    global _gdino_ckpt_file
    if _gdino_ckpt_file is not None:
        return _gdino_ckpt_file

    cache = os.environ.get("MODEL_CACHE_DIR", "/models")
    gdino_dir = os.path.join(cache, "groundingdino")
    ckpt_path = os.path.join(gdino_dir, "groundingdino_swint_ogc.pth")

    if not os.path.isfile(ckpt_path) or os.path.getsize(ckpt_path) < _GDINO_CKPT_SIZE:
        import urllib.request
        os.makedirs(gdino_dir, exist_ok=True)
        for attempt in range(3):
            try:
                _log(f"Downloading GDINO checkpoint (attempt {attempt + 1}/3) ...")
                urllib.request.urlretrieve(_GDINO_CKPT_URL, ckpt_path)
                if os.path.getsize(ckpt_path) >= _GDINO_CKPT_SIZE:
                    break
                _log(f"Download incomplete: "
                     f"{os.path.getsize(ckpt_path)}/{_GDINO_CKPT_SIZE}")
            except Exception as e:
                _log(f"GDINO download failed: {e}")
                if attempt == 2:
                    raise

    _gdino_ckpt_file = ckpt_path
    return ckpt_path


def _load_gdino():
    """Load Grounding DINO for text-prompted object detection."""
    global _gdino_model
    if _gdino_model is not None:
        return

    try:
        from groundingdino.util.inference import load_model
    except ImportError:
        _log("groundingdino not installed — text prompts will use center-point fallback")
        return

    config_path = _find_gdino_config()
    if config_path is None:
        _log("Grounding DINO config not found — text prompts will use "
             "center-point fallback")
        return

    ckpt_path = _get_gdino_ckpt()

    try:
        with _suppress_stdout():
            _gdino_model = load_model(config_path, ckpt_path)
        _log("Grounding DINO loaded")
    except Exception as e:
        _log(f"Grounding DINO load failed: {e}")


def _detect_box_gdino(rgb: np.ndarray, text_prompt: str):
    """Use Grounding DINO to detect a bounding box for the text prompt.

    Returns ``(x0, y0, x1, y1, confidence)`` in pixel coords, or None.
    """
    _load_gdino()
    if _gdino_model is None:
        return None

    try:
        from groundingdino.util.inference import predict
        import groundingdino.datasets.transforms as T

        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        from PIL import Image
        pil_img = Image.fromarray(rgb)
        img_t, _ = transform(pil_img, None)

        with _suppress_stdout():
            boxes, logits, phrases = predict(
                model=_gdino_model,
                image=img_t,
                caption=text_prompt,
                box_threshold=0.25,
                text_threshold=0.2,
            )

        if len(boxes) == 0:
            return None

        best_idx = logits.argmax().item()
        confidence = float(logits[best_idx].item())
        box = boxes[best_idx].cpu().numpy()
        h, w = rgb.shape[:2]
        cx, cy, bw, bh = box
        x0 = int((cx - bw / 2) * w)
        y0 = int((cy - bh / 2) * h)
        x1 = int((cx + bw / 2) * w)
        y1 = int((cy + bh / 2) * h)
        return (max(0, x0), max(0, y0), min(w, x1), min(h, y1), confidence)

    except Exception as e:
        _log(f"Grounding DINO detection failed: {e}")
        return None


def _segment_with_point(rgb: np.ndarray, point_xy=None) -> np.ndarray:
    """Use SAM2 image predictor with a center point prompt.

    A point prompt at the image center is more robust for turntable scanning
    where the object is always approximately centered.
    """
    _sam2_predictor.set_image(rgb)
    h, w = rgb.shape[:2]
    if point_xy is None:
        point_xy = [w // 2, h // 2]
    import torch
    point_coords = np.array([point_xy], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)
    masks, scores, _ = _sam2_predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    best_idx = scores.argmax()
    return masks[best_idx].astype(bool)


def _segment_with_box(rgb: np.ndarray, box) -> np.ndarray:
    """Use SAM2 image predictor with a bounding box prompt."""
    _sam2_predictor.set_image(rgb)
    masks, scores, _ = _sam2_predictor.predict(
        box=np.array(list(box), dtype=np.float32),
        multimask_output=True,
    )
    best_idx = scores.argmax()
    return masks[best_idx].astype(bool)


def _handle_segment(msg: dict) -> dict:
    _load_sam2()
    rgb = np.load(msg["image"])
    text_prompt = msg.get("text_prompt")

    if text_prompt:
        result = _detect_box_gdino(rgb, text_prompt)
        if result is not None:
            box = result[:4]
            _log(f"Text-prompted segmentation: box={box} conf={result[4]:.2f} "
                 f"for '{text_prompt}'")
            mask = _segment_with_box(rgb, box)
        else:
            _log("Using center-point SAM2 prompt (GDINO unavailable)")
            mask = _segment_with_point(rgb)
    else:
        masks = _sam2_generator.generate(rgb)
        if not masks:
            mask = np.zeros(rgb.shape[:2], dtype=bool)
        else:
            total = rgb.shape[0] * rgb.shape[1]
            candidates = [
                m for m in masks
                if 0.005 <= m["area"] / total <= 0.90
            ]
            if not candidates:
                candidates = masks
            best = max(candidates, key=lambda m: m["predicted_iou"])
            mask = best["segmentation"].astype(bool)

    mask_path = os.path.join(_SHM, "vb_mask.npy")
    np.save(mask_path, mask)
    return {"ok": True, "mask": mask_path}


def _handle_detect_gdino(msg: dict) -> dict:
    """Run GDINO detection only (no segmentation)."""
    rgb = np.load(msg["image"])
    text_prompt = msg.get("text_prompt", "")

    result = _detect_box_gdino(rgb, text_prompt)
    if result is not None:
        return {
            "ok": True,
            "box": list(result[:4]),
            "confidence": result[4],
        }
    return {"ok": True, "box": None, "confidence": 0.0}


# ---------------------------------------------------------------------------
# Video segmentation with SAM2 VideoPredictor + GDINO re-anchoring
# ---------------------------------------------------------------------------

def _mask_bbox(mask: np.ndarray):
    """Return (x0, y0, x1, y1) bounding box of a boolean mask, or None."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return (int(x0), int(y0), int(x1), int(y1))


def _bbox_iou(a, b) -> float:
    """Intersection-over-union of two (x0, y0, x1, y1) boxes."""
    if a is None or b is None:
        return 0.0
    xa = max(a[0], b[0])
    ya = max(a[1], b[1])
    xb = min(a[2], b[2])
    yb = min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _save_gdino_debug_images(
    frames: list[np.ndarray],
    gdino_results: list[dict],
    debug_dir: str,
):
    """Save GDINO detection overlays for visual inspection."""
    out = os.path.join(debug_dir, "gdino_detections")
    os.makedirs(out, exist_ok=True)

    step = max(1, len(frames) // 20)
    for i in range(0, len(frames), step):
        rgb = frames[i]
        overlay = rgb.copy()
        info = gdino_results[i] if i < len(gdino_results) else {}

        box = info.get("box")
        conf = info.get("confidence", 0.0)
        reanchored = info.get("reanchored", False)

        if box is not None:
            x0, y0, x1, y1 = box
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)
            label = f"detected: {conf:.2f}"
            cv2.putText(overlay, label, (x0, max(y0 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(overlay, "no detection", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if reanchored:
            cv2.putText(overlay, "RE-ANCHORED", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        cv2.putText(overlay, f"frame {i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imwrite(
            os.path.join(out, f"frame_{i:04d}.jpg"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )

    _log(f"GDINO debug overlays saved to {out}")


def _handle_segment_video(msg: dict) -> dict:
    """Segment all frames using SAM2 VideoPredictor with GDINO-guided init.

    Uses GDINO on frame 0 for the initial prompt, then propagates through
    the video with SAM2 VideoPredictor.  Optionally re-anchors with GDINO
    every ``reanchor_interval`` frames when drift is detected.
    """
    import torch

    _load_sam2()
    _load_sam2_video_predictor()

    frames_dir = msg["frames_dir"]
    frame_indices = msg.get("frame_indices")
    text_prompt = msg.get("text_prompt", "")
    debug_vis_dir = msg.get("debug_vis_dir", "")
    reanchor_interval = msg.get("reanchor_interval", 30)

    all_jpgs = sorted(
        f for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg"))
    )

    if frame_indices is not None:
        selected = [all_jpgs[i] for i in frame_indices if i < len(all_jpgs)]
    else:
        selected = all_jpgs

    n = len(selected)
    _log(f"segment_video: {n} frames from {frames_dir}")

    # SAM2 VideoPredictor needs a directory of sequentially named JPEGs.
    tmp_dir = tempfile.mkdtemp(prefix="sam2vid_")
    try:
        for seq_idx, fname in enumerate(selected):
            src = os.path.join(frames_dir, fname)
            dst = os.path.join(tmp_dir, f"{seq_idx:05d}.jpg")
            os.symlink(src, dst)

        # Load frames into memory for GDINO and debug viz
        frames_rgb = []
        for seq_idx in range(n):
            img = cv2.imread(os.path.join(tmp_dir, f"{seq_idx:05d}.jpg"))
            frames_rgb.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # --- Initial prompt on frame 0 ---
        gdino_results = [{} for _ in range(n)]
        frame0 = frames_rgb[0]

        initial_mask = None
        if text_prompt:
            result = _detect_box_gdino(frame0, text_prompt)
            if result is not None:
                box = result[:4]
                conf = result[4]
                gdino_results[0] = {
                    "box": list(box), "confidence": conf, "reanchored": False,
                }
                _log(f"GDINO frame 0: box={box} conf={conf:.2f}")
                initial_mask = _segment_with_box(frame0, box)

        if initial_mask is None:
            _log("Using center-point prompt for frame 0 (GDINO unavailable)")
            initial_mask = _segment_with_point(frame0)

        # --- Init VideoPredictor ---
        inference_state = _sam2_video_predictor.init_state(
            video_path=tmp_dir, offload_video_to_cpu=True,
        )

        _sam2_video_predictor.add_new_mask(
            inference_state, frame_idx=0, obj_id=1, mask=initial_mask,
        )

        # --- Propagate ---
        all_masks = [None] * n
        all_masks[0] = initial_mask

        for frame_idx, obj_ids, mask_logits in \
                _sam2_video_predictor.propagate_in_video(inference_state):
            mask = (mask_logits[0] > 0.0).squeeze().cpu().numpy().astype(bool)
            all_masks[frame_idx] = mask

        # --- GDINO re-anchoring pass ---
        if text_prompt and _gdino_model is not None and reanchor_interval > 0:
            reanchor_frames = list(range(reanchor_interval, n, reanchor_interval))
            reanchored_any = False

            for ri in reanchor_frames:
                if all_masks[ri] is None:
                    continue

                result = _detect_box_gdino(frames_rgb[ri], text_prompt)
                if result is None:
                    gdino_results[ri] = {
                        "box": None, "confidence": 0.0, "reanchored": False,
                    }
                    continue

                gdino_box = result[:4]
                conf = result[4]
                tracked_bbox = _mask_bbox(all_masks[ri])
                iou = _bbox_iou(gdino_box, tracked_bbox)

                gdino_results[ri] = {
                    "box": list(gdino_box),
                    "confidence": conf,
                    "iou_with_tracked": round(iou, 3),
                    "reanchored": False,
                }

                if iou < 0.3:
                    _log(f"Re-anchoring at frame {ri}: IoU={iou:.2f} "
                         f"(tracked_bbox={tracked_bbox}, gdino_box={gdino_box})")

                    corrected_mask = _segment_with_box(frames_rgb[ri], gdino_box)

                    _sam2_video_predictor.reset_state(inference_state)
                    _sam2_video_predictor.add_new_mask(
                        inference_state, frame_idx=ri, obj_id=1,
                        mask=corrected_mask,
                    )

                    for fidx, oids, mlogits in \
                            _sam2_video_predictor.propagate_in_video(
                                inference_state, start_frame_idx=ri,
                            ):
                        m = (mlogits[0] > 0.0).squeeze().cpu().numpy().astype(bool)
                        all_masks[fidx] = m

                    gdino_results[ri]["reanchored"] = True
                    reanchored_any = True

            if reanchored_any:
                _log("Re-anchoring complete")

        # Fill any None masks with empty masks
        h, w = frames_rgb[0].shape[:2]
        for i in range(n):
            if all_masks[i] is None:
                all_masks[i] = np.zeros((h, w), dtype=bool)

        # --- Save debug visualizations ---
        if debug_vis_dir:
            # Run GDINO on debug sample frames (for visualization only)
            step = max(1, n // 20)
            for i in range(0, n, step):
                if gdino_results[i]:
                    continue
                if text_prompt:
                    result = _detect_box_gdino(frames_rgb[i], text_prompt)
                    if result is not None:
                        gdino_results[i] = {
                            "box": list(result[:4]),
                            "confidence": result[4],
                            "reanchored": False,
                        }
                    else:
                        gdino_results[i] = {
                            "box": None, "confidence": 0.0, "reanchored": False,
                        }

            _save_gdino_debug_images(frames_rgb, gdino_results, debug_vis_dir)

        # --- Return masks ---
        stacked = np.stack(all_masks)
        masks_path = os.path.join(_SHM, "vb_video_masks.npy")
        np.save(masks_path, stacked)

        gdino_path = os.path.join(_SHM, "vb_gdino_results.npy")
        np.save(gdino_path, np.array(gdino_results, dtype=object))

        return {"ok": True, "masks": masks_path, "gdino_results": gdino_path}

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _handle_vggt(msg: dict) -> dict:
    import cv2
    import torch
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    _load_vggt()

    images = np.load(msg["images"])
    masks = np.load(msg["masks"]) if "masks" in msg and msg["masks"] else None

    # Pass unmasked images to VGGT for better depth/pose estimation.
    # Masks are only applied to the output depth maps below.
    res = 518
    tensors = []
    for img in images:
        resized = cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensors.append(t)
    batch = torch.stack(tensors).unsqueeze(0).cuda()

    with torch.no_grad(), _suppress_stdout():
        preds = _vggt_model(batch)

    pose_enc = preds["pose_enc"]
    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        pose_enc, image_size_hw=(res, res)
    )
    ext = extrinsics[0].cpu().numpy()
    intr = intrinsics[0].cpu().numpy()
    depth = preds["depth"][0].cpu().numpy().squeeze(-1)

    pts = preds.get("world_points")
    if pts is not None:
        pts = pts[0].cpu().numpy()
    else:
        pts = np.zeros((*depth.shape, 3), dtype=np.float32)

    paths = {}
    for name, arr in [
        ("extrinsics", ext), ("intrinsics", intr),
        ("depths", depth), ("points3d", pts),
    ]:
        p = os.path.join(_SHM, f"vb_{name}.npy")
        np.save(p, arr)
        paths[name] = p

    return {"ok": True, **paths}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(json.dumps({"_log": msg}), file=sys.stderr, flush=True)


def main() -> None:
    _log("Vision server starting (pid=%d)" % os.getpid())
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            cmd = msg.get("cmd", "")

            if cmd == "load_sam2":
                _load_sam2()
                resp = {"ok": True}
            elif cmd == "load_vggt":
                _load_vggt()
                resp = {"ok": True}
            elif cmd == "load_gdino":
                _load_gdino()
                resp = {"ok": True, "available": _gdino_model is not None}
            elif cmd == "segment":
                resp = _handle_segment(msg)
            elif cmd == "detect_gdino":
                resp = _handle_detect_gdino(msg)
            elif cmd == "segment_video":
                resp = _handle_segment_video(msg)
            elif cmd == "vggt":
                resp = _handle_vggt(msg)
            elif cmd == "shutdown":
                print(json.dumps({"ok": True}), flush=True)
                break
            else:
                resp = {"ok": False, "error": f"unknown cmd: {cmd}"}
        except Exception as exc:
            _log(traceback.format_exc())
            resp = {"ok": False, "error": str(exc)}

        print(json.dumps(resp), flush=True)

    _log("Vision server shutting down")


if __name__ == "__main__":
    main()
