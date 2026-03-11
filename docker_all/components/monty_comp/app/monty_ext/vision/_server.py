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

  → {"cmd": "vggt", "images": "/dev/shm/vb_batch.npy",
     "masks": "/dev/shm/vb_masks.npy"}   (masks optional)
  ← {"ok": true,
      "extrinsics": "/dev/shm/vb_ext.npy",
      "intrinsics": "/dev/shm/vb_intr.npy",
      "depths": "/dev/shm/vb_depths.npy",
      "points3d": "/dev/shm/vb_pts.npy"}

  → {"cmd": "shutdown"}
  ← {"ok": true}
"""

from __future__ import annotations

import json
import os
import sys
import traceback

import numpy as np

_SHM = "/dev/shm"

# ---------------------------------------------------------------------------
# Lazy-loaded models
# ---------------------------------------------------------------------------

_sam2_model = None
_sam2_generator = None
_vggt_model = None


def _load_sam2() -> None:
    global _sam2_model, _sam2_generator
    if _sam2_model is not None:
        return

    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

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
                _log(f"Download incomplete: {os.path.getsize(ckpt_file)}/{expected_size}")
            except Exception as e:
                _log(f"Download failed: {e}")
                if attempt == 2:
                    raise

    _log("Loading SAM2 ...")
    _sam2_model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        ckpt_file,
        device="cuda",
    )
    _sam2_generator = SAM2AutomaticMaskGenerator(
        model=_sam2_model,
        points_per_side=32,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.85,
    )
    _log("SAM2 loaded")


def _load_vggt() -> None:
    global _vggt_model
    if _vggt_model is not None:
        return

    import torch
    os.environ.setdefault("TORCH_HOME", os.environ.get("MODEL_CACHE_DIR", "/models"))

    from vggt.models.vggt import VGGT

    _log("Loading VGGT ...")
    _vggt_model = VGGT.from_pretrained("facebook/VGGT-1B")
    _vggt_model.eval().cuda()
    _log("VGGT loaded")


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _handle_segment(msg: dict) -> dict:
    _load_sam2()
    rgb = np.load(msg["image"])

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


def _handle_vggt(msg: dict) -> dict:
    import cv2
    import torch
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    _load_vggt()

    images = np.load(msg["images"])
    masks = np.load(msg["masks"]) if "masks" in msg and msg["masks"] else None

    if masks is not None:
        images = (images * masks[..., None]).astype(np.uint8)

    res = 518
    tensors = []
    for img in images:
        resized = cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensors.append(t)
    batch = torch.stack(tensors).unsqueeze(0).cuda()

    with torch.no_grad():
        preds = _vggt_model(batch)

    pose_enc = preds["pose_enc"]
    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        pose_enc, image_size_hw=(res, res)
    )
    ext = extrinsics[0].cpu().numpy()
    intr = intrinsics[0].cpu().numpy()
    depth = preds["depth"][0].cpu().numpy()

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
            elif cmd == "segment":
                resp = _handle_segment(msg)
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
