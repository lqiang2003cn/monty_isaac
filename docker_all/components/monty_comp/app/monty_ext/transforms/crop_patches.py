"""Transform that crops a single view_finder image into N non-overlapping patches.

Runs AFTER DepthTo3DLocations so that ``semantic_3d`` and ``sensor_frame_data``
already contain correct world-frame 3D coordinates for every pixel.  The
transform slices these arrays (along with ``rgba`` and ``depth``) into per-patch
sub-arrays and injects them back into the observation dict so that downstream
CameraSM instances can process each patch independently.

Grid layout example (default 10x10)::

    Full image: 1280 (W) x 800 (H)
    10 cols of 128px each, 10 rows of 80px each -> 100 patches, no overlap.
"""

from __future__ import annotations

import numpy as np

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.abstract_monty_classes import Observations


class CropPatchesFromViewFinder:
    """Crop view_finder observation into a grid of non-overlapping patches.

    Each patch is injected as ``observations[agent_id]["patch_{i}"]`` with keys
    ``rgba``, ``depth``, ``semantic_3d``, ``sensor_frame_data``, and
    ``world_camera``.

    Must run after :class:`DepthTo3DLocations` (which populates ``semantic_3d``,
    ``sensor_frame_data``, and ``world_camera`` on the view_finder).
    """

    def __init__(
        self,
        agent_id: AgentID,
        n_rows: int = 10,
        n_cols: int = 10,
        patch_h: int = 80,
        patch_w: int = 128,
    ):
        self.agent_id = agent_id
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.patch_h = patch_h
        self.patch_w = patch_w

    def __call__(self, observations: Observations, _ctx) -> Observations:
        vf = observations[self.agent_id]["view_finder"]

        H = self.n_rows * self.patch_h
        W = self.n_cols * self.patch_w
        ph, pw = self.patch_h, self.patch_w

        rgba = vf["rgba"]
        depth = vf["depth"]
        world_camera = vf["world_camera"]

        sem_3d_full = vf["semantic_3d"].reshape(H, W, -1)
        sf_data_full = vf["sensor_frame_data"].reshape(H, W, -1)

        for i in range(self.n_rows * self.n_cols):
            r0 = (i // self.n_cols) * ph
            c0 = (i % self.n_cols) * pw

            observations[self.agent_id][f"patch_{i}"] = {
                "rgba": rgba[r0:r0 + ph, c0:c0 + pw],
                "depth": depth[r0:r0 + ph, c0:c0 + pw],
                "semantic_3d": np.ascontiguousarray(
                    sem_3d_full[r0:r0 + ph, c0:c0 + pw].reshape(-1, sem_3d_full.shape[2])
                ),
                "sensor_frame_data": np.ascontiguousarray(
                    sf_data_full[r0:r0 + ph, c0:c0 + pw].reshape(-1, sf_data_full.shape[2])
                ),
                "world_camera": world_camera,
            }

        return observations
