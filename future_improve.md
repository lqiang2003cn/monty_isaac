# Future Improvements: Monty + Real-World Object Recognition

Context: We have a turntable scanning pipeline (two Orbbec Gemini 2 RGBD cameras,
spinning turntable, 4090 GPU) feeding object data to the Thousand Brains Project's
Monty framework. The improvements below go beyond better sensors — they change how
Monty itself processes, learns, and recognizes.

---

## 1. Inject DINOv2 Semantic Features into Monty's Learning Module

### What Monty does now

Monty's `CameraSM` sensor module extracts a fixed set of geometric and color features
from each RGBD patch:

- `pose_vectors` (surface normal + 2 principal curvature directions)
- `principal_curvatures` / `principal_curvatures_log`
- `gaussian_curvature`, `mean_curvature`
- `hsv` (color)
- `rgba`
- `on_object` (binary)
- `min_depth`, `mean_depth`

These are hand-crafted, geometry-first features. They work well for distinguishing
objects with different shapes (a mug vs a banana) but struggle with:

- Objects that have the same shape but different visual appearance (two mugs with
  different logos)
- Objects whose distinguishing features are semantic (a bottle of hot sauce vs a
  bottle of cocktail bitters — same shape, different label)
- Textured or patterned surfaces where curvature is flat but appearance varies

### What the improvement would change

Add DINOv2 feature vectors as a new feature type in Monty's CMP (Cortical Messaging
Protocol). At each sensor patch location, in addition to curvature and color, attach
a 384-dim (ViT-S/14) or 1024-dim (ViT-L/14) DINOv2 feature vector.

DINOv2 features are:

- Semantically meaningful: "handle" regions on different mugs produce similar features
- View-invariant: the same surface patch produces similar features from different angles
- Texture-aware: captures visual patterns that curvature cannot
- Pre-trained on 142M images — no task-specific training needed

### How to implement

**Step 1: Create a new sensor module class**

In `tbp.monty/src/tbp/monty/frameworks/models/sensor_modules.py`, subclass `CameraSM`:

```python
class DINOv2CameraSM(CameraSM):
    """CameraSM that also extracts DINOv2 patch features."""

    def __init__(self, dino_model="dinov2_vits14", **kwargs):
        super().__init__(**kwargs)
        self.dino = torch.hub.load("facebookresearch/dinov2", dino_model)
        self.dino.eval().cuda()

    def _extract_features(self, obs):
        features = super()._extract_features(obs)
        # Extract DINOv2 feature for the center patch
        rgb_patch = obs["rgba"][:, :, :3]  # H x W x 3
        dino_feat = self._get_dino_feature(rgb_patch)
        features["dino_v2"] = dino_feat  # 384-dim vector
        return features
```

**Step 2: Register the feature in the learning module**

The `EvidenceGraphLM` stores features at each graph node. Add `"dino_v2"` to the
feature list and define a tolerance (cosine distance threshold) for matching:

```yaml
tolerances:
  patch:
    principal_curvatures_log: [1, 1]
    pose_vectors: [45, 45, 45]
    dino_v2: 0.3  # cosine distance threshold
feature_weights:
  patch:
    pose_vectors: [1, 1, 1]
    dino_v2: 2.0  # weight DINOv2 heavily for shape-similar objects
```

**Step 3: Define a distance metric**

Monty's evidence matching uses per-feature distance functions. For DINOv2 vectors,
cosine distance is the right metric:

```python
def dino_v2_distance(feat_a, feat_b):
    return 1.0 - np.dot(feat_a, feat_b) / (
        np.linalg.norm(feat_a) * np.linalg.norm(feat_b)
    )
```

**Step 4: Dimensionality reduction (optional, for memory)**

384 dims per graph node is large. Options:

- PCA to 32-64 dims (fit PCA on first few objects, apply to all)
- Random projection (no fitting needed, preserves cosine distances)
- Store only if the object set contains shape-similar items

### Expected impact

- **Shape-similar objects**: large improvement. Currently Monty confuses mugs that
  differ only by label/color pattern. DINOv2 features would disambiguate them.
- **Shape-distinct objects**: minimal improvement. Curvature alone already separates
  a banana from a mug.
- **Training time**: ~10-20% slower per step (DINOv2 inference on a 4090 is ~1ms per
  patch, but adds up over hundreds of steps per episode).
- **Memory**: ~50-100% more per graph node (384 floats vs ~20 floats for current
  features). Mitigated by dimensionality reduction.

### Difficulty: Medium

Requires modifying Monty source code (sensor module, learning module config, distance
functions). No architectural changes. DINOv2 itself is a frozen, pre-trained model.

### References

- DINOv2 paper: Oquab et al., "DINOv2: Learning Robust Visual Features without
  Supervision" (2024)
- DINOv2 repo: https://github.com/facebookresearch/dinov2
- Monty sensor modules: `src/tbp/monty/frameworks/models/sensor_modules.py`
- Monty evidence LM: `src/tbp/monty/frameworks/models/evidence_matching/`

---

## 2. Direct Graph Construction from Dense RGBD (Bypass Sensorimotor Exploration)

### What Monty does now

Monty builds object models through **sequential sensorimotor exploration**: a small
virtual patch moves across the object surface one step at a time, extracting features
at each location. For a single object in one orientation, this takes ~500 exploratory
steps. Across 14 orientations for training, that's ~7000 steps per object.

This is biologically inspired (how a fingertip explores a surface) but extremely
slow for bootstrapping a model library from RGBD scans. Each step involves:

1. Motor policy decides next action
2. Environment applies action, returns new RGBD observation
3. Sensor module extracts features + pose from small patch
4. Learning module adds point to graph if sufficiently novel

### What the improvement would change

Given a full RGBD frame with SAM2 mask, directly compute ALL graph points in a
single pass — no sequential exploration needed. One frame → hundreds of graph points.
A full turntable rotation (72 frames) → a complete, dense object model in seconds
instead of minutes.

The key insight: Monty's graph is just a sparse point cloud where each point has a
3D location + orientation (surface normal, curvature directions) + features (color,
curvature values). All of these can be computed from a single RGBD frame for every
visible surface point simultaneously.

### How to implement

**Step 1: Dense feature extraction from a single RGBD frame**

```python
import open3d as o3d
import numpy as np

def extract_graph_points_from_frame(color, depth, mask, intrinsic):
    """Extract Monty-compatible graph points from a single masked RGBD frame."""
    # 1. Create masked point cloud
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    points = np.asarray(pcd.points)[mask.flatten()]
    colors = np.asarray(pcd.colors)[mask.flatten()]

    # 2. Estimate normals
    pcd_masked = o3d.geometry.PointCloud()
    pcd_masked.points = o3d.utility.Vector3dVector(points)
    pcd_masked.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    )

    # 3. Estimate principal curvatures (per-point)
    normals = np.asarray(pcd_masked.normals)
    curvatures = estimate_principal_curvatures(points, normals)  # custom function

    # 4. Subsample to Monty's graph density (~1cm spacing)
    pcd_down = pcd_masked.voxel_down_sample(voxel_size=0.01)

    # 5. Extract HSV from RGB colors
    hsv = rgb_to_hsv(colors)

    # 6. Package as Monty graph nodes
    graph_points = []
    for i in range(len(pcd_down.points)):
        graph_points.append({
            "location": pcd_down.points[i],
            "pose_vectors": np.stack([normals[i], curv_dir1[i], curv_dir2[i]]),
            "principal_curvatures_log": np.log1p(curvatures[i]),
            "hsv": hsv[i],
        })
    return graph_points
```

**Step 2: Merge across turntable views**

```python
full_graph = []
for angle, (color, depth, mask) in turntable_frames:
    T = rotation_matrix_y(angle)
    points = extract_graph_points_from_frame(color, depth, mask, intrinsic)
    for p in points:
        p["location"] = T @ p["location"]  # transform to object frame
        p["pose_vectors"] = T @ p["pose_vectors"]
    full_graph.extend(points)

# Deduplicate: remove points within 1cm of existing points with similar features
final_graph = deduplicate_graph(full_graph, dist_thresh=0.01, feat_thresh=0.1)
```

**Step 3: Save in Monty's graph format**

Monty's pretrained models are stored as dictionaries of `GraphLM` state. The graph
is essentially a dict with:

- `locations`: Nx3 array of 3D positions
- `features`: dict of feature arrays (curvatures, colors, etc.)
- `pose_vectors`: Nx3x3 array of orientation frames

Save the directly-constructed graph in this format and load it as a pretrained model.

### Expected impact

- **Speed**: 100-1000x faster for model building. Current: ~5 min per object
  (14 orientations × 500 steps × habitat rendering). Direct: ~5 seconds per object
  (72 RGBD frames × dense extraction + merge).
- **Coverage**: potentially better. The sequential explorer may miss concavities or
  hard-to-reach surface patches. Dense extraction from multiple views captures
  everything visible.
- **Accuracy**: comparable or slightly worse for individual point features (Monty's
  SM uses a zoomed-in 10x patch for precise curvature estimation; dense extraction
  from a full frame has lower resolution per point). Mitigated by using higher-res
  cameras or multi-scale extraction.
- **Biological plausibility**: zero. This is a pure engineering optimization that
  trades the sensorimotor learning philosophy for speed. Fine for bootstrapping a
  model library, not for research on sensorimotor learning itself.

### Difficulty: Medium-Hard

Requires understanding Monty's internal graph format, replicating the exact feature
extraction pipeline (especially curvature estimation, which Monty does in a specific
way), and ensuring the directly-built graph is compatible with Monty's evidence
matching during inference.

Key risk: Monty's curvature estimation from a zoomed-in patch may produce different
values than estimation from a full-frame point cloud. This could cause mismatch
between training (direct) and inference (sensorimotor). Mitigation: use the same
patch-based estimation at each sampled point.

### References

- Monty graph format: `src/tbp/monty/frameworks/models/graph_matching.py`
- Monty pretrained model loading: `frameworks/utils/logging_utils.py::load_stats()`
- Open3D normal estimation: https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html

---

## 3. Vision-Foundation-Model-Driven Exploration Policy

### What Monty does now

Monty's motor policy during inference decides where to move the sensor patch next.
The current policies are:

- **Surface agent (curvature-informed)**: follows principal curvature directions on
  the object surface, like tracing ridges with a fingertip. Effective but slow —
  it's a local search with no global planning.
- **Distant agent (random walk)**: tilts the camera randomly. Very inefficient.
- **Hypothesis-testing policy**: jumps to locations that would disambiguate between
  remaining object hypotheses. Smarter, but limited by what the learning module knows.

None of these policies use visual understanding of the scene to decide where to look.
They operate on geometric features only.

### What the improvement would change

Use a vision foundation model to compute an **information-gain map** over the visible
object surface, guiding the sensor to the most discriminative regions first.

Example: Monty is trying to distinguish between two similar mugs. The curvature is
identical everywhere. A DINOv2-based policy would recognize that the logo region has
high feature variance across object hypotheses and direct the sensor there first,
instead of randomly wandering the handle for 200 steps.

### How to implement

**Approach A: DINOv2 feature-space information gain**

```python
def compute_next_best_view(current_rgb, remaining_hypotheses, current_graph_model):
    """Suggest the next sensor location that maximizes information gain."""
    # 1. Extract dense DINOv2 features from current full-frame view
    dino_features = extract_dense_dino(current_rgb)  # H x W x 384

    # 2. For each remaining hypothesis (object + pose), predict what
    #    DINOv2 features we'd expect at each visible location
    expected_features = {}
    for hyp in remaining_hypotheses:
        expected_features[hyp] = project_graph_dino_features(
            current_graph_model[hyp.object_id],
            hyp.pose,
            camera_intrinsic,
        )

    # 3. Compute per-pixel information gain: how much would observing
    #    this location reduce hypothesis entropy?
    info_gain = np.zeros((H, W))
    for px in visible_object_pixels:
        feat_observed = dino_features[px]
        likelihoods = []
        for hyp in remaining_hypotheses:
            expected = expected_features[hyp][px]
            likelihood = cosine_similarity(feat_observed, expected)
            likelihoods.append(likelihood)
        info_gain[px] = entropy(likelihoods)  # high entropy = high info gain

    # 4. Move sensor to highest info-gain location
    best_pixel = np.unravel_index(np.argmax(info_gain), info_gain.shape)
    return pixel_to_3d_location(best_pixel, depth_map, intrinsic)
```

**Approach B: SAM2 part-aware exploration**

SAM2 can segment an object into semantic parts (handle, body, rim, base) using
its "segment everything" mode. Instead of random exploration:

1. Segment the object into parts
2. Sample one observation from each part
3. This ensures coverage of all semantic regions in minimum steps

This is simpler than Approach A and doesn't require hypothesis tracking, making it
suitable for the learning phase (not just inference).

**Approach C: Active learning with uncertainty sampling**

During inference, track which regions of the object have been observed vs not.
Use DINOv2 feature uncertainty (variance across remaining hypotheses) to identify
the most uncertain unobserved region and direct the sensor there.

### Expected impact

- **Inference speed**: 2-5x fewer steps to converge on correct object identity,
  especially for shape-similar objects. Currently Monty takes 50-200 steps for
  confident recognition; an informed policy could achieve the same in 20-50 steps.
- **Accuracy on hard cases**: significant improvement for objects that are only
  distinguishable in specific regions (labels, textures, small geometric details).
- **Training speed**: modest improvement if using SAM2 part-aware exploration
  (ensures coverage without redundant exploration of uniform surfaces).

### Difficulty: Hard

This requires modifying Monty's motor system (`MotorSystem`, `MotorPolicy` classes),
integrating a foundation model into the inference loop (adding latency per step),
and designing the information-gain computation to be compatible with Monty's
hypothesis space representation.

The hardest part is the interface between the foundation model's feature space and
Monty's hypothesis space — translating "which hypotheses would this observation
disambiguate?" into "where should I look next?" requires projecting graph models
into the camera frame for each remaining hypothesis.

### References

- Monty motor system: `src/tbp/monty/frameworks/models/motor_policies.py`
- Monty hypothesis testing: `src/tbp/monty/frameworks/models/evidence_matching/`
- Active perception literature: Bajcsy et al., "Revisiting Active Perception" (2018)
- DINOv2 dense features: `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")`

---

## 4. Multi-Modal Fusion: Combine RGBD + Tactile (X3plus Gripper)

### What Monty does now

Monty supports multiple sensor modules connected to different learning modules that
communicate via voting. In simulation, this means multiple camera patches at different
zoom levels. In the real world (LEGO robot), it's been one RGBD camera.

### What the improvement would change

The X3plus arm already exists in this project. If a tactile sensor (e.g., a GelSight
Mini or DIGIT sensor) were mounted on or near the gripper, Monty could combine:

- **Vision** (RGBD cameras): shape, color, texture from a distance
- **Touch** (tactile sensor): precise local geometry, compliance, surface roughness

This is exactly what Monty was designed for — it's inspired by how the brain fuses
vision and touch through cortical columns with the same algorithm.

### How to implement

1. Mount a tactile sensor on the X3plus gripper
2. Create a `TactileSM` sensor module that extracts features from tactile images
   (contact geometry, surface normal, friction/texture)
3. Create a `TactileEnvironmentInterface` that drives the arm to touch the object
   at locations suggested by the visual learning module
4. Connect both visual and tactile sensor modules to separate learning modules
5. Enable voting between the visual LM and tactile LM via `lm_to_lm_vote_matrix`

### Expected impact

- Disambiguates objects that look identical but feel different (hard vs soft, smooth
  vs rough, heavy vs light)
- More robust recognition — two independent sensory channels voting together
- Directly leverages the X3plus arm hardware already in this project

### Difficulty: Hard

Requires tactile sensor hardware, a new sensor module, a new environment interface
for arm-driven exploration, and careful coordination between visual and tactile
exploration policies.

### References

- Monty multi-LM voting: `src/tbp/monty/frameworks/models/evidence_matching/`
- Monty 5-LM config example: `src/tbp/monty/conf/experiment/config/monty/five_lm.yaml`
- GelSight Mini: https://www.gelsight.com/gelsightmini/
- DIGIT tactile sensor: https://digit.ml/

---

## Priority Ranking

| # | Improvement | Impact | Effort | Prerequisite |
|---|---|---|---|---|
| 1 | DINOv2 features in Monty | High for similar objects | Medium | Basic pipeline working |
| 2 | Direct graph construction | Huge for speed | Medium-Hard | Understanding Monty's graph format |
| 3 | VFM-driven exploration policy | High for inference speed | Hard | DINOv2 features (#1) |
| 4 | Multi-modal fusion (vision + touch) | High for robustness | Hard | Tactile sensor hardware + X3plus integration |

Recommended order: Start with #2 (speed up model building for rapid iteration),
then #1 (improve feature quality), then #3 (improve inference efficiency), then #4
(add a second modality once the visual pipeline is mature).
