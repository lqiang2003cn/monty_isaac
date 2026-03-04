#!/usr/bin/env python3
"""
Compute the full reachability map of the X3plus 5-DOF arm:
  For each reachable (x, y, z), what range of gripper pitch angles is achievable?

The arm structure:
  Joint 1 (yaw)  →  Joint 2 (pitch)  →  Joint 3 (pitch)  →  Joint 4 (pitch)  →  Joint 5 (roll)

Since the arm is rotationally symmetric about joint 1, we work in the arm's
vertical plane (radial distance r vs height z) and compute the reachable
(r, z, pitch) volume.  Joint 1 just sweeps this cross-section through ±90°.

Pitch = q2 + q3 + q4  (angle of the last planar link relative to horizontal).
  pitch =  0     → gripper pointing horizontally forward
  pitch = -π/2   → gripper pointing straight down
  pitch = +π/2   → gripper pointing straight up
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

# --- Link lengths from the URDF ---
L1 = 0.0829    # arm_link2 → arm_link3  (joint 3 offset along planar Y)
L2 = 0.0829    # arm_link3 → arm_link4  (joint 4 offset along planar Y)
L3 = 0.17455   # arm_link4 → wrist      (joint 5 offset along planar Y)

BASE_XY = 0.09825                      # joint 1 X-offset from base_link origin
BASE_Z  = 0.076 + 0.102 + 0.0405      # base_footprint + joint1_z + joint2_z = 0.2185 m

Q2_LIM = (-np.pi/2, np.pi/2)
Q3_LIM = (-np.pi/2, np.pi/2)
Q4_LIM = (-np.pi/2, np.pi/2)

# --- Analytical forward kinematics in the planar sub-chain ---
# Given (q2, q3, q4), compute the wrist point (r, z) relative to joint-2 origin
# and the gripper pitch angle.

N = 80  # samples per joint for the planar sub-chain

q2_vals = np.linspace(*Q2_LIM, N)
q3_vals = np.linspace(*Q3_LIM, N)
q4_vals = np.linspace(*Q4_LIM, N)

print(f"Sampling {N**3:,} planar configurations …")

# Vectorized: create meshgrid for (q2, q3, q4)
Q2, Q3, Q4 = np.meshgrid(q2_vals, q3_vals, q4_vals, indexing='ij')
Q2f = Q2.ravel()
Q3f = Q3.ravel()
Q4f = Q4.ravel()

# Planar FK: joint 2 is at the origin of the planar frame.
# After joint 2 rotation by q2, link1 extends along angle q2.
# After joint 3 rotation by q3, link2 extends along angle q2+q3.
# After joint 4 rotation by q4, link3 (to wrist) extends along angle q2+q3+q4.
#
# Endpoint of link1: (L1*cos(q2), L1*sin(q2))
# Endpoint of link2: above + (L2*cos(q2+q3), L2*sin(q2+q3))
# Wrist point:       above + (L3*cos(q2+q3+q4), L3*sin(q2+q3+q4))

phi12 = Q2f + Q3f
phi123 = phi12 + Q4f

r_local = L1*np.cos(Q2f) + L2*np.cos(phi12) + L3*np.cos(phi123)
z_local = L1*np.sin(Q2f) + L2*np.sin(phi12) + L3*np.sin(phi123)

# Convert to base_footprint frame
r_global = r_local + BASE_XY   # radial distance from base Z-axis
z_global = z_local + BASE_Z    # height from ground

pitch = phi123  # gripper pitch angle

print(f"Computed {len(r_global):,} points.")
print(f"  r range: [{r_global.min():.4f}, {r_global.max():.4f}] m")
print(f"  z range: [{z_global.min():.4f}, {z_global.max():.4f}] m")
print(f"  pitch range: [{np.degrees(pitch.min()):.1f}°, {np.degrees(pitch.max()):.1f}°]")

# ── Figure 1: Side cross-section colored by achievable pitch range ──

# Bin the (r, z) plane and find the min/max pitch at each bin
r_bins = np.linspace(r_global.min() - 0.005, r_global.max() + 0.005, 120)
z_bins = np.linspace(z_global.min() - 0.005, z_global.max() + 0.005, 120)

pitch_min_grid = np.full((len(r_bins)-1, len(z_bins)-1), np.nan)
pitch_max_grid = np.full((len(r_bins)-1, len(z_bins)-1), np.nan)
pitch_range_grid = np.full((len(r_bins)-1, len(z_bins)-1), np.nan)

r_idx = np.digitize(r_global, r_bins) - 1
z_idx = np.digitize(z_global, z_bins) - 1

valid = (r_idx >= 0) & (r_idx < len(r_bins)-1) & (z_idx >= 0) & (z_idx < len(z_bins)-1)

print("Building reachability grid …")
for i in range(len(r_global)):
    if not valid[i]:
        continue
    ri, zi = r_idx[i], z_idx[i]
    p = pitch[i]
    if np.isnan(pitch_min_grid[ri, zi]):
        pitch_min_grid[ri, zi] = p
        pitch_max_grid[ri, zi] = p
    else:
        if p < pitch_min_grid[ri, zi]:
            pitch_min_grid[ri, zi] = p
        if p > pitch_max_grid[ri, zi]:
            pitch_max_grid[ri, zi] = p

pitch_range_grid = np.degrees(pitch_max_grid - pitch_min_grid)

r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
R_mesh, Z_mesh = np.meshgrid(r_centers, z_centers, indexing='ij')

# ── Figure: 4-panel reachability ──

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("X3plus 5-DOF Arm — Reachability with Orientation", fontsize=15, fontweight="bold")

# Panel 1: pitch range (dexterity)
ax = axes[0, 0]
im = ax.pcolormesh(R_mesh, Z_mesh, pitch_range_grid, cmap="YlOrRd", shading="auto")
cb = fig.colorbar(im, ax=ax)
cb.set_label("Achievable pitch range (degrees)")
ax.set_xlabel("Radial distance r (m)")
ax.set_ylabel("Height z (m)")
ax.set_title("Orientation dexterity: pitch range at each (r, z)")
ax.set_aspect("equal")
ax.grid(True, alpha=0.2)

# Panel 2: can the gripper point straight down (pitch ≈ -90°)?
can_down = (np.degrees(pitch_min_grid) <= -80)  # within 10° of straight down
ax = axes[0, 1]
ax.pcolormesh(R_mesh, Z_mesh, can_down.astype(float), cmap="RdYlGn", shading="auto", vmin=0, vmax=1)
ax.set_xlabel("Radial distance r (m)")
ax.set_ylabel("Height z (m)")
ax.set_title("Can gripper point DOWN (pitch ≤ -80°)?  Green = yes")
ax.set_aspect("equal")
ax.grid(True, alpha=0.2)

# Panel 3: can the gripper point horizontally (pitch ≈ 0°)?
can_horiz = (np.degrees(pitch_min_grid) <= 5) & (np.degrees(pitch_max_grid) >= -5)
ax = axes[1, 0]
ax.pcolormesh(R_mesh, Z_mesh, can_horiz.astype(float), cmap="RdYlGn", shading="auto", vmin=0, vmax=1)
ax.set_xlabel("Radial distance r (m)")
ax.set_ylabel("Height z (m)")
ax.set_title("Can gripper point HORIZONTAL (|pitch| ≤ 5°)?  Green = yes")
ax.set_aspect("equal")
ax.grid(True, alpha=0.2)

# Panel 4: scatter of (r, z) colored by pitch
sub = np.random.choice(len(r_global), min(80000, len(r_global)), replace=False)
ax = axes[1, 1]
sc = ax.scatter(r_global[sub], z_global[sub], c=np.degrees(pitch[sub]),
                cmap="coolwarm", s=0.3, alpha=0.15, vmin=-135, vmax=135)
cb2 = fig.colorbar(sc, ax=ax)
cb2.set_label("Gripper pitch (degrees)")
ax.set_xlabel("Radial distance r (m)")
ax.set_ylabel("Height z (m)")
ax.set_title("All reachable (r, z, pitch) — color = pitch angle")
ax.set_aspect("equal")
ax.grid(True, alpha=0.2)

plt.tight_layout()
out = "x3plus_reachability.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")

# ── Print summary statistics ──
reachable_bins = np.count_nonzero(~np.isnan(pitch_range_grid))
total_bins = pitch_range_grid.size
print(f"\nReachable area: {reachable_bins} / {total_bins} grid cells "
      f"({100*reachable_bins/total_bins:.1f}%)")

mask = ~np.isnan(pitch_range_grid)
print(f"Pitch range across workspace:")
print(f"  Average achievable pitch range: {np.nanmean(pitch_range_grid):.1f}°")
print(f"  Max achievable pitch range:     {np.nanmax(pitch_range_grid):.1f}°  (most dexterous point)")
print(f"  Min achievable pitch range:     {np.nanmin(pitch_range_grid[mask]):.1f}°  (least dexterous point)")

down_pct = 100 * np.count_nonzero(can_down) / reachable_bins
horiz_pct = 100 * np.count_nonzero(can_horiz) / reachable_bins
print(f"\nGripper-down  reachable area: {down_pct:.1f}% of workspace")
print(f"Gripper-horiz reachable area: {horiz_pct:.1f}% of workspace")
