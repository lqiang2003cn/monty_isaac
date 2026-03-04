#!/usr/bin/env python3
"""Compute and visualize the reachable workspace of the X3plus 5-DOF arm."""

import numpy as np
import matplotlib.pyplot as plt

# --- Homogeneous transformation helpers ---

def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]])

def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]])

def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]])

def tf(xyz, rpy=(0,0,0)):
    T = rot_z(rpy[2]) @ rot_y(rpy[1]) @ rot_x(rpy[0])
    T[:3, 3] = xyz
    return T

def joint_tf(xyz, rpy, axis, q):
    """Fixed origin transform followed by rotation q around axis."""
    T_origin = tf(xyz, rpy)
    ax = np.array(axis, dtype=float)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * (K @ K)
    T_rot = np.eye(4)
    T_rot[:3, :3] = R
    return T_origin @ T_rot

# --- X3plus kinematic chain (from URDF) ---

T_base = tf([0, 0, 0.076])  # base_footprint → base_link

CHAIN = [
    # (xyz, rpy, axis, lower, upper)  — from each joint tag in the URDF
    ([0.09825, 0, 0.102],    [0, 0, 0],         [0, 0, -1], -1.5708,  1.5708),   # arm_joint1 – yaw
    ([0, 0, 0.0405],         [-1.5708, 0, 0],   [0, 0, -1], -1.5708,  1.5708),   # arm_joint2 – pitch
    ([0, -0.0829, 0],        [0, 0, 0],         [0, 0, -1], -1.5708,  1.5708),   # arm_joint3 – pitch
    ([0, -0.0829, 0],        [0, 0, 0],         [0, 0, -1], -1.5708,  1.5708),   # arm_joint4 – pitch
    ([-0.00215, -0.17455, 0],[1.5708, 0, 0],    [0, 0,  1], -1.5708,  3.14159),  # arm_joint5 – wrist
]

EE_OFFSET = tf([-0.0035, -0.012625, -0.0685], [0, -1.5708, 0])  # grip_joint origin


def fk(qs):
    T = T_base.copy()
    for i, (xyz, rpy, axis, *_) in enumerate(CHAIN):
        T = T @ joint_tf(xyz, rpy, axis, qs[i])
    T = T @ EE_OFFSET
    return T[:3, 3]


# --- Dense sampling ---

N_J1 = 15   # yaw samples
N_ARM = 18  # samples per pitch joint (2, 3, 4)
N_J5 = 5    # wrist samples (mostly affects orientation)

q_samples = [
    np.linspace(c[3], c[4], n)
    for c, n in zip(CHAIN, [N_J1, N_ARM, N_ARM, N_ARM, N_J5])
]

total = N_J1 * N_ARM**3 * N_J5
print(f"Sampling {total:,} configurations …")

points = np.empty((total, 3))
idx = 0
for q1 in q_samples[0]:
    for q2 in q_samples[1]:
        for q3 in q_samples[2]:
            for q4 in q_samples[3]:
                for q5 in q_samples[4]:
                    points[idx] = fk([q1, q2, q3, q4, q5])
                    idx += 1

print(f"Computed {idx:,} end-effector positions.\n")

# --- Statistics ---
r_xy = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
print("═══ Reachable Workspace Statistics ═══")
print(f"  X range : [{points[:,0].min():.4f}, {points[:,0].max():.4f}] m")
print(f"  Y range : [{points[:,1].min():.4f}, {points[:,1].max():.4f}] m")
print(f"  Z range : [{points[:,2].min():.4f}, {points[:,2].max():.4f}] m")
print(f"  Max horizontal reach (from base Z-axis) : {r_xy.max():.4f} m")
print(f"  Min horizontal reach                    : {r_xy.min():.4f} m")
print(f"  Max height (from ground)                : {points[:,2].max():.4f} m")
print(f"  Min height (from ground)                : {points[:,2].min():.4f} m")

# --- Visualization ---

fig = plt.figure(figsize=(20, 12))
fig.suptitle("X3plus 5-DOF Arm – Reachable Workspace", fontsize=14, fontweight="bold")

sub = np.random.choice(len(points), min(30000, len(points)), replace=False)

# 1) 3D scatter
ax1 = fig.add_subplot(2, 2, 1, projection="3d")
ax1.scatter(points[sub, 0], points[sub, 1], points[sub, 2],
            c=points[sub, 2], cmap="viridis", s=0.4, alpha=0.25)
ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)"); ax1.set_zlabel("Z (m)")
ax1.set_title("3-D view")

# 2) Top-down (XY)
ax2 = fig.add_subplot(2, 2, 2)
ax2.scatter(points[sub, 0], points[sub, 1],
            c=points[sub, 2], cmap="viridis", s=0.4, alpha=0.25)
ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)")
ax2.set_title("Top view  (XY)")
ax2.set_aspect("equal"); ax2.grid(True, alpha=0.3)

# 3) Side view – radial distance vs height
r_signed = np.where(points[:, 0] >= 0, r_xy, -r_xy)
ax3 = fig.add_subplot(2, 2, 3)
ax3.scatter(r_signed[sub], points[sub, 2],
            c="steelblue", s=0.4, alpha=0.25)
ax3.set_xlabel("Radial distance (m)"); ax3.set_ylabel("Z (m)")
ax3.set_title("Side cross-section  (R vs Z)")
ax3.set_aspect("equal"); ax3.grid(True, alpha=0.3)

# 4) Front view (YZ)
ax4 = fig.add_subplot(2, 2, 4)
ax4.scatter(points[sub, 1], points[sub, 2],
            c="coral", s=0.4, alpha=0.25)
ax4.set_xlabel("Y (m)"); ax4.set_ylabel("Z (m)")
ax4.set_title("Front view  (YZ)")
ax4.set_aspect("equal"); ax4.grid(True, alpha=0.3)

plt.tight_layout()
out = "x3plus_workspace.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")
