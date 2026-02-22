# hemisphere.py
"""
Hemisphere mesh generation.

Design goals:
- Single clean pass (no duplicate loops)
- phi_bias actually applied
- Springs built once, no duplicates
- Stiffness normalized correctly (shorter = stiffer)
"""

from __future__ import annotations

import math

import numpy as np

from breast.models import Point, Spring
from breast.types import GEN_HEMI


# ------------------------------------------------------------
# Rotation Matrix Builder (Euler XYZ order)
# Applies rotations in order: X -> Y -> Z
# ------------------------------------------------------------
def build_rotation_matrix(rx: float, ry: float, rz: float) -> np.typing.NDArray[np.float64]:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cx, -sx],
            [0.0, sx, cx],
        ],
        dtype=np.float32,
    )

    Ry = np.array(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ],
        dtype=np.float32,
    )

    Rz = np.array(
        [
            [cz, -sz, 0.0],
            [sz, cz, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    return Rz @ Ry @ Rx


# ------------------------------------------------------------
# Hemisphere Generator
# ------------------------------------------------------------
def generate_hemisphere(
    radius: float = 5.0,
    rings: int = 6,
    segments: int = 12,
    add_bending_springs: bool = True,
    phi_bias: float = 2.0,
    rot_x: float = 0.0,
    rot_y: float = 0.0,
    rot_z: float = 0.0,
) -> GEN_HEMI:
    points: list[Point] = []
    springs: list[Spring] = []
    faces: list[list[int]] = []

    Y_OFFSET = 10.0  # lift mesh so it hangs naturally

    # --------------------------------------------------------
    # 1. Latitude / longitude samples
    # --------------------------------------------------------
    phi_values: list[float] = []
    for r in range(1, rings + 1):
        t = r / rings
        phi_values.append((math.pi / 2.0) * (t**phi_bias))

    theta_values = [(2.0 * math.pi * s) / segments for s in range(segments)]

    cos_phi = [math.cos(p) for p in phi_values]
    sin_phi = [math.sin(p) for p in phi_values]
    cos_theta = [math.cos(t) for t in theta_values]
    sin_theta = [math.sin(t) for t in theta_values]

    # --------------------------------------------------------
    # 2. Apex (canonical orientation: +Y direction)
    # --------------------------------------------------------
    points.append(Point(0.0, Y_OFFSET + radius, 0.0, pinned=False))

    # --------------------------------------------------------
    # 3. Ring vertices (canonical orientation)
    # --------------------------------------------------------
    for r_idx in range(rings):
        height = radius * cos_phi[r_idx]
        ring_r = radius * sin_phi[r_idx]
        is_pinned = r_idx == rings - 1

        for s in range(segments):
            x = ring_r * cos_theta[s]
            z = ring_r * sin_theta[s]
            y = Y_OFFSET + height
            points.append(Point(x, y, z, pinned=is_pinned))

    # --------------------------------------------------------
    # 4. Base center (pinned)
    # --------------------------------------------------------
    center_idx = len(points)
    points.append(Point(0.0, Y_OFFSET, 0.0, pinned=True))

    # --------------------------------------------------------
    # 5. Apply Rotation (around hemisphere pivot)
    # --------------------------------------------------------
    if rot_x != 0.0 or rot_y != 0.0 or rot_z != 0.0:
        R = build_rotation_matrix(rot_x, rot_y, rot_z)

        pivot = np.array([0.0, Y_OFFSET, 0.0])

        for p in points:
            v = np.array([p.pos.x, p.pos.y, p.pos.z])
            v -= pivot
            v = R @ v
            v += pivot
            p.pos.x, p.pos.y, p.pos.z = v

    # --------------------------------------------------------
    # 6. Faces
    # --------------------------------------------------------
    for s in range(segments):
        faces.append([0, 1 + s, 1 + (s + 1) % segments])

    for r in range(1, rings):
        curr_base = 1 + (r - 1) * segments
        next_base = 1 + r * segments

        for s in range(segments):
            i1 = curr_base + s
            i2 = curr_base + (s + 1) % segments
            i3 = next_base + s
            i4 = next_base + (s + 1) % segments

            faces.append([i1, i3, i2])
            faces.append([i2, i3, i4])

    last_ring_base = 1 + (rings - 1) * segments
    for s in range(segments):
        faces.append(
            [
                last_ring_base + s,
                last_ring_base + (s + 1) % segments,
                center_idx,
            ]
        )

    # --------------------------------------------------------
    # 7. Springs
    # --------------------------------------------------------
    added: set[tuple[int, int]] = set()

    def add_spring(i: int, j: int, k_base: float) -> None:
        key = (min(i, j), max(i, j))
        if key in added:
            return

        a, b = points[i], points[j]
        rest = (
            (b.pos.x - a.pos.x) ** 2 + (b.pos.y - a.pos.y) ** 2 + (b.pos.z - a.pos.z) ** 2
        ) ** 0.5

        springs.append(Spring(a, b, stiffness=k_base, rest_length=rest))
        added.add(key)

    for f in faces:
        add_spring(f[0], f[1], 1.0)
        add_spring(f[1], f[2], 1.0)
        add_spring(f[2], f[0], 1.0)

    for r in range(1, rings):
        curr_base = 1 + (r - 1) * segments
        next_base = 1 + r * segments
        for s in range(segments):
            add_spring(curr_base + s, next_base + (s + 1) % segments, 0.6)
            add_spring(curr_base + (s + 1) % segments, next_base + s, 0.6)

    if add_bending_springs:
        for r in range(rings - 2):
            curr_base = 1 + r * segments
            skip_base = 1 + (r + 2) * segments
            for s in range(segments):
                add_spring(curr_base + s, skip_base + s, 0.25)

        for r in range(rings):
            ring_base = 1 + r * segments
            for s in range(segments):
                add_spring(ring_base + s, ring_base + (s + 2) % segments, 0.25)

    # --------------------------------------------------------
    # 8. Stiffness normalization
    # --------------------------------------------------------
    rest_lengths = [s.rest_length for s in springs]
    avg_len = sum(rest_lengths) / len(rest_lengths)
    MAX_STIFFNESS_MULTIPLIER = 3.0

    for sp in springs:
        if sp.rest_length > 1e-8:
            multiplier = min(avg_len / sp.rest_length, MAX_STIFFNESS_MULTIPLIER)
            sp.stiffness *= multiplier

    return points, springs, np.array(faces, dtype=np.int32)
