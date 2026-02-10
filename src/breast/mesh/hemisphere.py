# hemisphere.py
"""
Improved hemisphere mesh generation with:
1. Both diagonals per quad (prevents shearing)
2. Better edge length uniformity
3. Optional bending springs for smoother deformation
"""

import math

import numpy as np

from breast.models import Point, Spring
from breast.types import GEN_HEMI


def generate_hemisphere(
    radius: float = 5.0,
    rings: int = 6,
    segments: int = 12,
    add_bending_springs: bool = True,
    phi_bias: float = 2.0,  # NEW: controls where resolution goes
) -> GEN_HEMI:
    """
    Breast-optimized hemisphere mesh.

    Compatible with old code, but with:
    - Biased latitude sampling (better deformation)
    - More uniform spring lengths
    - Stable high-resolution behavior
    """

    points: list[Point] = []
    springs: list[Spring] = []
    faces: list[list[int]] = []

    # -----------------------------
    # 1. Biased latitude sampling
    # -----------------------------
    phi_values: list[float] = []
    for r in range(1, rings + 1):
        t = r / rings
        t = t**phi_bias  # <<< KEY CHANGE
        phi_values.append((math.pi / 2) * t)

    theta_values = [(2 * math.pi * s) / segments for s in range(segments)]

    cos_phi = [math.cos(p) for p in phi_values]
    sin_phi = [math.sin(p) for p in phi_values]
    cos_theta = [math.cos(t) for t in theta_values]
    sin_theta = [math.sin(t) for t in theta_values]

    # -----------------------------
    # 2. Apex (top point)
    # -----------------------------
    top_point = Point(0, 10, radius, pinned=False)
    points.append(top_point)

    # -----------------------------
    # 3. Rings
    # -----------------------------
    for r_idx in range(rings):
        z = radius * cos_phi[r_idx]
        ring_r = radius * sin_phi[r_idx]

        is_pinned = r_idx == rings - 1

        for s in range(segments):
            x = ring_r * cos_theta[s]
            y = ring_r * sin_theta[s]
            points.append(Point(x, y + 10, z, pinned=is_pinned))

    # -----------------------------
    # 4. Bottom center (pinned)
    # -----------------------------
    center_idx = len(points)
    points.append(Point(0, 10, 0, pinned=True))

    # -----------------------------
    # 5. Faces
    # -----------------------------
    for s in range(segments):
        faces.append([0, 1 + s, 1 + (s + 1) % segments])

    for r in range(1, rings):
        curr = 1 + (r - 1) * segments
        nxt = 1 + r * segments

        for s in range(segments):
            i1 = curr + s
            i2 = curr + (s + 1) % segments
            i3 = nxt + s
            i4 = nxt + (s + 1) % segments

            faces.append([i1, i3, i2])
            faces.append([i2, i3, i4])

    bottom = 1 + (rings - 1) * segments
    for s in range(segments):
        faces.append([bottom + s, bottom + (s + 1) % segments, center_idx])

    # -----------------------------
    # 6. Springs (length-normalized)
    # -----------------------------
    added: set[tuple[int, int]] = set()
    rest_lengths: list[float] = []

    def add_spring(i: int, j: int, base_k: float) -> None:
        pair = tuple(sorted((i, j)))
        if pair in added:
            return
        a, b = points[i], points[j]
        rest = (b.pos - a.pos).length()
        springs.append(Spring(a, b, stiffness=base_k, rest_length=rest))
        rest_lengths.append(rest)
        added.add(pair)

    # Structural
    for f in faces:
        add_spring(f[0], f[1], 0.8)
        add_spring(f[1], f[2], 0.8)
        add_spring(f[2], f[0], 0.8)

    # Diagonals
    for r in range(1, rings):
        curr = 1 + (r - 1) * segments
        nxt = 1 + r * segments
        for s in range(segments):
            add_spring(curr + s, nxt + (s + 1) % segments, 0.5)
            add_spring(curr + (s + 1) % segments, nxt + s, 0.5)

    # Bending
    if add_bending_springs:
        for r in range(rings - 2):
            curr = 1 + r * segments
            skip = 1 + (r + 2) * segments
            for s in range(segments):
                add_spring(curr + s, skip + s, 0.2)

        for r in range(rings):
            ring = 1 + r * segments
            for s in range(segments):
                add_spring(ring + s, ring + (s + 3) % segments, 0.2)

    # -----------------------------
    # 7. Normalize spring stiffness
    # -----------------------------
    avg_len = sum(rest_lengths) / len(rest_lengths)
    for s in springs:
        s.stiffness *= s.rest_length / avg_len

    return points, springs, np.array(faces, dtype=np.int32)
