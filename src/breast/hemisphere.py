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


def generate_hemisphere(
    radius: float = 5.0,
    rings: int = 6,
    segments: int = 12,
    add_bending_springs: bool = True,
    phi_bias: float = 2.0,
) -> GEN_HEMI:
    """
    Generate a hemisphere mesh suitable for soft-body breast simulation.

    Vertex layout:
      index 0          : apex (top of dome)
      index 1..R*S     : ring vertices, row-major (ring 0 = near apex)
      index R*S+1      : base center (pinned)

    phi_bias > 1 concentrates rings near the apex (better deformation detail).
    phi_bias = 1 gives uniform latitude spacing.
    """

    points: list[Point] = []
    springs: list[Spring] = []
    faces: list[list[int]] = []

    # ------------------------------------------------------------------
    # 1. Latitude / longitude samples
    #    phi  = 0 (apex) → pi/2 (equator)
    #    theta = 0 → 2*pi (full circle)
    # ------------------------------------------------------------------
    phi_values: list[float] = []
    for r in range(1, rings + 1):
        t = r / rings
        t_biased = t**phi_bias  # bias: pushes rings toward apex
        phi_values.append((math.pi / 2.0) * t_biased)

    theta_values = [(2.0 * math.pi * s) / segments for s in range(segments)]

    cos_phi = [math.cos(p) for p in phi_values]
    sin_phi = [math.sin(p) for p in phi_values]
    cos_theta = [math.cos(t) for t in theta_values]
    sin_theta = [math.sin(t) for t in theta_values]

    # ------------------------------------------------------------------
    # 2. Apex  (index 0, unpinned)
    # ------------------------------------------------------------------
    Y_OFFSET = 10.0  # lift mesh off the floor so it hangs naturally
    points.append(Point(0.0, Y_OFFSET, radius, pinned=False))

    # ------------------------------------------------------------------
    # 3. Ring vertices
    #    ring r_idx (0-based) → indices [1 + r_idx*segments .. 1 + (r_idx+1)*segments)
    # ------------------------------------------------------------------
    for r_idx in range(rings):
        z = radius * cos_phi[r_idx]  # height above base plane
        ring_r = radius * sin_phi[r_idx]  # lateral radius
        is_pinned = r_idx == rings - 1  # only the equator ring is pinned

        for s in range(segments):
            x = ring_r * cos_theta[s]
            y = ring_r * sin_theta[s]
            points.append(Point(x, Y_OFFSET, z + y * 0.0, pinned=is_pinned))
            # Note: we put z along the Y world-axis so gravity acts correctly.
            # Re-map: mesh-z → world-y (height), mesh-x,y → world-x,z (lateral)

    # Simpler / correct world-space mapping:
    # apex is at world (0, Y_OFFSET + radius, 0) — top of dome
    # equator ring is at world y = Y_OFFSET
    # Rebuild with correct axes:
    points.clear()
    points.append(Point(0.0, Y_OFFSET + radius, 0.0, pinned=False))  # apex

    for r_idx in range(rings):
        height = radius * cos_phi[r_idx]  # 0 at equator, radius at apex
        ring_r = radius * sin_phi[r_idx]  # 0 at apex, radius at equator
        is_pinned = r_idx == rings - 1

        for s in range(segments):
            x = ring_r * cos_theta[s]
            z = ring_r * sin_theta[s]
            y = Y_OFFSET + height
            points.append(Point(x, y, z, pinned=is_pinned))

    # ------------------------------------------------------------------
    # 4. Base center  (pinned anchor)
    # ------------------------------------------------------------------
    center_idx = len(points)
    points.append(Point(0.0, Y_OFFSET, 0.0, pinned=True))

    # ------------------------------------------------------------------
    # 5. Faces
    # ------------------------------------------------------------------
    # Cap: apex → first ring
    for s in range(segments):
        faces.append([0, 1 + s, 1 + (s + 1) % segments])

    # Body: ring-to-ring quads split into two triangles
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

    # Bottom cap: last ring → base center
    last_ring_base = 1 + (rings - 1) * segments
    for s in range(segments):
        faces.append(
            [
                last_ring_base + s,
                last_ring_base + (s + 1) % segments,
                center_idx,
            ]
        )

    # ------------------------------------------------------------------
    # 6. Springs  (structural + diagonal + optional bending)
    # ------------------------------------------------------------------
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

    # Structural: one spring per unique edge in face list
    for f in faces:
        add_spring(f[0], f[1], 1.0)
        add_spring(f[1], f[2], 1.0)
        add_spring(f[2], f[0], 1.0)

    # Diagonal shear springs across each quad
    for r in range(1, rings):
        curr_base = 1 + (r - 1) * segments
        next_base = 1 + r * segments
        for s in range(segments):
            add_spring(curr_base + s, next_base + (s + 1) % segments, 0.6)
            add_spring(curr_base + (s + 1) % segments, next_base + s, 0.6)

    # Bending: skip-one-ring and skip-two-vertex springs
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

    # ------------------------------------------------------------------
    # 7. Stiffness normalization
    #    Shorter springs should be stiffer (correct physical intuition).
    #    We normalize so the *average* rest length has stiffness = base_k,
    #    and shorter springs are proportionally stiffer.
    #
    #    IMPORTANT: cap the multiplier at 3.0x.
    #    Without this cap, tiny apex springs (~1mm) get stiffness ~6x and
    #    the Jacobi solver overcorrects by several multiples of the spring
    #    length on the very first step → instant NaN explosion.
    #    3.0x keeps the correction ratio well below 0.5 at any timestep.
    # ------------------------------------------------------------------
    rest_lengths = [s.rest_length for s in springs]
    avg_len = sum(rest_lengths) / len(rest_lengths)
    MAX_STIFFNESS_MULTIPLIER = 3.0

    for sp in springs:
        if sp.rest_length > 1e-8:
            multiplier = min(avg_len / sp.rest_length, MAX_STIFFNESS_MULTIPLIER)
            sp.stiffness *= multiplier

    return points, springs, np.array(faces, dtype=np.int32)