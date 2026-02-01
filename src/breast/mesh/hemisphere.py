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
) -> GEN_HEMI:
    """
    Generate an improved UV-sphere style hemisphere mesh.

    Improvements over original:
    - Both diagonals per quad (prevents shearing)
    - Optional bending springs (smoother deformation)
    - Better spring stiffness distribution

    Args:
        radius: Hemisphere radius (arbitrary units)
        rings: Number of latitude rings (excluding top point)
        segments: Number of longitude segments
        add_bending_springs: Add springs that skip one edge (for bending resistance)

    Returns:
        (points, springs, faces) tuple
    """
    points: list[Point] = []
    springs: list[Spring] = []
    faces: list[list[int]] = []

    # 1. Create TOP POINT (apex/nipple area)
    top_point = Point(0, 10, radius, pinned=False)
    points.append(top_point)

    # 2. Create RING POINTS (from near-top to bottom)
    for r in range(1, rings + 1):
        # Angle from top (0) to equator (pi/2)
        phi = (math.pi / 2) * (r / rings)

        # Height and radius
        z_height = radius * math.cos(phi)
        ring_radius = radius * math.sin(phi)

        for s in range(segments):
            # Angle around the ring
            theta = (2 * math.pi * s) / segments

            # Calculate position
            x = ring_radius * math.cos(theta)
            y = ring_radius * math.sin(theta)
            z = z_height

            # Pin the bottom ring (chest wall)
            is_pinned = r == rings

            points.append(Point(x, y + 10, z, pinned=is_pinned))

    # 3. Add CENTER POINT for closing bottom
    center_idx = len(points)
    points.append(Point(0, 10, 0, pinned=True))

    print(f"Generated {len(points)} points")

    # 4. Create FACES
    # Top cap: triangles from apex to first ring
    for s in range(segments):
        i1 = 0  # Top point
        i2 = 1 + s
        i3 = 1 + (s + 1) % segments
        faces.append([i1, i2, i3])

    # Middle: quads between rings (split into triangles)
    for r in range(1, rings):
        curr_ring_start = 1 + (r - 1) * segments
        next_ring_start = 1 + r * segments

        for s in range(segments):
            # Four corners of quad
            i1 = curr_ring_start + s
            i2 = curr_ring_start + (s + 1) % segments
            i3 = next_ring_start + s
            i4 = next_ring_start + (s + 1) % segments

            # Two triangles (counter-clockwise winding)
            faces.append([i1, i3, i2])
            faces.append([i2, i3, i4])

    # Bottom cap: triangles from last ring to center
    bottom_ring_start = 1 + (rings - 1) * segments
    for s in range(segments):
        i1 = bottom_ring_start + s
        i2 = bottom_ring_start + (s + 1) % segments
        i3 = center_idx
        faces.append([i1, i2, i3])

    print(f"Generated {len(faces)} faces")

    # 5. Create SPRINGS with improved topology
    added_springs: set[tuple[int, ...]] = set()

    def add_unique_spring(
        i: int, j: int, stiffness: float = 0.5, spring_type: str = "edge"
    ) -> None:
        """Add spring if not already present."""
        pair = tuple(sorted((i, j)))
        if pair not in added_springs:
            springs.append(Spring(points[i], points[j], stiffness=stiffness))
            added_springs.add(pair)

    # 5a. EDGE SPRINGS (from triangle edges) - structural
    for f in faces:
        add_unique_spring(f[0], f[1], stiffness=0.8, spring_type="edge")
        add_unique_spring(f[1], f[2], stiffness=0.8, spring_type="edge")
        add_unique_spring(f[2], f[0], stiffness=0.8, spring_type="edge")

    # 5b. DIAGONAL SPRINGS (both per quad) - prevents shearing
    # This is the KEY improvement!
    for r in range(1, rings):
        curr_ring_start = 1 + (r - 1) * segments
        next_ring_start = 1 + r * segments

        for s in range(segments):
            i1 = curr_ring_start + s
            i2 = curr_ring_start + (s + 1) % segments
            i3 = next_ring_start + s
            i4 = next_ring_start + (s + 1) % segments

            # Both diagonals (was only one before!)
            add_unique_spring(i1, i4, stiffness=0.5, spring_type="diagonal")
            add_unique_spring(i2, i3, stiffness=0.5, spring_type="diagonal")

    # 5c. BENDING SPRINGS (optional) - smoother deformation
    if add_bending_springs:
        # Longitudinal bending springs (skip one ring)
        # Note: ring indices are 0-based but we skip ring 0 (apex)
        # So we have rings at indices 1, 2, ..., rings
        for r in range(rings - 1):
            curr_ring_start = 1 + r * segments
            # We want to connect ring r to ring r+2
            # Ring r+2 exists if r+2 < rings (since we go from 0 to rings-1)
            if r + 2 < rings:
                skip_ring_start = 1 + (r + 2) * segments
                for s in range(segments):
                    i1 = curr_ring_start + s
                    i2 = skip_ring_start + s
                    add_unique_spring(i1, i2, stiffness=0.2, spring_type="bending")

        # Latitudinal bending springs (skip one segment)
        for r in range(1, rings + 1):
            ring_start = 1 + (r - 1) * segments
            for s in range(segments):
                i1 = ring_start + s
                i2 = ring_start + (s + 2) % segments
                add_unique_spring(i1, i2, stiffness=0.2, spring_type="bending")

    print(f"Generated {len(springs)} springs")

    # Count spring types
    edge_count = sum(1 for s in springs if abs(s.stiffness - 0.8) < 0.01)
    diag_count = sum(1 for s in springs if abs(s.stiffness - 0.5) < 0.01)
    bend_count = sum(1 for s in springs if abs(s.stiffness - 0.2) < 0.01)

    print(f"  Edge springs:     {edge_count} (stiff)")
    print(f"  Diagonal springs: {diag_count} (medium)")
    print(f"  Bending springs:  {bend_count} (soft)")

    np_faces = np.array(faces, dtype=np.int32)
    return points, springs, np_faces
