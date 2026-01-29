# mesh/hemisphere.py
import math

import numpy as np

from breast.models import Point, Spring


def generate_hemisphere(
    radius: float = 5.0, rings: int = 6, segments: int = 12
) -> tuple[list[Point], list[Spring], np.ndarray]:
    """
    Generate a UV-sphere style hemisphere mesh.

    Args:
            radius: Hemisphere radius
            rings: Number of latitude rings (excluding top point)
            segments: Number of longitude segments

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

        # Height decreases from radius to 0
        z_height = radius * math.cos(phi)

        # Ring radius increases
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

    print(f"Debug: Created {len(points)} points")
    print(f"  Top: ({points[0].pos.x:.2f}, {points[0].pos.y:.2f}, {points[0].pos.z:.2f})")
    print(
        f"  Mid: ({points[len(points) // 2].pos.x:.2f}, {points[len(points) // 2].pos.y:.2f}, {points[len(points) // 2].pos.z:.2f})"
    )
    print(
        f"  Bottom: ({points[center_idx - 1].pos.x:.2f}, {points[center_idx - 1].pos.y:.2f}, {points[center_idx - 1].pos.z:.2f})"
    )

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

    print(f"Debug: Created {len(faces)} faces")

    # 5. Create SPRINGS from triangle edges
    added_springs: set[tuple[int, ...]] = set()

    def add_unique_spring(i: int, j: int) -> None:
        pair = tuple(sorted((i, j)))
        if pair not in added_springs:
            springs.append(Spring(points[i], points[j], stiffness=0.5))
            added_springs.add(pair)

    # Springs from all triangle edges
    for f in faces:
        add_unique_spring(f[0], f[1])
        add_unique_spring(f[1], f[2])
        add_unique_spring(f[2], f[0])

    # Add diagonal springs for extra stability
    for r in range(1, rings):
        curr_ring_start = 1 + (r - 1) * segments
        next_ring_start = 1 + r * segments

        for s in range(segments):
            i1 = curr_ring_start + s
            i4 = next_ring_start + (s + 1) % segments
            add_unique_spring(i1, i4)

    print(f"Debug: Created {len(springs)} springs")
    print(np.array(faces, dtype=np.int32))
    return points, springs, np.array(faces, dtype=np.int32)
