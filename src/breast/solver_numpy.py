# solver_numpy_stable.py
"""
Ultra-stable solver with aggressive damping and better initialization.
"""

from typing import Any

from numba import njit, prange  # type: ignore
import numpy as np

from breast.models import Point, Spring
from breast.types import FACE, PROJ

# ===============================
# PHYSICS KERNELS
# ===============================


@njit(fastmath=True, cache=True, parallel=True)  # type: ignore
def integrate_verlet(
    pos: np.ndarray,
    prev_pos: np.ndarray,
    pinned_mask: np.ndarray,
    gx: PROJ,
    gy: PROJ,
    gz: PROJ,
    friction: float,
    dt: float,
    y_floor: float,
    max_vel: np.floating[Any],
) -> None:
    """Verlet integration with aggressive velocity clamping."""
    dt_sq = dt * dt
    for i in prange(len(pos)):
        if pinned_mask[i]:
            continue

        vx = (pos[i, 0] - prev_pos[i, 0]) * friction
        vy = (pos[i, 1] - prev_pos[i, 1]) * friction
        vz = (pos[i, 2] - prev_pos[i, 2]) * friction

        # Aggressive velocity clamping
        vx = min(max_vel, max(-max_vel, vx))
        vy = min(max_vel, max(-max_vel, vy))
        vz = min(max_vel, max(-max_vel, vz))

        prev_pos[i, :] = pos[i, :]

        pos[i, 0] += vx + gx * dt_sq
        pos[i, 1] += vy + gy * dt_sq
        pos[i, 2] += vz + gz * dt_sq

        # Floor collision
        if pos[i, 1] < y_floor:
            pos[i, 1] = y_floor
            prev_pos[i, 1] = y_floor


@njit(fastmath=True, cache=True, parallel=True)  # type: ignore
def solve_springs_sequential(
    pos: np.ndarray,
    spring_i: np.typing.NDArray[np.int32],
    spring_j: np.typing.NDArray[np.int32],
    rest_lengths: PROJ,
    stiffness: np.float32,
    pinned_mask: np.ndarray,
) -> None:
    """
    Sequential spring solving (more stable than parallel for stiff systems).
    """
    factor = 0.5 * stiffness

    for i in prange(len(spring_i)):
        a = spring_i[i]
        b = spring_j[i]

        if pinned_mask[a] and pinned_mask[b]:
            continue

        p1 = pos[a]
        p2 = pos[b]

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]

        dist_sq = dx * dx + dy * dy + dz * dz
        if dist_sq < 1e-8:
            continue

        dist = np.sqrt(dist_sq)
        diff = (dist - rest_lengths[i]) / dist

        off_x = dx * diff * factor
        off_y = dy * diff * factor
        off_z = dz * diff * factor

        if not pinned_mask[a]:
            pos[a, 0] += off_x
            pos[a, 1] += off_y
            pos[a, 2] += off_z

        if not pinned_mask[b]:
            pos[b, 0] -= off_x
            pos[b, 1] -= off_y
            pos[b, 2] -= off_z


@njit(fastmath=True, cache=True, parallel=True)  # type: ignore
def apply_pressure_fast(
    pos: np.ndarray,
    faces: FACE,
    pressure_val: np.float32,
    pinned_mask: np.ndarray,
) -> None:
    """Apply volumetric pressure forces."""
    if abs(pressure_val) < 1e-9:
        return

    for i in prange(len(faces)):
        i1, i2, i3 = faces[i]
        p1 = pos[i1]
        p2 = pos[i2]
        p3 = pos[i3]

        u = p2 - p1
        v = p3 - p1

        n = np.cross(u, v)
        area = np.linalg.norm(n)

        if area < 1e-8:
            continue

        force = (n / area) * pressure_val * area / 3.0

        if not pinned_mask[i1]:
            pos[i1] += force
        if not pinned_mask[i2]:
            pos[i2] += force
        if not pinned_mask[i3]:
            pos[i3] += force


@njit(fastmath=True, cache=True, parallel=True)  # type: ignore
def calculate_volume_fast(pos: np.ndarray, faces: FACE) -> float:
    """Calculate mesh volume."""
    n = len(faces)
    partial_sums = np.zeros(n, dtype=np.float64)
    for i in prange(n):
        i1, i2, i3 = faces[i]
        p1 = pos[i1]
        p2 = pos[i2]
        p3 = pos[i3]
        partial_sums[i] = np.dot(p1, np.cross(p2, p3))
    return abs(np.sum(partial_sums)) / 6.0


# ===============================
# SOLVER CLASS
# ===============================


class UltraStableSolver:
    """
    Ultra-stable solver with:
    - Sequential constraint solving (more stable)
    - Aggressive damping
    - Better initialization
    - Diagnostic output
    """

    def __init__(
        self,
        points: list[Point],
        springs: list[Spring],
        faces: FACE,
        gravity: float = -9.8,
        scale: float = 1.0,
    ) -> None:
        # Scale positions to meters
        self.faces = faces.astype(np.int32)
        self.pos = np.array(
            [[p.pos.x * scale, p.pos.y * scale, p.pos.z * scale] for p in points],
            dtype=np.float32,
        )

        # CRITICAL: Initialize prev_pos equal to pos (zero initial velocity!)
        self.prev_pos = self.pos.copy()

        self.pinned_mask = np.array([p.pinned for p in points], dtype=np.bool_)
        self.pinned_pos = self.pos[self.pinned_mask].copy()

        # Springs
        p_to_idx = {id(p): i for i, p in enumerate(points)}
        self.spring_i = np.array([p_to_idx[id(s.a)] for s in springs], dtype=np.int32)
        self.spring_j = np.array([p_to_idx[id(s.b)] for s in springs], dtype=np.int32)
        self.rest_lengths = np.array([s.rest_length * scale for s in springs], dtype=np.float32)

        # Physics params (VERY conservative for stability)
        self.gravity = np.array([0.0, gravity, 0.0], dtype=np.float32)
        self.temp_pos = np.empty_like(self.pos)
        self.displacement = np.empty(len(self.pos), dtype=np.float32)
        self.ground_y = -0.5
        self.friction = 0.995  # Very high damping

        # Material properties (very soft)
        self.stiffness = np.float32(0.1)
        self.pressure_stiffness = np.float32(0.001)

        self.rest_volume = calculate_volume_fast(self.pos, self.faces)
        self.is_exploded = False
        self.avg_edge_length = np.mean(self.rest_lengths)

        # Diagnostics
        self.max_velocity = 0.0
        self.max_displacement = 0.0
        self.steps_stable = 0

        print("Solver initialized:")
        print(f"  - {len(points)} points")
        print(f"  - {len(springs)} springs")
        print(f"  - {len(faces)} faces")
        print(f"  - Scale: {scale}m/unit")
        print(f"  - Avg edge length: {self.avg_edge_length:.4f}m")
        print(f"  - Rest volume: {self.rest_volume:.6f}m³")
        print("  - Initial velocity: 0.0 m/s (critical!)")

    def update(self, dt: float) -> None:
        """Update simulation by one timestep."""
        if self.is_exploded:
            return

        # Very conservative velocity limit
        max_disp = self.avg_edge_length * 0.1  # Only 10% of edge length!
        max_vel = max_disp / dt

        # Store old positions for diagnostics
        np.copyto(self.temp_pos, self.pos)

        # Integration
        integrate_verlet(
            self.pos,
            self.prev_pos,
            self.pinned_mask,
            self.gravity[0],
            self.gravity[1],
            self.gravity[2],
            self.friction,
            dt,
            self.ground_y,
            max_vel,
        )

        # Volumetric pressure (very gentle)
        current_vol = calculate_volume_fast(self.pos, self.faces)
        pressure_val = (self.rest_volume - current_vol) * self.pressure_stiffness

        # Many iterations with low stiffness (more stable than few with high stiffness)
        for i in range(10):
            solve_springs_sequential(
                self.pos,
                self.spring_i,
                self.spring_j,
                self.rest_lengths,
                self.stiffness,
                self.pinned_mask,
            )

            if i % 3 == 0:
                apply_pressure_fast(self.pos, self.faces, pressure_val, self.pinned_mask)

            # Enforce pinned constraints
            self.pos[self.pinned_mask] = self.pinned_pos

        # Diagnostics
        np.subtract(self.pos, self.temp_pos, out=self.displacement)
        displacement = np.max(np.abs(self.displacement))
        velocity = displacement / dt

        self.max_displacement = max(self.max_displacement, displacement)
        self.max_velocity = max(self.max_velocity, velocity)

        # Check for explosion
        if not np.isfinite(self.pos).all():
            self.is_exploded = True
            print("Warning: Simulation became unstable!")
            print(f"  Max velocity reached: {self.max_velocity:.4f} m/s")
            print(f"  Max displacement: {self.max_displacement:.4f} m")
        else:
            self.steps_stable += 1
            if self.steps_stable % 100 == 0:
                print(
                    f"Stable for {self.steps_stable} steps | Max vel: {self.max_velocity:.4f} m/s"
                )
