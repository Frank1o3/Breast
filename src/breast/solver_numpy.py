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


@njit(fastmath=True, cache=True, parallel= True)  # type: ignore
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


def color_springs(
    spring_i: np.typing.NDArray[np.int32],
    spring_j: np.typing.NDArray[np.int32],
    num_points: int,
) -> list[np.typing.NDArray[np.int32]]:
    """Returns list of arrays, each array is indices of non-conflicting springs."""
    colors = np.full(len(spring_i), -1, dtype=np.int32)
    neighbor_colors: list[set[int]] = [set() for _ in range(num_points)]

    for s in range(len(spring_i)):
        a, b = spring_i[s], spring_j[s]
        used = neighbor_colors[a] | neighbor_colors[b]
        c = 0
        while c in used:
            c += 1
        colors[s] = c
        neighbor_colors[a].add(c)
        neighbor_colors[b].add(c)

    num_colors = int(colors.max()) + 1
    return [np.where(colors == c)[0].astype(np.int32) for c in range(num_colors)]


@njit(fastmath=True, cache=True, parallel=True)  # type: ignore
def solve_springs_group(
    pos: np.ndarray,
    group: np.typing.NDArray[np.int32],   # indices INTO spring_i/spring_j for this color
    spring_i: np.typing.NDArray[np.int32],
    spring_j: np.typing.NDArray[np.int32],
    rest_lengths: PROJ,
    stiffness: np.float32,
    pinned_mask: np.ndarray,
) -> None:
    """
    Parallel spring solving for a single color group.
    Springs within a group are guaranteed to share no vertices,
    so prange is safe — no write conflicts.
    """
    factor = np.float32(0.5) * stiffness

    for k in prange(len(group)):
        s = group[k]          # actual spring index
        a = spring_i[s]
        b = spring_j[s]

        if pinned_mask[a] and pinned_mask[b]:
            continue

        dx = pos[b, 0] - pos[a, 0]
        dy = pos[b, 1] - pos[a, 1]
        dz = pos[b, 2] - pos[a, 2]

        dist_sq = dx * dx + dy * dy + dz * dz
        if dist_sq < 1e-8:
            continue

        dist = np.sqrt(dist_sq)
        diff = (dist - rest_lengths[s]) / dist

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


@njit(fastmath=True, cache=True)  # type: ignore
def apply_pressure_fast(
    pos: np.ndarray,
    faces: FACE,
    pressure_val: np.float32,
    pinned_mask: np.ndarray,
) -> None:
    """Apply volumetric pressure forces."""
    if abs(pressure_val) < 1e-9:
        return

    for i in range(len(faces)):
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
    """Calculate mesh volume in parallel.

    Each face contribution is independent (read-only on pos), so prange
    is safe. Numba reduces the per-thread partial sums automatically when
    you accumulate into a scalar inside prange.
    """
    total = 0.0
    for i in prange(len(faces)):
        i1, i2, i3 = faces[i]
        p1 = pos[i1]
        p2 = pos[i2]
        p3 = pos[i3]
        total += p1[0] * (p2[1] * p3[2] - p2[2] * p3[1]) \
               + p1[1] * (p2[2] * p3[0] - p2[0] * p3[2]) \
               + p1[2] * (p2[0] * p3[1] - p2[1] * p3[0])
    return abs(total) / 6.0


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
        self.spring_color_groups = color_springs(self.spring_i, self.spring_j, len(points))
        self.rest_lengths = np.array([s.rest_length * scale for s in springs], dtype=np.float32)

        # Physics params (VERY conservative for stability)
        self.gravity = np.array([0.0, gravity, 0.0], dtype=np.float32)
        self.ground_y = -0.5
        self.friction = 0.98  # Very high damping

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
        old_pos = self.pos.copy()

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

        # Relative, dimensionless volume error
        vol_error = (self.rest_volume - current_vol) / self.rest_volume

        # Hard clamp to avoid pressure explosions
        vol_error = np.clip(vol_error, -0.1, 0.1)  # ±10% max correction

        pressure_val = vol_error * self.pressure_stiffness


        # Spring solving — iterate over color groups.
        # Within each group, springs share no vertices → safe to prange.
        # Groups must be solved sequentially w.r.t. each other (data dependency).
        for iteration in range(5):
            for group in self.spring_color_groups:
                solve_springs_group(
                    self.pos,
                    group,
                    self.spring_i,
                    self.spring_j,
                    self.rest_lengths,
                    self.stiffness,
                    self.pinned_mask,
                )

            if iteration % 3 == 0:
                apply_pressure_fast(self.pos, self.faces, pressure_val, self.pinned_mask)

            # Enforce pinned constraints after each full iteration
            self.pos[self.pinned_mask] = self.pinned_pos

        # Diagnostics
        displacement = np.max(np.abs(self.pos - old_pos))
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
