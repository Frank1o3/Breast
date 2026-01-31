# solver_numpy.py

from typing import Any

from numba import njit  # type: ignore
import numpy as np

from breast.models import Point, Spring
from breast.types import FACE, PROJ

# ===============================

# SAFE NUMBA PHYSICS KERNELS

# ===============================


@njit(fastmath=True, cache=True)  # type: ignore
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
    """
    Verlet integration with velocity clamping.
    pinned_mask == True → point does NOT move
    """
    dt_sq = dt * dt
    for i in range(len(pos)):
        if pinned_mask[i]:
            continue

        vx = (pos[i, 0] - prev_pos[i, 0]) * friction
        vy = (pos[i, 1] - prev_pos[i, 1]) * friction
        vz = (pos[i, 2] - prev_pos[i, 2]) * friction

        # Clamp velocity
        vx = min(max_vel, max(-max_vel, vx))
        vy = min(max_vel, max(-max_vel, vy))
        vz = min(max_vel, max(-max_vel, vz))

        prev_pos[i, :] = pos[i, :]

        pos[i, 0] += vx + gx * dt_sq
        pos[i, 1] += vy + gy * dt_sq
        pos[i, 2] += vz + gz * dt_sq

        # Floor collision (soft clamp)
        if pos[i, 1] < y_floor:
            pos[i, 1] = y_floor
            prev_pos[i, 1] = y_floor


@njit(fastmath=True, cache=True)  # type: ignore
def solve_springs_fast(
    pos: np.ndarray,
    spring_i: np.typing.NDArray[np.int32],
    spring_j: np.typing.NDArray[np.int32],
    rest_lengths: PROJ,
    stiffness: np.float32,
    pinned_mask: np.ndarray,
) -> None:
    factor = 0.5 * stiffness

    for i in range(len(spring_i)):
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


@njit(fastmath=True, cache=True)  # type:ignore
def apply_pressure_fast(pos: np.ndarray, faces: FACE, pressure_val: np.float32) -> None:
    if abs(pressure_val) < 1e-9:
        return

    for i in range(len(faces)):
        i1, i2, i3 = faces[i]
        p1 = pos[i1]
        p2 = pos[i2]
        p3 = pos[i3]

        u = p2 - p1
        v = p3 - p1

        # Cross product = area * normal
        n = np.cross(u, v)
        area = np.linalg.norm(n)

        if area < 1e-8:
            continue

        force = (n / area) * pressure_val * area / 3.0

        pos[i1] += force
        pos[i2] += force
        pos[i3] += force


@njit(fastmath=True, cache=True)  # type: ignore
def calculate_volume_fast(pos: np.ndarray, faces: FACE) -> float:
    total = 0.0
    for i in range(len(faces)):
        i1, i2, i3 = faces[i]
        p1 = pos[i1]
        p2 = pos[i2]
        p3 = pos[i3]
        total += np.dot(p1, np.cross(p2, p3))
    return abs(total) / 6.0


# ===============================

# SOLVER CLASS

# ===============================


class NumpyBreastSolver:
    def __init__(
        self, points: list[Point], springs: list[Spring], faces: FACE, gravity: float = -9.8
    ) -> None:
        self.faces = faces.astype(np.int32)
        self.pos = np.array(
            [[p.pos.x, p.pos.y, p.pos.z] for p in points],
            dtype=np.float32,
        )
        self.prev_pos = self.pos.copy()

        self.pinned_mask = np.array([p.pinned for p in points], dtype=np.bool_)
        self.pinned_pos = self.pos[self.pinned_mask].copy()

        # Springs
        p_to_idx = {id(p): i for i, p in enumerate(points)}
        self.spring_i = np.array([p_to_idx[id(s.a)] for s in springs], dtype=np.int32)
        self.spring_j = np.array([p_to_idx[id(s.b)] for s in springs], dtype=np.int32)
        self.rest_lengths = np.array([s.rest_length for s in springs], dtype=np.float32)

        # Adaptive limit base
        self.avg_edge_length = np.mean(self.rest_lengths)

        # Physics params (WORLD SPACE)
        self.gravity = np.array([0.0, gravity, 0.0], dtype=np.float32)
        self.ground_y = -2.0
        self.friction = 0.55

        self.stiffness = np.float32(0.1)
        self.pressure_stiffness = np.float32(0.001)

        self.rest_volume = calculate_volume_fast(self.pos, self.faces)
        self.is_exploded = False

    def update(self, dt: float) -> None:
        if self.is_exploded:
            return

        # Compute adaptive velocity limit
        max_disp = self.avg_edge_length * 0.25
        max_vel = max_disp / dt

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

        current_vol = calculate_volume_fast(self.pos, self.faces)
        pressure_val = (self.rest_volume - current_vol) * self.pressure_stiffness

        for i in range(8):
            solve_springs_fast(
                self.pos,
                self.spring_i,
                self.spring_j,
                self.rest_lengths,
                self.stiffness,
                self.pinned_mask,
            )
            if i % 4 == 0:
                apply_pressure_fast(self.pos, self.faces, pressure_val)

            self.pos[self.pinned_mask] = self.pinned_pos

        if not np.isfinite(self.pos).all():
            self.is_exploded = True
