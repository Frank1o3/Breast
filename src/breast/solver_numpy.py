# solver_numpy.py
"""
Fully-parallel Jacobi soft-body solver.

Architecture
------------
Every physics operation is expressed as a bulk numpy/numba operation over
ALL elements at once — zero sequential Python loops in the hot path.

Spring solving uses the Jacobi approach:
  1. Read positions (no writes yet)
  2. Compute ALL spring correction vectors simultaneously  →  (S, 3) array
  3. Scatter-add into a delta buffer
  4. Divide by vertex valence (average the corrections)
  5. Write back to pos in one shot

This is 100 % data-parallel and maps directly to SIMD / GPU execution.
Convergence per iteration is slightly slower than Gauss-Seidel, but each
iteration executes an order of magnitude faster.

Pressure uses the divergence-theorem volume formula and outward face-normal
forces — also fully vectorized.

Verlet integration is a simple @njit parallel kernel.
"""

from __future__ import annotations

from numba import njit, prange  # type: ignore
import numpy as np

from breast.models import Point, Spring
from breast.types import FACE

# ==========================================================================
# Numba kernels  (compiled once, called every sub-step)
# ==========================================================================


@njit(fastmath=True, cache=True, parallel=True)
def _verlet_integrate(
    pos: np.ndarray,  # (N, 3) float32  — current positions
    prev_pos: np.ndarray,  # (N, 3) float32  — previous positions (modified in-place)
    pinned: np.ndarray,  # (N,)   bool
    grav: np.ndarray,  # (3,)   float32
    damping: float,  # velocity retention per step  (0 < d ≤ 1)
    dt: float,
    floor_y: float,
    max_vel: float,
) -> None:
    """
    Verlet integration with per-vertex velocity clamping and floor collision.

    velocity = (pos - prev_pos) * damping          — implicit damping
    new_pos  = pos + velocity + gravity * dt²
    """
    dt2 = dt * dt
    gx, gy, gz = grav[0], grav[1], grav[2]

    for i in prange(len(pos)):
        if pinned[i]:
            continue

        # Velocity from previous frame (implicit Verlet)
        vx = (pos[i, 0] - prev_pos[i, 0]) * damping
        vy = (pos[i, 1] - prev_pos[i, 1]) * damping
        vz = (pos[i, 2] - prev_pos[i, 2]) * damping

        # Clamp velocity magnitude to prevent explosion
        spd_sq = vx * vx + vy * vy + vz * vz
        if spd_sq > max_vel * max_vel:
            inv = max_vel / (spd_sq**0.5)
            vx *= inv
            vy *= inv
            vz *= inv

        # Store current as previous
        prev_pos[i, 0] = pos[i, 0]
        prev_pos[i, 1] = pos[i, 1]
        prev_pos[i, 2] = pos[i, 2]

        # Integrate
        pos[i, 0] += vx + gx * dt2
        pos[i, 1] += vy + gy * dt2
        pos[i, 2] += vz + gz * dt2

        # Floor
        if pos[i, 1] < floor_y:
            pos[i, 1] = floor_y
            prev_pos[i, 1] = floor_y  # zero vertical velocity at floor


@njit(fastmath=True, cache=True, parallel=True)
def _compute_spring_deltas(
    pos: np.ndarray,  # (N, 3) float32  — READ ONLY
    spring_i: np.ndarray,  # (S,)   int32
    spring_j: np.ndarray,  # (S,)   int32
    rest_lengths: np.ndarray,  # (S,)   float32
    stiffness: np.ndarray,  # (S,)   float32  — per-spring stiffness
    delta: np.ndarray,  # (N, 3) float32  — accumulator (zeroed before call)
    counts: np.ndarray,  # (N,)   float32  — valence counter (zeroed before call)
    pinned: np.ndarray,  # (N,)   bool
) -> None:
    """
    Jacobi spring correction — fully parallel read phase.

    For each spring (a, b):
      d     = pos[b] - pos[a]
      dist  = |d|
      corr  = d * (1 - rest / dist) * stiffness * 0.5

      delta[a] +=  corr   (if not pinned)
      delta[b] += -corr   (if not pinned)
      counts[a,b] += 1

    NOTE: np.add.at is not available in Numba, so we use a sequential
          scatter after the parallel computation step.  The parallel
          part (computing per-spring corrections) is the expensive one.
    """
    # Phase 1: compute per-spring correction vectors in parallel
    # Store in temporary arrays indexed by spring
    n_springs = len(spring_i)
    corr_x = np.empty(n_springs, dtype=np.float32)
    corr_y = np.empty(n_springs, dtype=np.float32)
    corr_z = np.empty(n_springs, dtype=np.float32)

    for s in prange(n_springs):
        a = spring_i[s]
        b = spring_j[s]

        dx = pos[b, 0] - pos[a, 0]
        dy = pos[b, 1] - pos[a, 1]
        dz = pos[b, 2] - pos[a, 2]

        dist_sq = dx * dx + dy * dy + dz * dz
        if dist_sq < 1e-10:
            corr_x[s] = 0.0
            corr_y[s] = 0.0
            corr_z[s] = 0.0
            continue

        dist = dist_sq**0.5
        k = stiffness[s] * 0.5 * (1.0 - rest_lengths[s] / dist)

        corr_x[s] = dx * k
        corr_y[s] = dy * k
        corr_z[s] = dz * k

    # Phase 2: sequential scatter (atomic-free, avoids race conditions)
    for s in range(n_springs):
        a = spring_i[s]
        b = spring_j[s]
        cx = corr_x[s]
        cy = corr_y[s]
        cz = corr_z[s]

        if not pinned[a]:
            delta[a, 0] += cx
            delta[a, 1] += cy
            delta[a, 2] += cz
            counts[a] += 1.0

        if not pinned[b]:
            delta[b, 0] -= cx
            delta[b, 1] -= cy
            delta[b, 2] -= cz
            counts[b] += 1.0


@njit(fastmath=True, cache=True, parallel=True)
def _apply_deltas(
    pos: np.ndarray,  # (N, 3) float32
    delta: np.ndarray,  # (N, 3) float32
    counts: np.ndarray,  # (N,)   float32
    pinned: np.ndarray,  # (N,)   bool
) -> None:
    """Apply averaged Jacobi corrections."""
    for i in prange(len(pos)):
        if pinned[i] or counts[i] < 0.5:
            continue
        inv = 1.0 / counts[i]
        pos[i, 0] += delta[i, 0] * inv
        pos[i, 1] += delta[i, 1] * inv
        pos[i, 2] += delta[i, 2] * inv


@njit(fastmath=True, cache=True, parallel=True)
def _calc_volume(pos: np.ndarray, faces: np.ndarray) -> float:
    """
    Signed volume via divergence theorem.

    V = (1/6) * Σ  dot(v0, cross(v1, v2))
    """
    total = 0.0
    for i in prange(len(faces)):
        i0, i1, i2 = faces[i, 0], faces[i, 1], faces[i, 2]
        x0, y0, z0 = pos[i0, 0], pos[i0, 1], pos[i0, 2]
        x1, y1, z1 = pos[i1, 0], pos[i1, 1], pos[i1, 2]
        x2, y2, z2 = pos[i2, 0], pos[i2, 1], pos[i2, 2]
        total += x0 * (y1 * z2 - y2 * z1) + y0 * (z1 * x2 - z2 * x1) + z0 * (x1 * y2 - x2 * y1)
    return abs(total) / 6.0


@njit(fastmath=True, cache=True, parallel=True)
def _apply_pressure(
    pos: np.ndarray,  # (N, 3) float32
    faces: np.ndarray,  # (F, 3) int32
    pressure: float,  # scalar  (pressure_stiffness * vol_error)
    pinned: np.ndarray,  # (N,)   bool
) -> None:
    """
    Outward pressure force along each face's area-weighted normal.

    F_per_vertex = (cross(e0, e1) / 2) * pressure / 3
                    ^^ outward normal * area ^^

    Sequential scatter (same race-condition reason as springs).
    """
    if abs(pressure) < 1e-12:
        return

    n_faces = len(faces)
    # Phase 1: compute per-face force vectors in parallel
    fx = np.empty(n_faces, dtype=np.float32)
    fy = np.empty(n_faces, dtype=np.float32)
    fz = np.empty(n_faces, dtype=np.float32)

    for i in prange(n_faces):
        i0, i1, i2 = faces[i, 0], faces[i, 1], faces[i, 2]
        e0x = pos[i1, 0] - pos[i0, 0]
        e0y = pos[i1, 1] - pos[i0, 1]
        e0z = pos[i1, 2] - pos[i0, 2]
        e1x = pos[i2, 0] - pos[i0, 0]
        e1y = pos[i2, 1] - pos[i0, 1]
        e1z = pos[i2, 2] - pos[i0, 2]

        # cross product = outward normal * area
        nx = e0y * e1z - e0z * e1y
        ny = e0z * e1x - e0x * e1z
        nz = e0x * e1y - e0y * e1x

        # Force per vertex = (normal * area * pressure) / 3
        scale = pressure / 3.0
        fx[i] = nx * scale
        fy[i] = ny * scale
        fz[i] = nz * scale

    # Phase 2: sequential scatter
    for i in range(n_faces):
        i0, i1, i2 = faces[i, 0], faces[i, 1], faces[i, 2]
        if not pinned[i0]:
            pos[i0, 0] += fx[i]
            pos[i0, 1] += fy[i]
            pos[i0, 2] += fz[i]
        if not pinned[i1]:
            pos[i1, 0] += fx[i]
            pos[i1, 1] += fy[i]
            pos[i1, 2] += fz[i]
        if not pinned[i2]:
            pos[i2, 0] += fx[i]
            pos[i2, 1] += fy[i]
            pos[i2, 2] += fz[i]


# ==========================================================================
# Solver class
# ==========================================================================


class UltraStableSolver:
    """
    Jacobi soft-body solver.

    Update loop per sub-step
    ------------------------
    1. Verlet integrate  (parallel kernel)
    2. For each iteration:
       a. Compute spring deltas  (parallel compute, sequential scatter)
       b. Apply averaged deltas  (parallel kernel)
    3. Apply pressure once       (parallel compute, sequential scatter)
    4. Enforce pin constraints   (masked numpy assignment)
    """

    def __init__(
        self,
        points: list[Point],
        springs: list[Spring],
        faces: FACE,
        gravity: float = -9.8,
        scale: float = 1.0,
    ) -> None:
        n = len(points)
        self.faces = faces.astype(np.int32)

        # Positions in world-space metres
        self.pos = np.array(
            [[p.pos.x * scale, p.pos.y * scale, p.pos.z * scale] for p in points],
            dtype=np.float32,
        )
        self.prev_pos = self.pos.copy()  # zero initial velocity

        self.pinned = np.array([p.pinned for p in points], dtype=np.bool_)
        self._pinned_pos = self.pos[self.pinned].copy()  # saved for enforcement

        # Spring arrays (flat, index-based — no object overhead)
        pid = {id(p): i for i, p in enumerate(points)}
        self.spring_i = np.array([pid[id(s.a)] for s in springs], dtype=np.int32)
        self.spring_j = np.array([pid[id(s.b)] for s in springs], dtype=np.int32)
        self.rest_lengths = np.array([s.rest_length * scale for s in springs], dtype=np.float32)
        self.spring_k = np.array([s.stiffness for s in springs], dtype=np.float32)

        # Pre-allocate Jacobi work buffers (reused every step — no allocation in hot path)
        self._delta = np.zeros((n, 3), dtype=np.float32)
        self._counts = np.zeros(n, dtype=np.float32)

        # Physics parameters
        self.gravity = np.array([0.0, gravity, 0.0], dtype=np.float32)
        self.floor_y = -0.5

        # damping: 0.999 at 120 Hz substeps retains velocity well
        # reduce if unstable, increase if over-damped
        self.damping = np.float32(0.999)

        # stiffness multiplier (tune live via sim.py with Z/X keys)
        # Keep at 0.1: spring_k is already normalized up to 3x base,
        # so 0.1 global gives effective k range of [0.025 .. 0.3] — stable.
        # Increasing toward 1.0 will add rigidity but risks instability.
        self.stiffness = np.float32(0.1)
        # pressure: how strongly volume is maintained
        self.pressure_stiffness = np.float32(0.05)

        # Jacobi iterations per sub-step (4-8 is typical sweet spot)
        self.iterations = 6

        # Rest volume (computed once at init)
        self.rest_volume = _calc_volume(self.pos, self.faces)

        # Stability tracking
        self.is_exploded = False
        self.steps_stable = 0
        self.avg_edge = float(np.mean(self.rest_lengths))

        print(f"[Solver] {n} pts | {len(springs)} springs | {len(faces)} faces")
        print(
            f"[Solver] scale={scale} | avg_edge={self.avg_edge:.4f}m | V0={self.rest_volume:.6f}m³"
        )

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self, dt: float) -> None:
        if self.is_exploded:
            return

        max_vel = self.avg_edge * 0.5 / dt  # max half-edge per step

        # 1. Verlet integration
        _verlet_integrate(
            self.pos,
            self.prev_pos,
            self.pinned,
            self.gravity,
            float(self.damping),
            dt,
            self.floor_y,
            max_vel,
        )

        # 2. Compute effective spring stiffness for this step
        #    (user can tune self.stiffness at runtime)
        eff_k = self.spring_k * float(self.stiffness)

        # 3. Jacobi spring iterations
        for _ in range(self.iterations):
            self._delta[:] = 0.0
            self._counts[:] = 0.0

            _compute_spring_deltas(
                self.pos,
                self.spring_i,
                self.spring_j,
                self.rest_lengths,
                eff_k,
                self._delta,
                self._counts,
                self.pinned,
            )

            _apply_deltas(self.pos, self._delta, self._counts, self.pinned)

        # 4. Pressure  (applied ONCE, after springs have settled for this step)
        cur_vol = _calc_volume(self.pos, self.faces)
        vol_error = np.clip(
            (self.rest_volume - cur_vol) / max(self.rest_volume, 1e-8),
            -0.3,
            0.3,  # allow up to ±30% volume correction
        )
        pressure_val = float(vol_error * self.pressure_stiffness)

        _apply_pressure(self.pos, self.faces, pressure_val, self.pinned)

        # 5. Enforce pinned constraints
        self.pos[self.pinned] = self._pinned_pos

        # 6. Stability check
        if not np.isfinite(self.pos).all():
            self.is_exploded = True
            print("[Solver] EXPLOSION — NaN/Inf detected, resetting next frame")
            return

        self.steps_stable += 1
        if self.steps_stable % 500 == 0:
            cur_vel = np.max(np.abs(self.pos - self.prev_pos)) / dt
            print(
                f"[Solver] {self.steps_stable} steps stable | max_vel={cur_vel:.4f} m/s | V={cur_vol:.5f}"
            )
