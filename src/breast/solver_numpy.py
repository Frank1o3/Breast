from numba import njit
import numpy as np

# --- SAFE NUMBA PHYSICS KERNELS ---


@njit(fastmath=True, cache=True)
def integrate_verlet(pos, prev_pos, gravity, friction, dt, y_floor):
    """
    Handles integration with safety clamps to prevent explosions.
    """
    dt_sq = dt * dt
    gx, gy, gz = gravity

    # SAFETY: Max velocity per frame (prevents infinity)
    MAX_VEL = 5.0

    for i in range(len(pos)):
        # Calculate velocity implicitly
        vx = (pos[i, 0] - prev_pos[i, 0]) * friction
        vy = (pos[i, 1] - prev_pos[i, 1]) * friction
        vz = (pos[i, 2] - prev_pos[i, 2]) * friction

        # CLAMP VELOCITY (Anti-Explosion)
        if vx > MAX_VEL:
            vx = MAX_VEL
        if vx < -MAX_VEL:
            vx = -MAX_VEL
        if vy > MAX_VEL:
            vy = MAX_VEL
        if vy < -MAX_VEL:
            vy = -MAX_VEL
        if vz > MAX_VEL:
            vz = MAX_VEL
        if vz < -MAX_VEL:
            vz = -MAX_VEL

        # Store history
        prev_pos[i, 0] = pos[i, 0]
        prev_pos[i, 1] = pos[i, 1]
        prev_pos[i, 2] = pos[i, 2]

        # Apply Integration
        pos[i, 0] += vx + gx * dt_sq
        pos[i, 1] += vy + gy * dt_sq
        pos[i, 2] += vz + gz * dt_sq

        # Floor Collision
        if pos[i, 1] < y_floor:
            pos[i, 1] = y_floor
            prev_pos[i, 1] = y_floor


@njit(fastmath=True, cache=True)
def solve_springs_fast(pos, spring_indices, rest_lengths, stiffness):
    factor = 0.5 * stiffness

    for i in range(len(spring_indices)):
        idx_a = spring_indices[i, 0]
        idx_b = spring_indices[i, 1]

        p1 = pos[idx_a]
        p2 = pos[idx_b]

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]

        dist_sq = dx * dx + dy * dy + dz * dz

        # SAFETY: Prevent division by zero
        if dist_sq < 1e-8:
            continue

        dist = np.sqrt(dist_sq)
        diff = (dist - rest_lengths[i]) / dist

        # Apply force
        off_x = dx * diff * factor
        off_y = dy * diff * factor
        off_z = dz * diff * factor

        pos[idx_a, 0] += off_x
        pos[idx_a, 1] += off_y
        pos[idx_a, 2] += off_z

        pos[idx_b, 0] -= off_x
        pos[idx_b, 1] -= off_y
        pos[idx_b, 2] -= off_z


@njit(fastmath=True, cache=True)
def apply_pressure_fast(pos, faces, pressure_val):
    # If pressure is tiny, skip it to save time
    if abs(pressure_val) < 1e-9:
        return

    force_scale = pressure_val / 6.0

    for i in range(len(faces)):
        i1, i2, i3 = faces[i]

        p1 = pos[i1]
        p2 = pos[i2]
        p3 = pos[i3]

        # Vector edges
        u_x, u_y, u_z = p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]
        v_x, v_y, v_z = p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]

        # Cross Product
        nx = u_y * v_z - u_z * v_y
        ny = u_z * v_x - u_x * v_z
        nz = u_x * v_y - u_y * v_x

        # Force per vertex
        fx = nx * force_scale
        fy = ny * force_scale
        fz = nz * force_scale

        # Distribute
        pos[i1, 0] += fx
        pos[i1, 1] += fy
        pos[i1, 2] += fz
        pos[i2, 0] += fx
        pos[i2, 1] += fy
        pos[i2, 2] += fz
        pos[i3, 0] += fx
        pos[i3, 1] += fy
        pos[i3, 2] += fz


@njit(fastmath=True, cache=True)
def calculate_volume_fast(pos, faces):
    total_vol = 0.0
    for i in range(len(faces)):
        i1, i2, i3 = faces[i]
        p1 = pos[i1]
        p2 = pos[i2]
        p3 = pos[i3]

        cx = (p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1])
        cy = (p2[2] - p1[2]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[2] - p1[2])
        cz = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

        total_vol += p1[0] * cx + p1[1] * cy + p1[2] * cz
    return abs(total_vol) / 6.0


class NumpyBreastSolver:
    def __init__(self, points, springs, faces, gravity=-9.8):
        # 1. Setup Arrays
        self.faces = faces.astype(np.int32)
        self.pos = np.array([[p.pos.x, p.pos.y, p.pos.z] for p in points], dtype=np.float64)
        self.prev_pos = self.pos.copy()

        self.pinned_mask = np.array([p.pinned for p in points], dtype=bool)
        self.pinned_pos = self.pos[self.pinned_mask].copy()

        # 2. Springs
        p_to_idx = {id(p): i for i, p in enumerate(points)}
        indices = [[p_to_idx[id(s.a)], p_to_idx[id(s.b)]] for s in springs]
        lengths = [s.rest_length for s in springs]

        self.spring_indices = np.array(indices, dtype=np.int32)
        self.rest_lengths = np.array(lengths, dtype=np.float64)

        # 3. Parameters (LOWER STIFFNESS INITIALLY)
        self.gravity = np.array([0, gravity, 0], dtype=np.float64)
        self.ground_y = -10.0
        self.friction = 0.99

        # Start softer to prevent explosion on frame 1
        self.stiffness = 0.1
        self.pressure_stiffness = 0.001

        self.rest_volume = calculate_volume_fast(self.pos, self.faces)
        self.is_exploded = False

    def update(self, dt: float) -> None:
        if self.is_exploded:
            return

        # 1. Physics Step
        integrate_verlet(self.pos, self.prev_pos, self.gravity, self.friction, dt, self.ground_y)

        # 2. Constraint Solving
        # Calculate volume only once per frame to save speed
        current_vol = calculate_volume_fast(self.pos, self.faces)
        pressure_val = (self.rest_volume - current_vol) * self.pressure_stiffness

        for _ in range(8):
            solve_springs_fast(self.pos, self.spring_indices, self.rest_lengths, self.stiffness)
            apply_pressure_fast(self.pos, self.faces, pressure_val)
            self.pos[self.pinned_mask] = self.pinned_pos

        # 3. Check for Explosion
        if np.isnan(self.pos[0, 0]) or np.abs(self.pos[0, 0]) > 1e5:
            self.is_exploded = True
