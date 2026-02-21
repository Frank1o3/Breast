"""
Soft Body Sim
"""

import ctypes
import math
from multiprocessing import Array, Process, Queue
from multiprocessing.sharedctypes import SynchronizedArray
import os
import sys
import time

import moderngl
import numpy as np
import pygame

from breast.hemisphere import generate_hemisphere
from breast.models import Point, Spring, Vector3
from breast.renderer import Renderer
from breast.solver_cpp import UltraStableSolver
from breast.types import FACE

# ===============================
# CONSTANTS
# ===============================

MESH_TO_METERS = 0.1

PHYSICS_FPS = 120
SUB_STEPS   = 5

# How fast parameters interpolate toward their target value (units/second).
# Smaller  = slower/safer ramp.  Larger = snappier but riskier on big jumps.
STIFFNESS_RATE = 2.5   # stiffness units per second
PRESSURE_RATE  = 2.5   # pressure  units per second

# ===============================
# PHYSICS WORKER
# ===============================


def physics_worker(
    shared_positions: SynchronizedArray[float],
    command_queue: Queue[dict[str, str | float]],
    num_points: int,
    points: list[Point],
    springs: list[Spring],
    faces: FACE,
) -> None:
    """Physics worker — ramps parameters gradually to avoid explosions."""
    print(f"[Physics Worker {os.getpid()}] Starting")

    def make_solver() -> UltraStableSolver:
        return UltraStableSolver(
            points, springs, faces,
            gravity=-9.8,
            scale=MESH_TO_METERS,
        )

    solver = make_solver()

    dt      = 1.0 / PHYSICS_FPS
    sub_dt  = dt / SUB_STEPS
    running = True

    # ── Parameter ramp state ──────────────────────────────────────────────────
    # current_* is what the solver actually sees right now.
    # target_*  is where the user wants to end up.
    # On reset the solver resets to defaults, so current must also reset.
    current_stiffness = solver.stiffness
    current_pressure  = solver.pressure_stiffness
    target_stiffness  = current_stiffness
    target_pressure   = current_pressure


    last_time = time.perf_counter()

    shared_buf = np.frombuffer(
        shared_positions.get_obj(), dtype=np.float32
    ).reshape((num_points, 3))

    while running:
        # ── Commands ─────────────────────────────────────────────────────────
        while not command_queue.empty():
            cmd = command_queue.get()

            if cmd["type"] == "quit":
                running = False
                break

            elif cmd["type"] == "reset":
                solver = make_solver()
                # Reset current to solver defaults so the ramp starts from a
                # safe known state rather than wherever we were before.
                current_stiffness = solver.stiffness
                current_pressure  = solver.pressure_stiffness
                # Keep targets — user wants those values eventually.
                # The ramp will re-approach them safely from the defaults.

            elif cmd["type"] == "set_stiffness":
                target_stiffness = float(cmd["value"])

            elif cmd["type"] == "set_pressure":
                target_pressure = float(cmd["value"])

        if not running:
            break

        # ── Ramp current → target ─────────────────────────────────────────────
        # Move current values toward targets at a fixed rate per second,
        # scaled by the real elapsed dt to stay frame-rate independent.
        real_dt = time.perf_counter() - last_time   # may be slightly off; use dt as fallback
        ramp_dt = min(real_dt, dt * 2)               # clamp runaway spikes

        def ramp(current: float, target: float, rate: float) -> float:
            delta = target - current
            step  = rate * ramp_dt
            if abs(delta) <= step:
                return target
            return current + math.copysign(step, delta)

        current_stiffness = ramp(current_stiffness, target_stiffness, STIFFNESS_RATE)
        current_pressure  = ramp(current_pressure,  target_pressure,  PRESSURE_RATE)

        solver.stiffness          = current_stiffness
        solver.pressure_stiffness = current_pressure

        # ── Physics substeps ──────────────────────────────────────────────────
        for _ in range(SUB_STEPS):
            solver.update(sub_dt)

        # ── Explosion guard ───────────────────────────────────────────────────
        if solver.is_exploded:
            print(f"[Physics Worker {os.getpid()}] Explosion — resetting")
            solver            = make_solver()
            current_stiffness = solver.stiffness
            current_pressure  = solver.pressure_stiffness

        # ── Write positions ───────────────────────────────────────────────────
        shared_buf[:] = solver.pos

        # ── Rate limiting ─────────────────────────────────────────────────────
        elapsed    = time.perf_counter() - last_time
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        last_time = time.perf_counter()

    print(f"[Physics Worker {os.getpid()}] Shutdown")


# ===============================
# MAIN
# ===============================


def main() -> None:
    print("=" * 60)
    print("Soft Body Sim")
    print("=" * 60)

    print("\nGenerating mesh...")
    first_ran = False
    rings    = 40
    segments = 50
    radius   = 4.5
    phi_bias = 1.4

    points, springs, faces = generate_hemisphere(
        radius=radius, rings=rings, segments=segments,
        phi_bias=phi_bias, rot_x=math.radians(-75),
    )

    num_points = len(points)
    print(f"Mesh: {num_points} points, {len(springs)} springs, {len(faces)} faces")

    # Shared memory
    shared_positions = Array(ctypes.c_float, num_points * 3, lock=True)
    shared_buf = np.frombuffer(
        shared_positions.get_obj(), dtype=np.float32
    ).reshape((num_points, 3))

    initial_pos = np.array(
        [[p.pos.x * MESH_TO_METERS, p.pos.y * MESH_TO_METERS, p.pos.z * MESH_TO_METERS]
        for p in points],
        dtype=np.float32,
    )
    shared_buf[:] = initial_pos

    command_queue: Queue[dict[str, str | float]] = Queue()

    print(f"\nStarting physics worker ({PHYSICS_FPS} FPS, {SUB_STEPS} substeps)...")
    physics_process = Process(
        target=physics_worker,
        args=(shared_positions, command_queue, num_points, points, springs, faces),
        daemon=True,
    )
    physics_process.start()

    # ── Renderer ──────────────────────────────────────────────────────────────
    print("Initializing renderer...")
    pygame.init()
    width, height = 1200, 900
    pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("Soft Body Sim")
    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)

    ctx   = moderngl.create_context()
    clock = pygame.time.Clock()

    render_solver = UltraStableSolver(points, springs, faces, scale=MESH_TO_METERS)
    renderer      = Renderer(ctx, render_solver, width, height)

    camera_rot = [0.0, 0.0]
    camera_pos = Vector3(0.0, 0.1, 0.17)

    # ── Target parameters (what the user is dialling toward) ──────────────────
    target_stiffness = 0.55
    target_pressure  = 0.55

    print("\n" + "=" * 60)
    print("CONTROLS")
    print("=" * 60)
    print("Mouse:   Look    | WASD: Move  | Q/E: Up/Down")
    print("Z/X:     Stiffness±  | C/V: Pressure±")
    print("R: Reset | ESC: Quit")
    print("=" * 60)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    first_ran = False
                    command_queue.put({"type": "reset"})

        keys = pygame.key.get_pressed()

        # ── Mouse look ────────────────────────────────────────────────────────
        mouse_dx, mouse_dy = pygame.mouse.get_rel()
        sensitivity   = 0.002
        camera_rot[1] -= mouse_dx * sensitivity
        camera_rot[0] -= mouse_dy * sensitivity
        camera_rot[0]  = float(np.clip(camera_rot[0], -1.5, 1.5))

        # ── Camera movement ───────────────────────────────────────────────────
        _, yaw   = camera_rot
        forward  = Vector3(np.sin(yaw), 0.0, -np.cos(yaw))
        right    = Vector3(np.cos(yaw), 0.0, -np.sin(yaw))
        if np.cos(yaw) < -0.1 or np.cos(yaw) > 0.1:
            forward = Vector3(-forward.x, forward.y, -forward.z)

        speed = 0.002
        if keys[pygame.K_w]: camera_pos += forward * speed
        if keys[pygame.K_s]: camera_pos -= forward * speed
        if keys[pygame.K_a]: camera_pos -= right   * speed
        if keys[pygame.K_d]: camera_pos += right   * speed
        if keys[pygame.K_q]: camera_pos.y -= speed
        if keys[pygame.K_e]: camera_pos.y += speed

        # ── Parameter targeting ───────────────────────────────────────────────
        # We only update the TARGET here.  The physics worker ramps toward it.
        param_changed = not first_ran

        if keys[pygame.K_z]:
            target_stiffness -= 0.001
            param_changed = True
        elif keys[pygame.K_x]:
            target_stiffness += 0.001
            param_changed = True

        if keys[pygame.K_c]:
            target_pressure -= 0.001
            param_changed = True
        elif keys[pygame.K_v]:
            target_pressure += 0.001
            param_changed = True

        if param_changed:
            command_queue.put({"type": "set_stiffness", "value": target_stiffness})
            command_queue.put({"type": "set_pressure",  "value": target_pressure})
            first_ran = True

        # ── Render ────────────────────────────────────────────────────────────
        render_solver.pos[:] = shared_buf

        renderer.draw(
            render_solver,
            camera_rot,
            camera_pos,
            0.1,
            target_stiffness,
            target_pressure,
            clock.get_fps(),
        )

        clock.tick(60)

    print("\nShutting down...")
    command_queue.put({"type": "quit"})
    physics_process.join(timeout=2.0)
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
