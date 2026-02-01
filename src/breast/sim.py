# sim_emergency_fix.py
"""
EMERGENCY STABILITY FIX

Changes from your current sim:
1. NO double scaling (this was making mesh invisible!)
2. Ultra-conservative physics parameters
3. Proper initialization (prev_pos = pos)
4. Sequential solving (more stable)
5. Better diagnostics
"""

import ctypes
from multiprocessing import Array, Process, Queue
from multiprocessing.sharedctypes import SynchronizedArray
import os
import sys
import time

import moderngl
import numpy as np
import pygame

from breast.mesh.hemisphere import generate_hemisphere
from breast.models import Point, Spring, Vector3
from breast.renderer import Renderer
from breast.solver_numpy import UltraStableSolver
from breast.types import FACE

# ===============================
# CONSTANTS
# ===============================

# CRITICAL: This is the ONLY place scale is applied
MESH_TO_METERS = 0.1  # 5 units * 0.01 = 0.05m radius (5cm)

# Physics (very conservative)
PHYSICS_FPS = 120
SUB_STEPS = 15  # Many small steps = more stable

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
    """Single physics worker - no race conditions."""
    print(f"[Physics Worker {os.getpid()}] Starting")

    solver = UltraStableSolver(
        points,
        springs,
        faces,
        gravity=-9.8,
        scale=MESH_TO_METERS,
    )

    dt = 1.0 / PHYSICS_FPS
    sub_dt = dt / SUB_STEPS
    running = True
    last_time = time.perf_counter()

    shared_buf = np.frombuffer(shared_positions.get_obj(), dtype=np.float32).reshape(
        (num_points, 3)
    )

    while running:
        # Commands
        while not command_queue.empty():
            cmd = command_queue.get()

            if cmd["type"] == "quit":
                running = False
                break
            elif cmd["type"] == "reset":
                solver = UltraStableSolver(points, springs, faces, scale=MESH_TO_METERS)
            elif cmd["type"] == "set_stiffness":
                solver.stiffness = np.float32(cmd["value"])
            elif cmd["type"] == "set_pressure":
                solver.pressure_stiffness = np.float32(cmd["value"])
            elif cmd["type"] == "set_friction":
                solver.friction = float(cmd["value"])

        # Physics substeps
        for _ in range(SUB_STEPS):
            solver.update(sub_dt)

        # Reset on explosion
        if solver.is_exploded:
            print(f"[Physics Worker {os.getpid()}] Explosion - resetting")
            solver = UltraStableSolver(points, springs, faces, scale=MESH_TO_METERS)

        # Write positions (already in meters!)
        shared_buf[:] = solver.pos

        # Rate limiting
        elapsed = time.perf_counter() - last_time
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
    print("EMERGENCY STABILITY FIX")
    print("=" * 60)

    # Generate mesh
    print("\nGenerating mesh...")
    rings = 24  # Lower resolution for testing
    segments = 24
    radius = 5.0

    points, springs, faces = generate_hemisphere(
        radius=radius,
        rings=rings,
        segments=segments,
    )

    num_points = len(points)
    print(f"Mesh: {num_points} points, {len(springs)} springs, {len(faces)} faces")
    print(f"Scale: {MESH_TO_METERS}m/unit = {radius * MESH_TO_METERS}m radius")

    # Shared memory
    shared_positions = Array(ctypes.c_float, num_points * 3, lock=True)
    shared_buf = np.frombuffer(shared_positions.get_obj(), dtype=np.float32).reshape(
        (num_points, 3)
    )

    # Initial positions (in meters)
    initial_pos = np.array(
        [
            [
                p.pos.x * MESH_TO_METERS,
                p.pos.y * MESH_TO_METERS,
                p.pos.z * MESH_TO_METERS,
            ]
            for p in points
        ],
        dtype=np.float32,
    )
    shared_buf[:] = initial_pos

    command_queue: Queue[dict[str, str | float]] = Queue()

    # Physics worker
    print(f"\nStarting physics worker ({PHYSICS_FPS} FPS, {SUB_STEPS} substeps)...")
    physics_process = Process(
        target=physics_worker,
        args=(shared_positions, command_queue, num_points, points, springs, faces),
        daemon=True,
    )
    physics_process.start()

    # Rendering
    print("Initializing renderer...")
    pygame.init()
    width, height = 1200, 900
    pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("Emergency Stability Fix")
    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)

    ctx = moderngl.create_context()
    clock = pygame.time.Clock()

    render_solver = UltraStableSolver(points, springs, faces, scale=MESH_TO_METERS)
    renderer = Renderer(ctx, render_solver, width, height)

    # Camera (adjusted for smaller mesh)
    camera_rot = [0.0, 0.0]
    camera_pos = Vector3(0.0, 0.1, 0.17)

    # Physics params
    current_stiffness = 0.001
    current_pressure = 0.001
    current_friction = 0.001

    print("\n" + "=" * 60)
    print("CONTROLS")
    print("=" * 60)
    print("Arrows: Rotate | WASD: Move | Q/E: Up/Down")
    print("Z/X: Stiffness | C/V: Pressure | N/M: Friction")
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
                    command_queue.put({"type": "reset"})

        keys = pygame.key.get_pressed()
        mouse_dx, mouse_dy = pygame.mouse.get_rel()
        sensitivity = 0.002
        camera_rot[1] -= mouse_dx * sensitivity  # yaw
        camera_rot[0] -= mouse_dy * sensitivity  # pitch

        # Clamp pitch to prevent flipping (Gimbal lock)
        camera_rot[0] = np.clip(camera_rot[0], -1.5, 1.5)

        # 4. Calculate Directional Vectors
        pitch, yaw = camera_rot
        # Note: These trig functions are based on your specific renderer matrix order
        forward = Vector3(np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch))

        right = Vector3(np.cos(yaw), 0.0, np.sin(yaw))

        # 5. Directional Movement (Fly-cam)
        keys = pygame.key.get_pressed()
        camera_speed = 0.002  # Adjusted for your 0.1m scale

        if keys[pygame.K_w]:
            camera_pos -= forward * camera_speed
        if keys[pygame.K_s]:
            camera_pos += forward * camera_speed
        if keys[pygame.K_a]:
            camera_pos -= right * camera_speed
        if keys[pygame.K_d]:
            camera_pos += right * camera_speed

        # Vertical movement (World-space)
        if keys[pygame.K_q]:
            camera_pos.y -= camera_speed
        if keys[pygame.K_e]:
            camera_pos.y += camera_speed

        # Physics parameters
        if keys[pygame.K_z]:
            current_stiffness = current_stiffness - 0.001
            command_queue.put({"type": "set_stiffness", "value": current_stiffness})
        if keys[pygame.K_x]:
            current_stiffness = current_stiffness + 0.001
            command_queue.put({"type": "set_stiffness", "value": current_stiffness})
        if keys[pygame.K_c]:
            current_pressure = current_pressure - 0.001
            command_queue.put({"type": "set_pressure", "value": current_pressure})
        if keys[pygame.K_v]:
            current_pressure = current_pressure + 0.001
            command_queue.put({"type": "set_pressure", "value": current_pressure})
        if keys[pygame.K_n]:
            current_friction = current_friction - 0.001
            command_queue.put({"type": "set_friction", "value": current_friction})
        if keys[pygame.K_m]:
            current_friction = current_friction + 0.001
            command_queue.put({"type": "set_friction", "value": current_friction})

        # Read positions (already in meters!)
        render_solver.pos[:] = shared_buf

        # CRITICAL: Pass scale=1.0 because pos is already in meters!
        # The old code passed scale=15.0 which then got multiplied by 0.01
        # in the renderer, making it way too small
        renderer.draw(
            render_solver,
            camera_rot,
            camera_pos,
            0.1,  # NO SCALING - already in meters!
            current_stiffness,
            current_pressure,
            current_friction,
            clock.get_fps(),
        )

        clock.tick(60)

    # Cleanup
    print("\nShutting down...")
    command_queue.put({"type": "quit"})
    physics_process.join(timeout=2.0)
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
