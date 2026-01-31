# sim.py

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

from breast.mesh.hemisphere import generate_hemisphere
from breast.models import Point, Spring
from breast.renderer import Renderer
from breast.solver_numpy import NumpyBreastSolver
from breast.types import FACE

# ===============================

# RENDER / WORLD CONSTANTS

# ===============================

WORLD_SCALE = 0.01  # solver units → world units (renderer applies this)

# ===============================

# PHYSICS WORKER

# ===============================


def physics_worker(
    shared_positions: SynchronizedArray[float],
    command_queue: Queue[dict[str, str | np.float32]],
    num_points: int,
    points: list[Point],
    springs: list[Spring],
    faces: FACE,
    target_fps: float,
) -> None:
    print(f"[Physics Worker {os.getpid()}] Starting")

    solver = NumpyBreastSolver(points, springs, faces)

    dt = 1.0 / target_fps
    sub_steps = 20
    running = True
    last_time = time.perf_counter()

    # NumPy view of shared memory (created ONCE)
    shared_buf = np.frombuffer(shared_positions.get_obj(), dtype=np.float32).reshape(
        (num_points, 3)
    )

    while running:
        # -----------------------
        # Commands
        # -----------------------
        while not command_queue.empty():
            cmd = command_queue.get()

            if cmd["type"] == "quit":
                running = False
                break
            elif cmd["type"] == "reset":
                solver = NumpyBreastSolver(points, springs, faces)
            elif cmd["type"] == "set_stiffness" and isinstance(cmd["value"], np.float32):  # pyright: ignore[reportArgumentType]
                solver.stiffness = cmd["value"]
            elif cmd["type"] == "set_pressure" and isinstance(cmd["value"], np.float32):  # pyright: ignore[reportArgumentType]
                solver.pressure_stiffness = cmd["value"]

        # -----------------------
        # Physics
        # -----------------------
        for _ in range(sub_steps):
            solver.update(dt / sub_steps)

        if solver.is_exploded:
            print(f"[Physics Worker {os.getpid()}] Explosion detected — resetting")
            solver = NumpyBreastSolver(points, springs, faces)
            solver.is_exploded = False

        # -----------------------
        # Write WORLD-space positions
        # -----------------------
        shared_buf[:] = solver.pos

        # -----------------------
        # Rate limiting
        # -----------------------
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
    print("Generating hemisphere mesh...")
    rings = 24
    radius = 5.0
    segments = math.ceil(rings * 2)

    points, springs, faces = generate_hemisphere(
        radius=radius,
        rings=rings,
        segments=segments,
    )

    num_points = len(points)
    print(f"Mesh: {num_points} points, {len(springs)} springs, {len(faces)} faces")

    # -----------------------
    # Shared Memory
    # -----------------------
    shared_positions = Array(ctypes.c_float, num_points * 3, lock=True)

    shared_buf = np.frombuffer(shared_positions.get_obj(), dtype=np.float32).reshape(
        (num_points, 3)
    )

    # Initial positions
    initial_pos = np.array(
        [[p.pos.x, p.pos.y, p.pos.z] for p in points],
        dtype=np.float32,
    )
    shared_buf[:] = initial_pos

    command_queue: Queue[dict[str, str | np.float32]] = Queue()

    # -----------------------
    # Physics Processes
    # -----------------------
    physics_fps = 60

    # Detect cores and allocate
    total_cores = 3
    num_physics_workers = max(1, total_cores - 2)
    print(f"Detected {total_cores} cores. Spawning {num_physics_workers} physics workers.")

    physics_processes: list[Process] = []
    for _ in range(num_physics_workers):
        p = Process(
            target=physics_worker,
            args=(
                shared_positions,
                command_queue,
                num_points,
                points,
                springs,
                faces,
                physics_fps,
            ),
            daemon=True,
        )
        p.start()
        physics_processes.append(p)

    # -----------------------
    # Rendering Setup
    # -----------------------
    pygame.init()
    width, height = 1000, 800
    pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("Breast Physics Simulation")

    ctx = moderngl.create_context()
    clock = pygame.time.Clock()

    render_solver = NumpyBreastSolver(points, springs, faces)
    renderer = Renderer(ctx, render_solver, width, height)

    # -----------------------
    # Camera / Visual State
    # -----------------------
    camera_rot = [0.3, 0.0]
    camera_pos = [0.0, 1.5, 4.0]
    visual_scale = 15.0

    # Physics parameters
    current_stiffness = 0.1
    current_pressure = 0.001

    render_fps_target = 60
    running = True

    # -----------------------
    # Main Loop
    # -----------------------
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                # command_queue.put({"type": "quit"}) # Handled in cleanup for all workers

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    # command_queue.put({"type": "quit"}) # Handled in cleanup for all workers

                elif event.key == pygame.K_r:
                    command_queue.put({"type": "reset"})

        # Camera input
        keys = pygame.key.get_pressed()

        # Arrow keys - rotation
        if keys[pygame.K_LEFT]:
            camera_rot[1] -= 0.02
        if keys[pygame.K_RIGHT]:
            camera_rot[1] += 0.02
        if keys[pygame.K_UP]:
            camera_rot[0] -= 0.02
        if keys[pygame.K_DOWN]:
            camera_rot[0] += 0.02

        # WASD - position
        camera_speed = 0.05
        if keys[pygame.K_w]:
            camera_pos[2] -= camera_speed  # Move forward
        if keys[pygame.K_s]:
            camera_pos[2] += camera_speed  # Move back
        if keys[pygame.K_a]:
            camera_pos[0] -= camera_speed  # Move left
        if keys[pygame.K_d]:
            camera_pos[0] += camera_speed  # Move right
        if keys[pygame.K_q]:
            camera_pos[1] -= camera_speed  # Move down
        if keys[pygame.K_e]:
            camera_pos[1] += camera_speed  # Move up

        # Zoom
        if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]:
            visual_scale += 0.5
        if keys[pygame.K_MINUS]:
            visual_scale = max(1.0, visual_scale - 0.5)

        # Physics parameters
        if keys[pygame.K_z]:
            current_stiffness = max(0.01, current_stiffness - 0.01)
            command_queue.put({"type": "set_stiffness", "value": np.float32(current_stiffness)})
        if keys[pygame.K_x]:
            current_stiffness = min(2.0, current_stiffness + 0.01)
            command_queue.put({"type": "set_stiffness", "value": np.float32(current_stiffness)})
        if keys[pygame.K_c]:
            current_pressure = max(0.0, current_pressure - 0.0001)
            command_queue.put({"type": "set_pressure", "value": np.float32(current_pressure)})
        if keys[pygame.K_v]:
            current_pressure = min(0.5, current_pressure + 0.0001)
            command_queue.put({"type": "set_pressure", "value": np.float32(current_pressure)})

        # Read shared positions (WORLD space)
        render_solver.pos[:] = shared_buf

        # Render with all parameters
        renderer.draw(
            render_solver,
            camera_rot,
            camera_pos,
            visual_scale,
            current_stiffness,
            current_pressure,
            clock.get_fps(),
        )

        clock.tick(render_fps_target)

    # -----------------------
    # Cleanup
    # -----------------------
    for _ in range(num_physics_workers):
        command_queue.put({"type": "quit"})

    for p in physics_processes:
        p.join(timeout=1.0)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
