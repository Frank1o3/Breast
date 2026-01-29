import math
import sys
from multiprocessing import Process, Queue, Array, Value
import ctypes

import moderngl
import numpy as np
import pygame

from breast.mesh.hemisphere import generate_hemisphere
from breast.renderer import Renderer
from breast.solver_numpy import NumpyBreastSolver


def physics_worker(
    shared_positions, shared_lock, command_queue, num_points, points, springs, faces, target_fps
):
    """
    Physics worker process - runs independently at high speed
    """
    print(f"[Physics Worker] Starting on separate process")

    # Create solver
    solver = NumpyBreastSolver(points, springs, faces)

    # Physics loop
    running = True
    frame_count = 0
    dt = 1.0 / target_fps
    sub_steps = 10

    import time

    last_time = time.perf_counter()

    while running:
        # Check for commands from main process
        while not command_queue.empty():
            cmd = command_queue.get()

            if cmd["type"] == "quit":
                running = False
                break
            elif cmd["type"] == "reset":
                solver = NumpyBreastSolver(points, springs, faces)
                print("[Physics Worker] Reset")
            elif cmd["type"] == "set_stiffness":
                solver.stiffness = cmd["value"]
            elif cmd["type"] == "set_pressure":
                solver.pressure_stiffness = cmd["value"]
            elif cmd["type"] == "apply_force":
                idx = cmd["idx"]
                force = cmd["force"]
                if not solver.pinned_mask[idx]:
                    solver.pos[idx] += force * 0.1
                    # Apply to nearby points
                    target_pos = solver.pos[idx]
                    for i in range(len(solver.pos)):
                        if i == idx or solver.pinned_mask[i]:
                            continue
                        dist = np.linalg.norm(solver.pos[i] - target_pos)
                        if dist < 2.0:
                            falloff = 1.0 - (dist / 2.0)
                            solver.pos[i] += force * 0.05 * falloff

        # Physics update
        for _ in range(sub_steps):
            solver.update(dt / sub_steps)

        if solver.is_exploded:
            print("[Physics Worker] Simulation exploded!")
            running = False
            break

        # Write positions to shared memory
        with shared_lock.get_lock():
            np.frombuffer(shared_positions.get_obj()).reshape((num_points, 3))[:] = solver.pos

        # Rate limiting to target FPS
        current_time = time.perf_counter()
        elapsed = current_time - last_time
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        last_time = time.perf_counter()

        frame_count += 1

        # Periodic stats
        if frame_count % (target_fps * 5) == 0:
            actual_fps = 1.0 / (time.perf_counter() - last_time + dt)
            print(f"[Physics Worker] Running at ~{actual_fps:.0f} physics FPS")

    print("[Physics Worker] Shutting down")


def main() -> None:
    # 1. Setup Data (Mesh)
    print("Generating Mesh...")
    rings = 24
    radius = 5.0
    segments = math.ceil(rings * 2)
    points, springs, faces = generate_hemisphere(radius=radius, rings=rings, segments=segments)

    num_points = len(points)

    # 2. Setup shared memory for positions
    # Create a flat array: num_points * 3 floats
    shared_positions = Array(ctypes.c_double, num_points * 3)
    shared_lock = Value("i", 0)  # Simple lock

    # Initialize with starting positions
    pos_array = np.array([[p.pos.x, p.pos.y, p.pos.z] for p in points], dtype=np.float64)
    np.frombuffer(shared_positions.get_obj()).reshape((num_points, 3))[:] = pos_array

    # 3. Setup command queue for main->physics communication
    command_queue = Queue()

    # 4. Start physics worker process
    physics_fps = 120  # Physics runs at 120 FPS
    physics_process = Process(
        target=physics_worker,
        args=(
            shared_positions,
            shared_lock,
            command_queue,
            num_points,
            points,
            springs,
            faces,
            physics_fps,
        ),
    )
    physics_process.start()

    # 5. Initialize Pygame with OpenGL (rendering process)
    width, height = 1000, 800
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("Breast Physics Simulation - Multi-Process")
    ctx = moderngl.create_context()

    # 6. Create a minimal solver just for rendering (won't do physics)
    render_solver = NumpyBreastSolver(points, springs, faces)
    renderer = Renderer(ctx, render_solver, width, height)

    # Simulation state
    running = True
    paused = False
    camera_rot = [0.0, 0.0]
    scale = 15.0
    frame_count = 0

    # Mouse interaction
    mouse_dragging = False
    mouse_force = np.array([0.0, 0.0, 0.0])
    mouse_prev_pos = None
    selected_point_idx = None
    mouse_force_strength = 50.0

    # Parameter tracking (for UI)
    current_stiffness = 0.1
    current_pressure = 0.001
    stiffness_step = 0.001
    pressure_step = 0.00001

    print("\n" + "=" * 60)
    print("MULTI-PROCESS MODE - Physics running at 120 FPS")
    print("=" * 60)
    print("Camera:")
    print("  Arrow Keys      - Rotate camera")
    print("  +/-             - Zoom in/out")
    print("\nSimulation:")
    print("  Space           - Pause/Resume physics")
    print("  R               - Reset simulation")
    print("\nRendering:")
    print("  W               - Cycle modes (Filled/Wireframe/Both)")
    print("\nPhysics Parameters:")
    print("  Q / A           - Increase/Decrease Stiffness")
    print("  E / D           - Increase/Decrease Pressure")
    print("  Shift + Q/A/E/D - 10x faster adjustment")
    print("\nMouse Interaction:")
    print("  Left Click+Drag - Apply force")
    print("  Right Click+Drag- Apply stronger force")
    print("=" * 60)
    print("\nInitial Parameters:")
    print(f"  Stiffness: {current_stiffness:.4f}")
    print(f"  Pressure:  {current_pressure:.5f}")
    print(f"  Physics FPS: {physics_fps}")
    print()

    render_fps_target = 60  # Rendering at 60 FPS is plenty

    while running:
        # Handle Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                command_queue.put({"type": "quit"})

            elif event.type == pygame.KEYDOWN:
                keys_pressed = pygame.key.get_pressed()
                shift_held = keys_pressed[pygame.K_LSHIFT] or keys_pressed[pygame.K_RSHIFT]

                if event.key == pygame.K_SPACE:
                    paused = not paused
                    # Note: In multi-process, pausing is more complex
                    # For now we just stop sending updates
                    print(f"[Rendering {'PAUSED' if paused else 'RESUMED'}]")
                    print("  (Physics worker continues running)")

                elif event.key == pygame.K_w:
                    mode_name = renderer.cycle_render_mode()
                    print(f"[Render Mode: {mode_name}]")

                elif event.key == pygame.K_r:
                    command_queue.put({"type": "reset"})
                    current_stiffness = 0.1
                    current_pressure = 0.001
                    print("[Simulation RESET]")

                # Stiffness
                elif event.key == pygame.K_q:
                    step = stiffness_step * (10 if shift_held else 1)
                    current_stiffness = min(1.0, current_stiffness + step)
                    command_queue.put({"type": "set_stiffness", "value": current_stiffness})
                    print(f"Stiffness: {current_stiffness:.4f}")

                elif event.key == pygame.K_a:
                    step = stiffness_step * (10 if shift_held else 1)
                    current_stiffness = max(0.01, current_stiffness - step)
                    command_queue.put({"type": "set_stiffness", "value": current_stiffness})
                    print(f"Stiffness: {current_stiffness:.4f}")

                # Pressure
                elif event.key == pygame.K_e:
                    step = pressure_step * (10 if shift_held else 1)
                    current_pressure = min(0.1, current_pressure + step)
                    command_queue.put({"type": "set_pressure", "value": current_pressure})
                    print(f"Pressure:  {current_pressure:.5f}")

                elif event.key == pygame.K_d:
                    step = pressure_step * (10 if shift_held else 1)
                    current_pressure = max(0.0, current_pressure - step)
                    command_queue.put({"type": "set_pressure", "value": current_pressure})
                    print(f"Pressure:  {current_pressure:.5f}")

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button in [1, 3]:
                    mouse_dragging = True
                    mouse_prev_pos = np.array(event.pos)
                    # Read current positions for picking
                    with shared_lock.get_lock():
                        current_pos = (
                            np.frombuffer(shared_positions.get_obj())
                            .reshape((num_points, 3))
                            .copy()
                        )
                    render_solver.pos = current_pos
                    selected_point_idx = find_nearest_point(
                        event.pos, render_solver, camera_rot, scale, width, height
                    )

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button in [1, 3]:
                    mouse_dragging = False
                    mouse_prev_pos = None
                    selected_point_idx = None
                    mouse_force = np.array([0.0, 0.0, 0.0])

            elif event.type == pygame.MOUSEMOTION:
                if mouse_dragging and mouse_prev_pos is not None:
                    current_pos = np.array(event.pos)
                    delta = current_pos - mouse_prev_pos
                    mouse_prev_pos = current_pos

                    strength = mouse_force_strength * (2.0 if event.buttons[2] else 1.0)
                    mouse_force = np.array(
                        [delta[0] * strength * 0.01, -delta[1] * strength * 0.01, 0.0]
                    )

                    if selected_point_idx is not None:
                        # Send force command to physics worker
                        rotated_force = rotate_force(mouse_force, camera_rot)
                        command_queue.put(
                            {
                                "type": "apply_force",
                                "idx": selected_point_idx,
                                "force": rotated_force,
                            }
                        )

        # Continuous Input
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            camera_rot[1] -= 0.05
        if keys[pygame.K_RIGHT]:
            camera_rot[1] += 0.05
        if keys[pygame.K_UP]:
            camera_rot[0] -= 0.05
        if keys[pygame.K_DOWN]:
            camera_rot[0] += 0.05
        if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]:
            scale += 0.5
        if keys[pygame.K_MINUS]:
            scale = max(1.0, scale - 0.5)

        # Read positions from shared memory
        with shared_lock.get_lock():
            render_solver.pos = (
                np.frombuffer(shared_positions.get_obj()).reshape((num_points, 3)).copy()
            )

        # Render
        renderer.draw(render_solver, camera_rot, scale)

        # Update window caption
        if frame_count % 30 == 0:
            fps = clock.get_fps()
            modes = ["Filled", "Wireframe", "Filled+Edges"]
            pygame.display.set_caption(
                f"Multi-Process Sim - {modes[renderer.render_mode]} | Render:{fps:.0f} Physics:{physics_fps} FPS | "
                f"Stiff:{current_stiffness:.2f} Press:{current_pressure:.4f}"
            )

        clock.tick(render_fps_target)
        frame_count += 1

    # Cleanup
    print("\n[Main] Shutting down...")
    command_queue.put({"type": "quit"})
    physics_process.join(timeout=2.0)
    if physics_process.is_alive():
        print("[Main] Force terminating physics process")
        physics_process.terminate()

    pygame.quit()
    sys.exit()


def rotate_force(force, cam_rot):
    """Rotate force based on camera rotation"""
    cx, sx = np.cos(cam_rot[0]), np.sin(cam_rot[0])
    cy, sy = np.cos(cam_rot[1]), np.sin(cam_rot[1])

    fx, fy, fz = force
    fx_rot = fx * cy + fz * sy
    fz_rot = -fx * sy + fz * cy
    fy_rot = fy * cx - fz_rot * sx
    fz_final = fy * sx + fz_rot * cx

    return np.array([fx_rot, fy_rot, fz_final])


def find_nearest_point(mouse_pos, solver, camera_rot, scale, width, height):
    """Find nearest non-pinned point to cursor"""
    mx, my = mouse_pos
    min_dist = float("inf")
    nearest = None

    cx, sx = np.cos(camera_rot[0]), np.sin(camera_rot[0])
    cy, sy = np.cos(camera_rot[1]), np.sin(camera_rot[1])

    for i, pos in enumerate(solver.pos):
        if solver.pinned_mask[i]:
            continue

        x, y, z = pos
        x_rot = x * cy + z * sy
        z_rot = -x * sy + z * cy
        y_rot = y * cx - z_rot * sx

        s = scale / 100.0
        sx_pos = width / 2 + x_rot * s * width / 2
        sy_pos = height / 2 - y_rot * s * height / 2

        dist = math.sqrt((sx_pos - mx) ** 2 + (sy_pos - my) ** 2)
        if dist < min_dist:
            min_dist = dist
            nearest = i

    return nearest if min_dist < 50 else None


if __name__ == "__main__":
    main()
