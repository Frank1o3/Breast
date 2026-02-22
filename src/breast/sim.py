"""
Soft Body Sim
"""

import math
import sys

import moderngl
import numpy as np
import pygame

from breast.engine import SimThread
from breast.hemisphere import generate_hemisphere
from breast.models import Vector3
from breast.renderer import Renderer

# ===============================
# CONSTANTS
# ===============================

MESH_TO_METERS = 0.1
PHYSICS_FPS = 120
SUB_STEPS = 5

# ===============================
# MAIN
# ===============================


def main() -> None:
    print("=" * 60)
    print("Soft Body Sim")
    print("=" * 60)

    print("\nGenerating mesh...")
    rings = 40
    segments = 45
    radius = 4.5
    phi_bias = 1.54

    points, springs, faces = generate_hemisphere(
        radius=radius,
        rings=rings,
        segments=segments,
        phi_bias=phi_bias,
        rot_x=math.radians(-15),
    )

    num_points = len(points)
    print(f"Mesh: {num_points} points, {len(springs)} springs, {len(faces)} faces")

    # ── Build numpy arrays for C++ constructors ───────────────────────────────
    pts_arr = np.array(
        [[p.pos.x, p.pos.y, p.pos.z, float(p.pinned)] for p in points],
        dtype=np.float32,
    )

    pid = {id(p): i for i, p in enumerate(points)}
    sp_idx = np.array(
        [[pid[id(sp.a)], pid[id(sp.b)]] for sp in springs],
        dtype=np.int32,
    )
    sp_f = np.array(
        [[sp.stiffness, sp.rest_length] for sp in springs],
        dtype=np.float32,
    )
    faces_arr = faces.astype(np.int32)

    # ── Physics thread ────────────────────────────────────────────────────────
    print(f"\nStarting physics thread ({PHYSICS_FPS} FPS, {SUB_STEPS} substeps)...")
    sim = SimThread(
        pts_arr,
        sp_idx,
        sp_f,
        faces_arr,
        gravity_y=-9.8,
        scale=MESH_TO_METERS,
        physics_fps=PHYSICS_FPS,
        sub_steps=SUB_STEPS,
    )

    # Grab zero-copy position view once — stays valid for the lifetime of sim
    pos = sim.get_pos()

    # Seed targets before starting so the ramp begins at the right value
    target_stiffness = 0.55
    target_pressure = 0.55
    sim.set_stiffness(target_stiffness)
    sim.set_pressure(target_pressure)
    sim.start()

    # ── Renderer ──────────────────────────────────────────────────────────────
    print("Initializing renderer...")
    pygame.init()
    width, height = 1200, 900
    pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("Soft Body Sim")
    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)

    ctx = moderngl.create_context()
    clock = pygame.time.Clock()

    # Renderer only needs faces + the live pos view for VBO uploads
    renderer = Renderer(ctx, pos, faces_arr, width, height)

    camera_rot = [0.0, 0.0]
    camera_pos = Vector3(0.0, 0.1, 0.17)

    # Pre-allocate scaled position buffer — reused every frame, no heap alloc
    world_pos = np.empty((num_points, 3), dtype=np.float32)

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
                    sim.reset()

        keys = pygame.key.get_pressed()

        # ── Mouse look ────────────────────────────────────────────────────────
        mouse_dx, mouse_dy = pygame.mouse.get_rel()
        sensitivity = 0.002
        camera_rot[1] -= mouse_dx * sensitivity
        camera_rot[0] -= mouse_dy * sensitivity
        camera_rot[0] = float(np.clip(camera_rot[0], -1.5, 1.5))

        # ── Camera movement ───────────────────────────────────────────────────
        _, yaw = camera_rot
        forward = Vector3(math.sin(yaw), 0.0, -math.cos(yaw))
        right = Vector3(math.cos(yaw), 0.0, math.sin(yaw))

        speed = 0.002
        if keys[pygame.K_w]:
            camera_pos -= forward * speed
        if keys[pygame.K_s]:
            camera_pos += forward * speed
        if keys[pygame.K_a]:
            camera_pos -= right * speed
        if keys[pygame.K_d]:
            camera_pos += right * speed
        if keys[pygame.K_q]:
            camera_pos.y -= speed
        if keys[pygame.K_e]:
            camera_pos.y += speed

        # ── Parameter targeting ───────────────────────────────────────────────
        changed = False
        if keys[pygame.K_z]:
            target_stiffness -= 0.001
            changed = True
        elif keys[pygame.K_x]:
            target_stiffness += 0.001
            changed = True
        if keys[pygame.K_c]:
            target_pressure -= 0.001
            changed = True
        elif keys[pygame.K_v]:
            target_pressure += 0.001
            changed = True

        if changed:
            sim.set_stiffness(target_stiffness)
            sim.set_pressure(target_pressure)

        # ── Render ────────────────────────────────────────────────────────────
        # pos is a zero-copy view — physics thread writes, render thread reads.
        # One frame of tearing is imperceptible and avoids any lock overhead.
        np.multiply(pos, MESH_TO_METERS, out=world_pos)

        renderer.draw(
            world_pos,
            camera_rot,
            camera_pos,
            target_stiffness,
            target_pressure,
            clock.get_fps(),
        )

        clock.tick(60)

    print("\nShutting down...")
    sim.stop()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
