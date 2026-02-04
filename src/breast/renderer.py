# renderer.py
from pathlib import Path

import moderngl
import numpy as np
import pygame

from breast.models import Vector3
from breast.solver_numpy import UltraStableSolver
from breast.types import PROJ, VIEW

# ------------------------
# Matrix helpers
# ------------------------


def perspective(fov_y: float, aspect: float, near: float, far: float) -> PROJ:
    f = 1.0 / np.tan(fov_y * 0.5)
    return np.array(
        [
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (far + near) / (near - far), (2.0 * far * near) / (near - far)],
            [0.0, 0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    ).T


def rotation_x(angle: float) -> PROJ:
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    ).T


def rotation_y(angle: float) -> PROJ:
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    ).T


def translate(x: float, y: float, z: float) -> PROJ:
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m.T


# ------------------------
# Renderer
# ------------------------


class Renderer:
    def __init__(
        self,
        ctx: moderngl.Context,
        solver: UltraStableSolver,
        width: int = 800,
        height: int = 600,
    ):
        self.ctx = ctx
        self.ctx.enable(moderngl.DEPTH_TEST)

        self.width = width
        self.height = height
        self.solver = solver

        pygame.font.init()
        self.font = pygame.font.SysFont("monospace", 18)

        base = Path(__file__).parent / "shaders"

        # 3D program
        self.prog = self.ctx.program(
            vertex_shader=(base / "mesh.vert").read_text(),
            geometry_shader=(base / "mesh.geom").read_text(),
            fragment_shader=(base / "mesh.frag").read_text(),
        )

        # UI program
        self.ui_prog = self.ctx.program(
            vertex_shader=(base / "ui.vert").read_text(),
            fragment_shader=(base / "ui.frag").read_text(),
        )

        # UI quad (updated every frame)
        self.ui_vbo = self.ctx.buffer(reserve=4 * 4 * 4)
        self.ui_vao = self.ctx.vertex_array(
            self.ui_prog,
            [(self.ui_vbo, "2f 2f", "in_pos", "in_uv")],
        )
        self.ui_texture: moderngl.Texture | None = None

        # Mesh buffers
        self.vbo = self.ctx.buffer(reserve=self.solver.pos.nbytes, dynamic=True)
        self.ebo = self.ctx.buffer(self.solver.faces.astype("i4").ravel().tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, "3f", "in_position")],
            self.ebo,
        )

        print(f"Renderer initialized: {len(self.solver.faces)} triangles")

    # ------------------------
    # Draw
    # ------------------------

    def draw(
        self,
        solver: UltraStableSolver,
        camera_rot: list[float],
        camera_pos: Vector3,
        scale: float,
        stiffness: float,
        pressure: float,
        current_friction: float,
        fps: float,
    ) -> None:
        self.ctx.clear(0.1, 0.1, 0.15, 1.0)

        world_pos = solver.pos * scale
        self.vbo.orphan()
        self.vbo.write(world_pos.astype("f4"))

        view, proj = self._get_matrices(camera_rot, tuple(camera_pos))  # type: ignore
        self.prog["u_view"].write(view.tobytes())  # type: ignore
        self.prog["u_proj"].write(proj.tobytes())  # type: ignore

        light_world = np.array([5.0, 8.0, 3.0, 1.0], dtype=np.float32)
        light_view = (view.T @ light_world)[:3]

        self.prog["u_light_pos_view"].value = tuple(light_view)  # type: ignore
        self.prog["u_light_color"].value = (1.0, 0.95, 0.9)  # type: ignore

        self.vao.render()

        self._draw_ui_overlay(stiffness, pressure, current_friction, fps)
        pygame.display.flip()

    # ------------------------
    # Camera
    # ------------------------

    def _get_matrices(
        self, camera_rot: list[float], camera_pos: tuple[float, float, float]
    ) -> tuple[VIEW, PROJ]:
        pitch, yaw = camera_rot
        cx, cy, cz = camera_pos

        view = translate(-cx, -cy, -cz) @ rotation_y(-yaw) @ rotation_x(-pitch)

        proj = perspective(
            np.radians(60.0),
            self.width / self.height,
            0.1,
            100.0,
        )
        return view, proj

    # ------------------------
    # UI Overlay
    # ------------------------

    def _surface_to_texture(self, surface: pygame.Surface) -> moderngl.Texture:
        surface = pygame.transform.flip(surface, False, True)
        data = pygame.image.tobytes(surface, "RGBA", False)

        tex = self.ctx.texture(surface.get_size(), 4, data)
        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        tex.swizzle = "RGBA"
        return tex

    def _draw_ui_overlay(
        self,
        stiffness: float,
        pressure: float,
        friction: float,
        fps: float,
    ) -> None:
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        lines = [
            f"FPS: {fps:.1f}",
            f"Stiffness: {stiffness:.3f}",
            f"Pressure: {pressure:.3f}",
            f"Friction: {friction:.3f}",
        ]

        line_h = self.font.get_height()
        w = max(self.font.size(line)[0] for line in lines)
        h = line_h * len(lines)

        surface = pygame.Surface((w, h), pygame.SRCALPHA)

        y = 0
        for line in lines:
            surface.blit(self.font.render(line, True, (220, 220, 220)), (0, y))
            y += line_h

        if self.ui_texture:
            self.ui_texture.release()
        self.ui_texture = self._surface_to_texture(surface)
        self.ui_texture.use(0)

        # --- Compute top-left quad ---
        margin = 10
        ndc_w = 2.0 * w / self.width
        ndc_h = 2.0 * h / self.height
        mx = 2.0 * margin / self.width
        my = 2.0 * margin / self.height

        x0 = -1.0 + mx
        y0 = 1.0 - my
        x1 = x0 + ndc_w
        y1 = y0 - ndc_h

        quad = np.array(
            [
                x0,
                y0,
                0.0,
                1.0,
                x0,
                y1,
                0.0,
                0.0,
                x1,
                y0,
                1.0,
                1.0,
                x1,
                y1,
                1.0,
                0.0,
            ],
            dtype="f4",
        )

        self.ui_vbo.write(quad.tobytes())
        self.ui_prog["u_texture"] = 0
        self.ui_vao.render(mode=moderngl.TRIANGLE_STRIP)
