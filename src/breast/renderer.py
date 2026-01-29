from pathlib import Path

import moderngl
import numpy as np
import pygame

from breast.solver_numpy import NumpyBreastSolver


class Renderer:
    def __init__(
        self, ctx: moderngl.Context, solver: NumpyBreastSolver, width: int = 800, height: int = 600
    ) -> None:
        self.ctx = ctx
        self.ctx.enable(moderngl.DEPTH_TEST)

        # Load shaders with geometry shader
        shader_path = Path("src/breast/shaders")
        self.prog = self.ctx.program(
            vertex_shader=shader_path.joinpath("basic.vert").read_text(),
            geometry_shader=shader_path.joinpath("basic.geom").read_text(),
            fragment_shader=shader_path.joinpath("basic.frag").read_text(),
        )

        # Rendering mode: 0 = filled, 1 = wireframe only, 2 = filled with edges
        self.render_mode = 0

        # VBO (Positions ONLY) - Dynamic
        self.vbo = self.ctx.buffer(solver.pos.astype("f4").tobytes())

        # EBO (Triangle Indices ONLY) - Static
        indices = solver.faces.astype("i4").flatten()
        self.ebo = self.ctx.buffer(indices.tobytes())

        # VAO - Single vertex array for everything
        self.vao = self.ctx.vertex_array(
            self.prog, [(self.vbo, "3f", "in_vert")], index_buffer=self.ebo
        )

        print(f"Renderer initialized: {len(solver.faces)} triangles")

    def cycle_render_mode(self):
        """Cycle through render modes: filled -> wireframe -> filled+edges"""
        self.render_mode = (self.render_mode + 1) % 3
        modes = ["Filled", "Wireframe", "Filled + Edges"]
        print(f"Render mode: {modes[self.render_mode]}")
        return modes[self.render_mode]

    def draw(self, solver, camera_rot: tuple[float, float], scale: float) -> None:
        self.ctx.clear(0.1, 0.1, 0.1)

        # Update ONLY positions - single GPU upload per frame
        self.vbo.write(solver.pos.astype("f4").tobytes())

        # Calculate matrices
        mvp, model = self._get_matrices(camera_rot, scale, offset=(0.0, -0.6))

        # Send uniforms
        self.prog["mvp"].write(mvp.T.astype("f4").tobytes())
        self.prog["model"].write(model.T.astype("f4").tobytes())
        self.prog["light_pos"].value = (10.0, 20.0, 10.0)
        self.prog["wireframe_mode"].value = self.render_mode
        self.prog["wireframe_width"].value = 1.5
        self.prog["base_color"].value = (0.9, 0.7, 0.7, 1.0)

        # Single draw call - GPU does all the work
        self.vao.render(moderngl.TRIANGLES)

        pygame.display.flip()

    def _get_matrices(self, rot, scale, offset):
        """Calculate MVP and Model matrices"""
        # Rotation matrices
        cx, sx = np.cos(rot[0]), np.sin(rot[0])
        cy, sy = np.cos(rot[1]), np.sin(rot[1])

        RY = np.array([[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]], dtype="f4")
        RX = np.array([[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1]], dtype="f4")

        # Scale
        s = scale / 100.0
        S = np.diag([s, s, s, 1.0]).astype("f4")

        # Translation
        tx, ty = offset
        T = np.array([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, 0], [0, 0, 0, 1]], dtype="f4")

        # Model matrix (for lighting)
        model = RX @ RY @ S

        # MVP matrix (for projection)
        mvp = T @ model

        return mvp, model
