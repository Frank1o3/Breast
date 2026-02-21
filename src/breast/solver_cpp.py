# solver_cpp.py
"""
Drop-in replacement for solver_numpy.py  — uses the compiled C++ _engine module.

Public API is identical to solver_numpy.UltraStableSolver so sim.py,
physics_worker, and renderer.py need zero changes.

Fallback
--------
If _engine is not built yet, imports solver_numpy transparently and warns.
"""

from __future__ import annotations
import warnings

import numpy as np

from breast.models import Point, Spring
from breast.types import FACE

# ─────────────────────────────────────────────────────────────────────────────
# Try compiled extension
# ─────────────────────────────────────────────────────────────────────────────
try:
    import breast._engine as _lib  # type: ignore

    _backend = "cpp"
except ImportError:
    warnings.warn(
        "\n[solver_cpp] C++ extension '_engine' not found — "
        "falling back to solver_numpy.\n"
        "Build with:\n"
        "  cd cpp && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release\n"
        "  cmake --build build\n",
        ImportWarning,
        stacklevel=2,
    )
    from breast.solver_numpy import UltraStableSolver

    _backend = "numpy"

# ─────────────────────────────────────────────────────────────────────────────
# C++ wrapper
# ─────────────────────────────────────────────────────────────────────────────
if _backend == "cpp":

    class UltraStableSolver:
        """
        Soft-body solver backed by the C++ / OpenMP _engine extension.
        Identical public API to solver_numpy.UltraStableSolver.
        """

        def __init__(
            self,
            points:  list[Point],
            springs: list[Spring],
            faces:   FACE,
            gravity: float = -9.8,
            scale:   float = 1.0,
        ) -> None:
            self.faces = faces.astype(np.int32)
            n = len(points)
            s = len(springs)

            # ── points array (N, 4): x  y  z  pinned ────────────────────────
            pts_arr = np.array(
                [[p.pos.x, p.pos.y, p.pos.z, float(p.pinned)] for p in points],
                dtype=np.float32,
            )

            # ── spring index array (S, 2): a  b ─────────────────────────────
            pid = {id(p): i for i, p in enumerate(points)}
            sp_idx = np.array(
                [[pid[id(sp.a)], pid[id(sp.b)]] for sp in springs],
                dtype=np.int32,
            )

            # ── spring float array (S, 2): stiffness  rest_length ────────────
            sp_f = np.array(
                [[sp.stiffness, sp.rest_length] for sp in springs],
                dtype=np.float32,
            )

            # ── construct C++ solver ─────────────────────────────────────────
            self._solver: _lib.Solver = _lib.Solver(
                pts_arr,
                sp_idx,
                sp_f,
                self.faces,
                float(gravity),
                float(scale),
            )

            # ── zero-copy position view ──────────────────────────────────────
            # This (N, 3) float32 view points directly into the C++ buffer.
            self.pos: np.ndarray = self._solver.get_pos()

            # compatibility flags
            self.is_exploded  = False
            self.steps_stable = 0
            self.avg_edge     = float(self._solver.avg_edge)
            self.rest_volume  = float(self._solver.rest_volume)

            print(
                f"[Solver/C++] {n} pts | {s} springs | "
                f"{len(faces)} faces | scale={scale}"
            )
            print(
                f"[Solver/C++] avg_edge={self.avg_edge:.4f}m | "
                f"V0={self.rest_volume:.6f}m³"
            )

        # ── parameters forwarded to C++ ──────────────────────────────────────

        @property
        def stiffness(self) -> float:
            return self._solver.stiffness

        @stiffness.setter
        def stiffness(self, v: float) -> None:
            self._solver.stiffness = float(v)

        @property
        def pressure_stiffness(self) -> float:
            return self._solver.pressure_stiffness

        @pressure_stiffness.setter
        def pressure_stiffness(self, v: float) -> None:
            self._solver.pressure_stiffness = float(v)

        @property
        def damping(self) -> float:
            return self._solver.damping

        @damping.setter
        def damping(self, v: float) -> None:
            self._solver.damping = float(v)

        @property
        def floor_y(self) -> float:
            return self._solver.floor_y

        @floor_y.setter
        def floor_y(self, v: float) -> None:
            self._solver.floor_y = float(v)

        @property
        def iterations(self) -> int:
            return self._solver.iterations

        @iterations.setter
        def iterations(self, v: int) -> None:
            self._solver.iterations = int(v)

        # ── main update ──────────────────────────────────────────────────────

        def update(self, dt: float) -> None:
            if self.is_exploded:
                return

            self._solver.update(dt)

            # Sync flags
            self.is_exploded  = self._solver.is_exploded
            self.steps_stable = self._solver.steps_stable

            # Refresh the numpy view reference (zero-copy, same buffer)
            self.pos = self._solver.get_pos()

            if self.steps_stable % 500 == 0 and self.steps_stable > 0:
                print(
                    f"[Solver/C++] {self.steps_stable} steps stable | "
                    f"V={self.rest_volume:.5f}m³"
                )

