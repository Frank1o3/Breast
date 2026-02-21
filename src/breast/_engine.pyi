"""
softsim engine — soft-body physics (C++ / OpenMP)
"""
from __future__ import annotations
import numpy
import numpy.typing
import typing
__all__: list[str] = ['PointInit', 'Solver', 'SpringInit', 'calc_mesh_volume']
class PointInit:
    pinned: bool
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def x(self) -> float:
        ...
    @x.setter
    def x(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def y(self) -> float:
        ...
    @y.setter
    def y(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def z(self) -> float:
        ...
    @z.setter
    def z(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class Solver:
    """
    Jacobi soft-body solver with Verlet integration and pressure.
    
    Constructor arrays
    ------------------
    points_array   (N,4) float32 : x  y  z  pinned
    spring_indices (S,2) int32   : vertex_a  vertex_b
    spring_floats  (S,2) float32 : stiffness  rest_length
    faces_array    (F,3) int32   : v0  v1  v2
    """
    def __init__(self, points_array: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], spring_indices: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], spring_floats: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], faces_array: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], gravity_y: typing.SupportsFloat | typing.SupportsIndex = -9.800000190734863, scale: typing.SupportsFloat | typing.SupportsIndex = 1.0) -> None:
        ...
    def get_pos(self) -> numpy.typing.NDArray[numpy.float32]:
        """
        Return zero-copy (N,3) float32 view of current positions.
        """
    def num_faces(self) -> int:
        ...
    def num_points(self) -> int:
        ...
    def num_springs(self) -> int:
        ...
    def set_pos(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> None:
        """
        Overwrite positions with a (N,3) float32 array.
        """
    def update(self, dt: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Advance simulation by dt seconds (one sub-step).
        """
    @property
    def avg_edge(self) -> float:
        ...
    @property
    def damping(self) -> float:
        ...
    @damping.setter
    def damping(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def floor_y(self) -> float:
        ...
    @floor_y.setter
    def floor_y(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def gravity_y(self) -> float:
        ...
    @gravity_y.setter
    def gravity_y(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def is_exploded(self) -> bool:
        ...
    @property
    def iterations(self) -> int:
        ...
    @iterations.setter
    def iterations(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def pressure_stiffness(self) -> float:
        ...
    @pressure_stiffness.setter
    def pressure_stiffness(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def rest_volume(self) -> float:
        ...
    @property
    def steps_stable(self) -> int:
        ...
    @property
    def stiffness(self) -> float:
        ...
    @stiffness.setter
    def stiffness(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class SpringInit:
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def a(self) -> int:
        ...
    @a.setter
    def a(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def b(self) -> int:
        ...
    @b.setter
    def b(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def rest_length(self) -> float:
        ...
    @rest_length.setter
    def rest_length(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def stiffness(self) -> float:
        ...
    @stiffness.setter
    def stiffness(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
def calc_mesh_volume(pos: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], faces: typing.Annotated[numpy.typing.ArrayLike, numpy.int32]) -> float:
    """
    Compute signed mesh volume (divergence theorem).
    """
