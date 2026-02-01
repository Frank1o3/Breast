# models.py
from __future__ import annotations

from collections.abc import Iterable
import math


class Vector3:
    __slots__ = ["x", "y", "z"]

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x, self.y, self.z = x, y, z

    def __iter__(self) -> Iterable[float]:
        yield self.x
        yield self.y
        yield self.z

    def __add__(self, other: Vector3) -> Vector3:
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __iadd__(self, other: Vector3) -> Vector3:
        self.x += other.x
        self.y += other.y
        self.z += other.z
        return self

    def __sub__(self, other: Vector3) -> Vector3:
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __isub__(self, other: Vector3) -> Vector3:
        self.x -= other.x
        self.y -= other.y
        self.z -= other.z
        return self

    def __mul__(self, scalar: float) -> Vector3:
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> Vector3:  # Handles: scalar * vector
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> Vector3:
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def length(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> Vector3:
        length = self.length()
        return self / length if length != 0 else Vector3(0, 0, 0)

    def cross(self, other: Vector3) -> Vector3:
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )


class Point:
    def __init__(self, x: float, y: float, z: float, pinned: bool = False) -> None:
        self.pos = Vector3(x, y, z)
        self.prev_pos = Vector3(x, y, z)
        self.pinned = pinned  # If True, physics won't move this point
        self.mass = 1.0


class Spring:
    def __init__(
        self,
        a: Point,
        b: Point,
        stiffness: float = 0.5,
        rest_length: float | None = None,
    ) -> None:
        self.a = a
        self.b = b
        self.stiffness = stiffness
        if rest_length is None:
            dist = (b.pos - a.pos).length()
            self.rest_length = dist
        else:
            self.rest_length = rest_length
