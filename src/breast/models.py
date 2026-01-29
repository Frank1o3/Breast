# models.py
import math


class Vector3:
    __slots__ = ["x", "y", "z"]

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other: Vector3) -> Vector3:
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector3) -> Vector3:
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: int | float) -> Vector3:
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar: int | float) -> Vector3:
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def length(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> Vector3:
        length = self.length()
        if length == 0:
            return Vector3(0, 0, 0)
        return self / length


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
