"""
Breast Physics Simulation Package

A soft-body physics simulation for realistic breast physics using
spring-mass systems and volume preservation (water balloon model).
"""

from .models import Point, Spring, Vector3

__version__ = "0.1.0"
__author__ = "Frank1o3"

__all__ = [
    "Vector3",
    "Point",
    "Spring",
]
