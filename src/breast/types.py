import numpy as np

from breast.models import Point, Spring

FACE = np.typing.NDArray[np.int32]
GEN_HEMI = tuple[list[Point], list[Spring], FACE]
VIEW = np.typing.NDArray[np.float64]
PROJ = np.typing.NDArray[np.float32]
