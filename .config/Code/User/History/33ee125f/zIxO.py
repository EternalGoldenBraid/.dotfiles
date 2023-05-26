"""Tests for the optimization_utils module."""

import numpy as np
from numpy.typing import NDArray
import open3d as o3d
import scipy

from optimization_utils import (project, f, optimize,
                                g1, g2, g3, kkt)

        