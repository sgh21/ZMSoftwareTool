"""Small transform helpers shared by simulation and identification."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def make_transform(xyz: np.ndarray | list[float], rpy: np.ndarray | list[float]) -> np.ndarray:
    """Return a 4x4 transform from xyz and intrinsic xyz Euler angles."""
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = np.asarray(xyz, dtype=float).reshape(3)
    return transform


def modified_dh_transform(alpha: float, a: float, theta: float, d: float) -> np.ndarray:
    """Modified D-H transform Rx(alpha) Tx(a) Rz(theta) Tz(d)."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -sa * d],
            [st * sa, ct * sa, ca, ca * d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def position_from_transform(transform: np.ndarray) -> np.ndarray:
    """Extract xyz position from a homogeneous transform."""
    return np.asarray(transform, dtype=float)[:3, 3].copy()


def test_code():

    T_B2L = [
        [-0.036171,    -0.999159,     0.019305,  3.335740524],
        [0.999280,    -0.036384,    -0.010757,  1.591246404],
        [0.011454,     0.018900,     0.999756,  -0.475598057],
        [0.0, 0.0, 0.0, 1.0],
    ]

    T_B2L = np.array(T_B2L, dtype=float)
    rotation = Rotation.from_matrix(T_B2L[:3, :3])
    rpy = rotation.as_euler("xyz")
    print("rpy:", rpy)

if __name__ == "__main__":
    test_code()

    # T_B2L = [
    #     [-0.036171,    -0.999159,     0.019305,  3.335740524],
    #     [0.999280,    -0.036384,    -0.010757,  1.591246404],
    #     [0.011454,     0.018900,     0.999756,  -0.475598057],
