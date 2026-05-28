"""Nominal robot constants used by the baseline.

Units are meters and radians.  The default table is a UR10-style Modified
Denavit-Hartenberg model and can be replaced with the plant-specific MD-H
table without touching the calibration algorithm.
"""

from __future__ import annotations

import numpy as np

NOMINAL_ROBOT = {
    "base_xyz": [3.335740524, 1.591246404, -0.475598057],
    "base_rpy": [0.0189015, -0.0114514, 1.6069778],
    # [ x: 0.0107601, y: 0.0193033, z: 1.6069822 ]
    "tool_xyz": [0.0, 0.0, 0.0390],
    "tool_rpy": [0.0, 0.0, 0.0],
    "mdh": {
        # T(i-1, i) = Rx(alpha_i) * Tx(a_i) * Rz(theta_i) * Tz(d_i)
        "alpha": [0.0, np.pi / 2.0, 0.0, 0.0, np.pi / 2.0, -np.pi / 2.0],
        "a": [0.0, 0.0, -0.6120, -0.5723, 0.0, 0.0],
        "d": [0.1273, 0.0, 0.0, 0.163941, 0.1157, 0.0922],
        "theta_offset": [0.0] * 6,
    },
    # "mdh": {
    #     "alpha": [0.0, 1.5709710564565065, -0.007090887209183312, -0.002741545047289054, 1.5702174713211476, -1.570245456117715],
    #     "a": [0.0, 0.0015151901499014418, -0.6005990327706019, -0.33753107569020835, 4.5537526658417155e-05, 4.5353975303961505e-05],
    #     "d": [0.12791543913494158, -16.453098345240495, -151.54845568857692, 168.16558279826702, 0.11567668888342877, 0.09226305383537628],
    #     "theta_offset": [-5.857187568830344e-05, 0.19183636241715463, 0.7465611216342868, -0.9383815849822763, -8.260465553727031e-05, -3.4509578391038465e-05],
    # },
    "joint_limits": [
        [-np.pi, np.pi],
        [-2.4, -0.4],
        [-2.6, 0.2],
        [-np.pi, np.pi],
        [-np.pi, np.pi],
        [-np.pi, np.pi],
    ],
}

