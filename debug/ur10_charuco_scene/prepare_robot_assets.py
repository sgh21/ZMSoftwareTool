"""Download the official UR10 CB3 visuals and export their URDF transforms.

Run with the existing ZMSoftware Python environment (PyYAML is required).
All downloaded files stay pinned to the source commit below.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import urllib.request

import yaml


COMMIT = "89bbe795f38a7ab00fb66fe8831dfff79dc99edf"
REPOSITORY = "https://github.com/UniversalRobots/Universal_Robots_ROS2_Description"
RAW = f"https://raw.githubusercontent.com/UniversalRobots/Universal_Robots_ROS2_Description/{COMMIT}/"
DESTINATION = Path(__file__).parent / "assets" / "ur10"
MESH_NAMES = ("base", "shoulder", "upperarm", "forearm", "wrist1", "wrist2", "wrist3")
SOURCE_FILES = [
    "LICENSE",
    "config/ur10/default_kinematics.yaml",
    "config/ur10/visual_parameters.yaml",
    "config/ur10/joint_limits.yaml",
    "config/ur10/physical_parameters.yaml",
    "urdf/ur_macro.xacro",
    "urdf/inc/ur_common.xacro",
    *(f"meshes/ur10/visual/{name}.dae" for name in MESH_NAMES),
]


class RosYamlLoader(yaml.SafeLoader):
    pass


RosYamlLoader.add_constructor(
    "!degrees", lambda loader, node: math.radians(float(loader.construct_scalar(node)))
)


def download(relative_path):
    output = DESTINATION / relative_path
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        with urllib.request.urlopen(RAW + relative_path) as response:
            output.write_bytes(response.read())
    return {"path": relative_path, "url": RAW + relative_path, "bytes": output.stat().st_size}


def origin(values):
    return {
        "xyz": [float(values[k]) for k in ("x", "y", "z")],
        "rpy": [float(values[k]) for k in ("roll", "pitch", "yaw")],
    }


def fixed_joint(name, parent, child, rpy):
    return {
        "name": name, "type": "fixed", "parent": parent, "child": child,
        "origin": {"xyz": [0.0, 0.0, 0.0], "rpy": rpy},
    }


def main():
    with ThreadPoolExecutor(max_workers=6) as executor:
        sources = list(executor.map(download, SOURCE_FILES))
    config = DESTINATION / "config" / "ur10"
    kinematics = yaml.load((config / "default_kinematics.yaml").read_text(), Loader=RosYamlLoader)["kinematics"]
    visuals = yaml.load((config / "visual_parameters.yaml").read_text(), Loader=RosYamlLoader)["mesh_files"]
    links = []
    for name, values in visuals.items():
        links.append({
            "name": "base_link_inertia" if name == "base" else f"{name}_link",
            "visual": {
                "mesh": values["visual"]["mesh"]["path"],
                "origin": origin(values["mesh_offset"]),
                "scale": [1.0, 1.0, 1.0],
            },
        })
    joints = [fixed_joint("base_link-base_link_inertia", "base_link", "base_link_inertia", [0.0, 0.0, math.pi])]
    parent = "base_link_inertia"
    moving_links = ("shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3")
    joint_names = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
    for name, joint_name in zip(moving_links, joint_names):
        child = f"{name}_link"
        joints.append({
            "name": joint_name, "type": "revolute", "parent": parent, "child": child,
            "origin": origin(kinematics[name]), "axis": [0.0, 0.0, 1.0],
        })
        parent = child
    joints.extend([
        fixed_joint("wrist_3-flange", "wrist_3_link", "flange", [0.0, -math.pi / 2, -math.pi / 2]),
        fixed_joint("flange-tool0", "flange", "tool0", [math.pi / 2, 0.0, math.pi / 2]),
    ])
    description = {
        "robot": "Universal Robots UR10 CB3",
        "repository": REPOSITORY,
        "commit": COMMIT,
        "source_tree": f"{REPOSITORY}/tree/{COMMIT}",
        "license": "BSD-3-Clause; source LICENSE included",
        "length_unit": "metre",
        "angle_unit": "radian",
        "convention": {
            "root": "base_link",
            "rpy_matrix": "Rz(yaw) @ Ry(pitch) @ Rx(roll)",
            "joint_transform": "T_parent_child(q) = T_origin @ R_axis(q)",
            "visual_transform": "T_world_visual = T_world_link @ T_visual_origin",
            "note": "Nominal factory model, not a specific robot's kinematic calibration. DAE internal node transforms and embedded materials must be preserved on import.",
        },
        "links": links,
        "joints": joints,
        "sources": sources,
    }
    output = DESTINATION / "robot_description.json"
    output.write_text(json.dumps(description, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output}; {len(links)} visuals, 6 revolute joints, {len(sources)} source files")


if __name__ == "__main__":
    main()
