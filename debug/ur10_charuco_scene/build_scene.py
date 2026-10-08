"""Build with Blender 4.0: blender -b --factory-startup -P build_scene.py.

OpenCV generates the board assets separately. This script constructs a static
scene and renders three views; it does not implement measurement acquisition.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Euler, Matrix, Vector

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "scene_config.json").read_text(encoding="utf-8"))
BOARD = CFG["charuco"]
PATTERN_SIZE = np.array(BOARD["squares_xy"]) * BOARD["square_length_m"]
ROBOT = json.loads((ROOT / "assets/ur10/robot_description.json").read_text(encoding="utf-8"))
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)


def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    m = Euler(rpy, "XYZ").to_matrix().to_4x4()
    m.translation = Vector(xyz)
    return m


def material(name, color, metallic=0, roughness=0.4):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    return mat


def finish(obj, name, mat, parent=None):
    obj.name = name
    if mat:
        obj.data.materials.clear()
        obj.data.materials.append(mat)
    if parent:
        obj.parent = parent
    return obj


def box(name, xyz, size, mat, bevel=0.001, parent=None):
    bpy.ops.mesh.primitive_cube_add(size=1, location=xyz)
    obj = finish(bpy.context.object, name, mat, parent)
    obj.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel:
        mod = obj.modifiers.new("Machined edges", "BEVEL")
        mod.width = bevel
        mod.segments = 3
        obj.modifiers.new("Face normals", "WEIGHTED_NORMAL")
        obj.data.use_auto_smooth = True
    return obj


def cylinder(name, xyz, radius, depth, mat, parent=None, vertices=64):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth, location=xyz)
    obj = finish(bpy.context.object, name, mat, parent)
    mod = obj.modifiers.new("Edge radius", "BEVEL")
    mod.width = min(0.0007, depth / 5)
    mod.segments = 2
    obj.modifiers.new("Face normals", "WEIGHTED_NORMAL")
    obj.data.use_auto_smooth = True
    for poly in obj.data.polygons:
        poly.use_smooth = len(poly.vertices) == 4
    return obj


def empty(name, parent=None, matrix=None):
    obj = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(obj)
    obj.empty_display_size = 0.05
    obj.parent = parent
    if matrix is not None:
        obj.matrix_basis = matrix
    return obj


def label(name, text, xyz, size, mat, parent=None, rotation=(0, 0, 0)):
    curve = bpy.data.curves.new(name, "FONT")
    curve.body = text
    curve.size = size
    curve.extrude = 0.00002
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.location = xyz
    obj.rotation_euler = rotation
    return finish(obj, name, mat, parent)


def robot(q):
    surface_cache = {}
    frames = {"base_link": empty("UR10 | base_link", matrix=transform(CFG["robot_base_xyz_m"]))}
    k = 0
    for joint in ROBOT["joints"]:
        origin = joint["origin"]
        origin_obj = empty(joint["name"] + " | origin", frames[joint["parent"]],
                           transform(origin["xyz"], origin["rpy"]))
        link = empty("UR10 | " + joint["child"], origin_obj)
        if joint["type"] == "revolute":
            link.rotation_euler.z = float(q[k])
            link["joint_angle_deg"] = float(np.rad2deg(q[k]))
            link["axis"] = "Local Z; rotate this link to adjust the joint"
            k += 1
        frames[joint["child"]] = link
    for link in ROBOT["links"]:
        visual = link["visual"]
        pivot = empty(link["name"] + " | visual origin", frames[link["name"]],
                      transform(visual["origin"]["xyz"], visual["origin"]["rpy"]))
        before = set(bpy.data.objects)
        bpy.ops.wm.collada_import(filepath=str(ROOT / "assets/ur10" / visual["mesh"]))
        imported = set(bpy.data.objects) - before
        for obj in imported:
            if obj.parent not in imported:
                obj.parent = pivot
            if obj.type == "MESH":
                obj.name = "UR10 | " + link["name"] + " | official mesh"
                for face in obj.data.polygons:
                    face.use_smooth = True
                # Preserve CAD material regions. Fresh Blender materials avoid
                # legacy COLLADA surface settings, using sRGB colors in linear space.
                for index, source in enumerate(list(obj.data.materials)):
                    if source:
                        color = tuple(source.diffuse_color[:3])
                        if color not in surface_cache:
                            linear = tuple(c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4 for c in color)
                            blue = color[2] > color[0] * 1.3
                            surface = CFG["robot_surface"]
                            if blue:
                                metal, rough = surface["cap_metallic"], surface["cap_roughness"]
                            elif .15 < max(color) < .55:
                                linear = surface["joint_dark_gray_linear_rgb"]
                                metal, rough = surface["joint_metallic"], surface["joint_roughness"]
                            elif max(color) <= .15:
                                metal, rough = .05, .6
                            else:
                                metal, rough = surface["link_metallic"], surface["link_roughness"]
                            surface_cache[color] = material("UR10 finish " + str(len(surface_cache)), linear, metal, rough)
                            if not blue and .15 < max(color) < .55:
                                shader = surface_cache[color].node_tree.nodes.get("Principled BSDF")
                                shader.inputs["Specular IOR Level"].default_value = .2
                        obj.data.materials[index] = surface_cache[color]
    return frames


def build_table(mats):
    top = CFG["table"]["top_z_m"]
    sx, sy, thick = CFG["table"]["size_m"]
    box("Optical workbench | steel top", (0, 0, top - thick / 2), (sx, sy, thick), mats["steel"], .008)
    box("Table | black edge", (0, 0, top - thick - .008), (sx + .008, sy + .008, .025), mats["black"], .005)
    for x in (-.73, .73):
        for y in (-.39, .39):
            box("Bench leg", (x, y, .34), (.085, .085, .68), mats["black"], .006)
            cylinder("Adjustable foot", (x, y, .018), .059, .036, mats["rubber"])
        box("Lower frame", (x, 0, .17), (.045, .79, .06), mats["black"])
    box("Lower frame cross-member", (0, .36, .21), (1.5, .045, .06), mats["black"])
    # Shallow visual seats for the threaded hole grid, merged into one mesh.
    verts, faces = [], []
    pitch = CFG["table"]["hole_pitch_m"]
    for x in np.arange(-.85, .851, pitch):
        for y in np.arange(-.5, .501, pitch):
            start = len(verts)
            verts.extend([(x + .0028 * math.cos(a), y + .0028 * math.sin(a), top + .00015)
                          for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)])
            faces.append(tuple(range(start, start + 16)))
    mesh = bpy.data.meshes.new("M6 grid face mesh")
    mesh.from_pydata(verts, [], faces)
    obj = bpy.data.objects.new("Optical bench | 50mm hole grid (visual)", mesh)
    bpy.context.collection.objects.link(obj)
    finish(obj, obj.name, mats["black"])
    base = CFG["robot_base_xyz_m"]
    box("UR10 machined base adapter", (base[0], base[1], top + .0125), (.235, .235, .025), mats["aluminium"], .006)
    for dx in (-.096, .096):
        for dy in (-.096, .096):
            cylinder("Base M8 fastener", (base[0] + dx, base[1] + dy, top + .029), .007, .007, mats["black"], vertices=6)
    label("Bench identity", "UR10  /  VISION METROLOGY", (-.78, -.535, .749), .020, mats["white"], rotation=(math.pi / 2, 0, 0))


def boards(mats):
    result = []
    width, height = PATTERN_SIZE
    plate_width, plate_height, _ = CFG["board_plate_size_m"]
    for i, xyz in enumerate(CFG["board_centers_m"]):
        name = f"board_{i + 1:02d}"
        x, y, z = xyz
        mount = empty(name + " | fixed to workbench", matrix=transform(xyz))
        mount["fixed_marker"] = True
        box(name + " | fixture", (0, 0, -.031), CFG["board_fixture_size_m"], mats["black"], .002, mount)
        box(name + " | aluminium backing", (0, 0, -.009), CFG["board_backing_size_m"], mats["aluminium"], .001, mount)
        box(name + " | matte white substrate", (0, 0, -.0027), CFG["board_plate_size_m"], mats["white"], .0005, mount)
        for dx in (-plate_width / 2 + .005, plate_width / 2 - .005):
            for dy in (-plate_height / 2 + .005, plate_height / 2 - .005):
                cylinder(name + " | corner screw", (dx, dy, .0002), .0026, .0012, mats["black"], mount, vertices=6)
        bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, .0001))
        plane = finish(bpy.context.object, name + " | OpenCV ChArUco pattern", None, mount)
        plane.scale = (width, height, 1)
        mat = material(name + " | printed matte pattern", (1, 1, 1), roughness=.88)
        nodes = mat.node_tree.nodes
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(str(ROOT / "assets/charuco" / (name + ".png")))
        tex.interpolation = "Closest"
        mat.node_tree.links.new(tex.outputs["Color"], nodes.get("Principled BSDF").inputs["Base Color"])
        nodes.get("Principled BSDF").inputs["Specular IOR Level"].default_value = .08
        plane.data.materials.append(mat)
        # Text is outside the pattern and does not alter the OpenCV texture.
        label(name + " | ID", f"P{i + 1:02d}  /  FIXED", (-.08, -CFG["board_fixture_size_m"][1] / 2 - .0002, -.031), .006, mats["white"], mount,
              rotation=(math.pi / 2, 0, 0))
        result.append(mount)
    return result


def camera_assembly(tool, mats):
    cc = CFG["camera"]
    tool = empty("Mount | initial rotation about tool Z", tool,
                 Matrix.Rotation(math.radians(cc["mount_z_rotation_deg"]), 4, "Z"))
    tool["initial_mount_z_deg"] = cc["mount_z_rotation_deg"]
    tool["rotation_note"] = "Whole flange adapter, L bracket and camera rotate together about tool0 local Z"
    camera_rig = empty("HIKROBOT | optical assembly", tool,
                       transform(cc["mount_to_optical_center_xyz_m"], (math.pi, 0, 0)))
    camera_rig["reference_model"] = cc["reference_model"]
    camera_rig["geometry_source"] = "Rebuilt to datasheet envelope; not official CAD"
    # Camera local axes: +X image right, +Y image up, -Z viewing direction.
    # Sensor plane z=0, C mount front z=-17.526mm; body extends behind it.
    body = box("HIKROBOT | 29 x 29 x 42 mm housing", (0, 0, .003474), (.029, .029, .042), mats["black"], .001, camera_rig)
    body["size_mm"] = [29, 29, 42]
    box("Camera rear panel", (0, 0, .025), (.028, .028, .003), mats["dark_metal"], .0005, camera_rig)
    cylinder("C mount | 1 inch", (0, 0, -.018), .0127, .003, mats["aluminium"], camera_rig)
    length = cc["lens_length_mm"] / 1000
    radius = cc["lens_diameter_mm"] / 2000
    front = -.017526 - length
    cylinder("MVL-MF1228M | 12 mm lens", (0, 0, -.017526 - length / 2), radius, length, mats["dark_metal"], camera_rig)
    for t in (.006, .020, .031):
        cylinder("Lens focus / aperture ring", (0, 0, -.017526 - t), radius + .0007, .004, mats["rubber"], camera_rig)
    cylinder("Front lens metal retaining ring", (0, 0, front - .0004), radius, .001, mats["aluminium"], camera_rig)
    cylinder("Lens optical glass", (0, 0, front - .001), .0102, .001, mats["glass"], camera_rig)
    # Round adapter + one continuous right-angle plate, bolted to the tool face.
    flange = cylinder("Mount | circular flange plate OD80 x 12mm", (0, 0, .006), .040, .012, mats["aluminium"], tool)
    flange["design_note"] = "Simulation adapter; machining dimensions and bolt interface need confirmation"
    section = [(-.030, .012), (.0605, .012), (.0605, .086),
               (.0545, .086), (.0545, .018), (-.030, .018)]
    verts = [(x, y, z) for y in (-.024, .024) for x, z in section]
    faces = [tuple(reversed(range(6))), tuple(range(6, 12))]
    faces.extend((i, (i + 1) % 6, (i + 1) % 6 + 6, i + 6) for i in range(6))
    mesh = bpy.data.meshes.new("One-piece L bracket profile")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    bracket = bpy.data.objects.new("Mount | one-piece L bracket 6mm", mesh)
    bpy.context.collection.objects.link(bracket)
    finish(bracket, bracket.name, mats["black"], tool)
    bevel = bracket.modifiers.new("L bracket machined edge radius", "BEVEL")
    bevel.width = .0008
    bevel.segments = 3
    bracket.data.use_auto_smooth = True
    bracket.modifiers.new("L bracket normals", "WEIGHTED_NORMAL")
    for angle in (45, 135, 225, 315):
        x, y = .025 * math.cos(math.radians(angle)), .025 * math.sin(math.radians(angle))
        cylinder("Mount | flange bolt washer", (x, y, .0185), .0056, .001, mats["aluminium"], tool)
        cylinder("Mount | flange socket screw head", (x, y, .0215), .0048, .005, mats["dark_metal"], tool)
        cylinder("Mount | flange hex recess", (x, y, .0241), .0025, .0002, mats["rubber"], tool, vertices=6)
    for y in (-.009, .009):
        for z in (.056, .076):
            screw = cylinder("Mount | camera side screw", (.0535, y, z), .0025, .002, mats["aluminium"], tool)
            screw.rotation_euler.y = math.pi / 2
    box("GigE connector", (.004, 0, .032), (.014, .012, .012), mats["aluminium"], .0007, camera_rig)
    label("Camera maker", "HIKROBOT", (-.010, -.01465, -.008), .004, mats["white"], camera_rig, (math.pi / 2, 0, 0))
    curve = bpy.data.curves.new("Camera cable", "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = .003
    curve.bevel_resolution = 3
    spline = curve.splines.new("BEZIER")
    spline.bezier_points.add(4)
    # Keep the free cable routed back toward the arm after changing mount yaw.
    cable_turn = Matrix.Rotation(math.radians(cc["mount_z_rotation_deg"]), 3, "Z")
    for p, co in zip(spline.bezier_points, [(0, 0, .036), (.01, .04, .09), (.06, .06, .15), (.12, .05, .17), (.17, .04, .18)]):
        p.co = cable_turn @ Vector(co)
        p.handle_left_type = p.handle_right_type = "AUTO"
    wire = bpy.data.objects.new("Camera | GigE service loop", curve)
    bpy.context.collection.objects.link(wire)
    finish(wire, wire.name, mats["rubber"], camera_rig)
    data = bpy.data.cameras.new("HIKROBOT | ideal pinhole intrinsics")
    data.lens = cc["focal_length_mm"]
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = cc["resolution_px"][0] * cc["pixel_pitch_um"] / 1000
    data.sensor_height = cc["resolution_px"][1] * cc["pixel_pitch_um"] / 1000
    # Start rays beyond the 57mm housing/lens, as the pinhole is at the sensor.
    data.clip_start = .065
    data.clip_end = 20
    cam = bpy.data.objects.new("WRIST_CAMERA | simulated MV-CS060-10GC", data)
    bpy.context.collection.objects.link(cam)
    cam.parent = camera_rig
    cam["optical_model"] = cc["model_note"]
    return cam, camera_rig


def look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def lighting(mats):
    box("Studio floor", (0, 0, -.055), (200, 200, .08), mats["floor"], 0)
    for name, pos, power, size in [
        ("Key softbox", (-1.5, -1.8, 3.8), 650, 2.7),
        ("Top softbox", (.0, 1.0, 3.5), 800, 2.0),
        ("Rim softbox", (2.0, .5, 2.6), 550, 1.5),
    ]:
        data = bpy.data.lights.new(name, "AREA")
        data.energy = power
        data.shape = "DISK"
        data.size = size
        obj = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(obj)
        obj.location = pos
        look_at(obj, (0, 0, .9))
    world = bpy.data.worlds.new("Soft studio environment")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (.68, .75, .83, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = .35
    bpy.context.scene.world = world


def export_geometry(scene, cam, mounts, q, sensor_distance):
    w, h = CFG["camera"]["resolution_px"]
    scene.camera = cam
    scene.render.resolution_x, scene.render.resolution_y = w, h
    bpy.context.view_layer.update()

    def project(world):
        p = world_to_camera_view(scene, cam, world)
        return [p.x * w, (1 - p.y) * h]

    width, height = PATTERN_SIZE
    plate_width, plate_height, plate_thickness = CFG["board_plate_size_m"]
    info = []
    for i, mount in enumerate(mounts):
        border = [(-width / 2, height / 2), (width / 2, height / 2),
                  (width / 2, -height / 2), (-width / 2, -height / 2)]
        plate_border = [(-plate_width / 2, plate_height / 2), (plate_width / 2, plate_height / 2),
                        (plate_width / 2, -plate_height / 2), (-plate_width / 2, -plate_height / 2)]
        corner_grid = [(-width / 2 + x * BOARD["square_length_m"], height / 2 - y * BOARD["square_length_m"])
                       for y in range(1, BOARD["squares_xy"][1]) for x in range(1, BOARD["squares_xy"][0])]
        info.append({
            "board_id": f"board_{i + 1:02d}", "marker_ids": list(range(i * 100, i * 100 + 31)),
            "projected_pattern_corners_px": [project(mount.matrix_world @ Vector((x, y, .0001))) for x, y in border],
            "projected_plate_corners_px": [project(mount.matrix_world @ Vector((x, y, -.0027 + plate_thickness / 2)))
                                           for x, y in plate_border],
            "projected_charuco_corners_px": [project(mount.matrix_world @ Vector((x, y, .0001))) for x, y in corner_grid],
            "world_matrix_m": [list(row) for row in mount.matrix_world],
        })
    focal_px = CFG["camera"]["focal_length_mm"] / (CFG["camera"]["pixel_pitch_um"] / 1000)
    tool_world = bpy.data.objects["UR10 | tool0"].matrix_world
    tool_to_camera = tool_world.inverted() @ cam.matrix_world
    # Convert the camera basis to OpenCV: X right, Y down, Z forward.
    tool_to_camera_cv = tool_to_camera @ Matrix.Diagonal((1, -1, -1, 1))
    field_width, field_height = w * sensor_distance / focal_px, h * sensor_distance / focal_px
    payload = {
        "image_size_px": [w, h], "target_board_id": "board_01", "boards": info,
        "charuco": BOARD,
        "board_pattern_size_m": PATTERN_SIZE.tolist(),
        "robot_pose_note": "Fixed to the pre-height-change joint angles; board size does not drive robot pose.",
        "K_ideal_px": [[focal_px, 0, w / 2], [0, focal_px, h / 2], [0, 0, 1]],
        "sensor_to_pattern_distance_m": sensor_distance,
        "pattern_plane_field_of_view_m": [field_width, field_height],
        "centered_parallel_plate_translation_margin_m": [(field_width - plate_width) / 2,
                                                         (field_height - plate_height) / 2],
        "margin_condition": "Centered, board-parallel view at the nominal distance; no tilt or in-plane rotation.",
        "tool0_to_camera_opencv_m": [list(row) for row in tool_to_camera_cv],
        "camera_model": CFG["camera"], "joint_angles_deg": np.rad2deg(q).tolist(),
        "robot_source_commit": ROBOT["commit"],
        "camera_world_matrix_blender_m": [list(row) for row in cam.matrix_world],
        "optical_axes": "Blender +X right, +Y up, -Z forward; OpenCV +X right, +Y down, +Z forward",
        "notes": "Nominal synthetic static scene; no acquired measurements or calibrated real-camera intrinsics.",
    }
    (OUT / "geometry.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.name = "01_Workcell_overview"
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "MILLIMETERS"
    mats = {
        "steel": material("Brushed stainless work surface", (.43, .46, .49), .8, .38),
        "aluminium": material("Machined aluminium", (.60, .63, .66), .8, .26),
        "black": material("Black anodised aluminium", (.018, .022, .027), .5, .33),
        "dark_metal": material("Lens barrel", (.03, .035, .043), .7, .25),
        "rubber": material("Rubber", (.012, .014, .018), .05, .6),
        "white": material("Matte white", (.89, .89, .89), 0, .8),
        "glass": material("Coated lens front", (.015, .06, .09), .65, .1),
        "floor": material("Studio floor", (.30, .34, .39), .12, .6),
    }
    build_table(mats)
    mounts = boards(mats)
    cc = CFG["camera"]
    q = np.deg2rad(CFG["robot_joint_angles_deg"])
    frames = robot(q)
    cam, rig = camera_assembly(frames["tool0"], mats)
    bpy.context.view_layer.update()
    pattern_center = mounts[0].matrix_world @ Vector((0, 0, .0001))
    distance = -(cam.matrix_world.inverted() @ pattern_center).z
    lighting(mats)
    data = bpy.data.cameras.new("Scene overview")
    data.lens = CFG["overview_camera"]["lens_mm"]
    overview = bpy.data.objects.new("OVERVIEW | workcell", data)
    bpy.context.collection.objects.link(overview)
    overview.location = CFG["overview_camera"]["xyz_m"]
    look_at(overview, CFG["overview_camera"]["target_m"])
    bpy.context.view_layer.update()
    detail_data = bpy.data.cameras.new("Flange mount close-up")
    detail_data.lens = 60
    detail = bpy.data.objects.new("DETAIL | circular flange and L bracket", detail_data)
    bpy.context.collection.objects.link(detail)
    optical_position = cam.matrix_world.translation
    detail.location = optical_position + Vector((.32, -.40, .075))
    look_at(detail, optical_position + Vector((.035, 0, .030)))
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 16 if args.preview else CFG["render_samples"]
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 6
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = "8"
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = CFG["render_exposure_ev"]
    scene.render.film_transparent = False
    export_geometry(scene, cam, mounts, q, distance)
    scene["README"] = "Static UR10 + MV-CS060-10GC reference + 12mm lens + one fixed OpenCV ChArUco plate. Units m."
    scene["source_repository"] = ROBOT["source_tree"]
    scene.camera = overview
    scene.render.resolution_x, scene.render.resolution_y = CFG["overview_resolution_px"]
    scene.render.image_settings.color_mode = "RGB"
    for obj in rig.children_recursive:
        obj.visible_camera = True
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.region_3d.view_perspective = "CAMERA"
                area.spaces.active.region_3d.view_camera_zoom = 8
                area.spaces.active.overlay.show_overlays = False
                area.spaces.active.shading.type = "MATERIAL"
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = cam
    cam.select_set(True)
    scene.render.filepath = str(OUT / "overview.png")
    sensor_scene = scene.copy()
    sensor_scene.name = "02_Wrist_camera_3072x2048"
    sensor_scene.use_fake_user = True
    sensor_scene.camera = cam
    sensor_scene.render.resolution_x, sensor_scene.render.resolution_y = cc["resolution_px"]
    sensor_scene.render.filepath = str(OUT / "wrist_camera.png")
    detail_scene = scene.copy()
    detail_scene.name = "03_Flange_L_bracket_detail"
    detail_scene.use_fake_user = True
    detail_scene.camera = detail
    detail_scene.render.resolution_x, detail_scene.render.resolution_y = 1400, 1100
    detail_scene.render.filepath = str(OUT / "mount_detail.png")
    scene["view_help"] = "Scene selector: 01 overview; 02 wrist camera; 03 circular flange and L bracket. F12 to render."
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "ur10_charuco_scene.blend"))
    if args.build_only:
        return
    if args.preview:
        scene.render.resolution_percentage = 50
    scene.render.filepath = str(OUT / ("overview_preview.png" if args.preview else "overview.png"))
    bpy.ops.render.render(write_still=True)
    bpy.ops.render.render(write_still=True, scene=detail_scene.name)
    bpy.ops.render.render(write_still=True, scene=sensor_scene.name)
    print("SCENE_COMPLETE", OUT)


if __name__ == "__main__":
    main()
