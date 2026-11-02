"""A photo studio built in Blender from code: a seamless paper backdrop, an
octabox key light, a strip light behind, a background light, a V-flat, a
camera on a tripod and the room around it (a make-up mirror, a clothes rack,
a sofa, a tether cart), for the viewer's 3D places.

Run with Blender's Python module (pip install bpy, Python 3.11):
  python studio.py <assets_dir> <out_dir> [preview|bake]

  preview  renders a few camera views with Cycles (for checking the look)
  bake     bakes the lighting into splats: <out_dir>/studio.ply (see
           place_kit.py)

The sofa is a Khronos glTF sample asset (SheenWoodLeatherSofa, credited in
docs/environments.md); everything else is modelled here. Units are metres;
Blender's z is up. The person stands on the paper at (0, 1).
"""
import math
import os
import random
import sys

import bpy  # first: it makes mathutils importable
import mathutils

import place_kit as kit
from place_kit import box, cylinder, emissive, mottled, painting, panel, principled, rod

ASSETS = sys.argv[1]
OUT = sys.argv[2]
MODE = sys.argv[3] if len(sys.argv) > 3 else "preview"
random.seed(11)

X0, X1, Y0, Y1, H = -4.5, 4.5, -4.0, 3.4, 4.2  # the room; floor at z = 0
SUBJECT = mathutils.Vector((0.0, 1.0, 1.3))  # where a person's chest is
SPACING = float(os.environ.get("SPACING", "0.024"))


def aim(o, target):
    """Turns `o` so its -z axis (a light's or a disc's front) faces `target`."""
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = (mathutils.Vector(target) - o.location).to_track_quat("-Z", "Y")


def hide_from_rays(o):
    """Seen by the camera but lighting nothing (the light object does that)."""
    for attr in ("visible_diffuse", "visible_glossy", "visible_transmission", "visible_volume_scatter",
                 "visible_shadow"):
        setattr(o, attr, False)


def stand(x, y, top, metal, legs=0.45):
    """A light stand: a pole and three splayed legs."""
    rod("stand pole", (x, y, 0.35), (x, y, top), 0.014, metal)
    for k in range(3):
        a = k * 2 * math.pi / 3 + 0.4
        rod("stand leg", (x, y, 0.38), (x + legs * math.cos(a), y + legs * math.sin(a), 0.01), 0.01, metal, 8)


def softbox(name, centre, radius, depth, target, body, glow, metal, sides=8):
    """A softbox: a glowing front facing `target`, a cone-shaped body behind it."""
    front = cylinder(name + " front", centre, radius, 0.01, glow, sides)
    aim(front, target)
    hide_from_rays(front)
    bpy.ops.mesh.primitive_cone_add(vertices=sides, radius1=radius, radius2=radius * 0.2, depth=depth,
                                    location=centre)
    cone = bpy.context.object
    cone.name = name + " body"
    cone.data.materials.append(body)
    aim(cone, target)
    # Back the cone off so its wide end sits at the front.
    cone.location = cone.location - cone.rotation_quaternion @ mathutils.Vector((0, 0, -1)) * (depth / 2 + 0.006)
    return front


def area_light(name, location, target, shape, size, size_y, energy, color=(1.0, 0.97, 0.93)):
    bpy.ops.object.light_add(type="AREA", location=location)
    light = bpy.context.object
    light.name = name
    light.data.shape = shape
    light.data.size = size
    light.data.size_y = size_y
    light.data.energy = energy
    light.data.color = color
    aim(light, target)
    return light


def build():
    paper = mottled("paper", (0.50, 0.49, 0.47), scale=0.8, low=0.97, high=1.02, rough=0.95)
    concrete = mottled("concrete", (0.30, 0.30, 0.29), scale=1.5, low=0.8, high=1.12, rough=0.45)
    charcoal = mottled("charcoal paint", (0.09, 0.09, 0.10), scale=1.0, low=0.9, high=1.08)
    ceiling = principled("ceiling", (0.04, 0.04, 0.045), 0.9)
    metal = principled("black metal", (0.03, 0.03, 0.03), 0.35)
    chrome = principled("chrome", (0.55, 0.56, 0.58), 0.25)
    fabric = principled("black fabric", (0.02, 0.02, 0.02), 0.9)
    white_board = principled("white foam", (0.90, 0.90, 0.89), 0.9)
    pine = principled("pine", (0.62, 0.45, 0.28), 0.6)
    softlight = emissive("diffuser", (1.0, 0.98, 0.95), 12.0)

    # The room.
    cx, cy, w, d = (X0 + X1) / 2, (Y0 + Y1) / 2, X1 - X0, Y1 - Y0
    # The floor, with a hole where the paper lies on it (hidden there, it
    # would show through between the paper's splats).
    px, py0, py1 = 1.36, -0.4, 2.6
    panel("floor", (cx, (Y0 + py0) / 2, 0), (w, py0 - Y0, 0), "+z", concrete)
    panel("floor", (cx, (py1 + Y1) / 2, 0), (w, Y1 - py1, 0), "+z", concrete)
    panel("floor", ((X0 - px) / 2, (py0 + py1) / 2, 0), (-px - X0, py1 - py0, 0), "+z", concrete)
    panel("floor", ((px + X1) / 2, (py0 + py1) / 2, 0), (X1 - px, py1 - py0, 0), "+z", concrete)
    panel("ceiling", (cx, cy, H), (w, d, 0), "-z", ceiling)
    panel("wall -x", (X0, cy, H / 2), (0, d, H), "+x", charcoal)
    panel("wall +x", (X1, cy, H / 2), (0, d, H), "-x", charcoal)
    panel("wall -y", (cx, Y0, H / 2), (w, 0, H), "+y", charcoal)
    panel("wall +y", (cx, Y1, H / 2), (w, 0, H), "-y", charcoal)
    # A door in the -y wall, pipes under the ceiling.
    box("door", (3.2, Y0 + 0.02, 1.05), (1.0, 0.04, 2.1), principled("door", (0.18, 0.18, 0.19), 0.5))
    box("door handle", (2.8, Y0 + 0.06, 1.0), (0.14, 0.03, 0.025), chrome)
    for x in (-3.0, -1.0, 1.0, 3.0):
        rod("ceiling pipe", (x, Y0, H - 0.3), (x, Y1, H - 0.3), 0.025, metal)
    rod("ceiling pipe", (X0, 0.0, H - 0.35), (X1, 0.0, H - 0.35), 0.03, metal)

    # Framed prints on the walls.
    prints = [((0.82, 0.80, 0.76), (0.30, 0.33, 0.38), (0.70, 0.52, 0.36), (0.12, 0.12, 0.13)),
              ((0.90, 0.88, 0.84), (0.55, 0.20, 0.16), (0.25, 0.27, 0.30), (0.78, 0.72, 0.60)),
              ((0.20, 0.30, 0.36), (0.80, 0.78, 0.72), (0.42, 0.50, 0.46), (0.95, 0.93, 0.88))]
    for k, (x, w, h) in enumerate(((-3.1, 0.9, 1.2), (-1.7, 0.9, 1.2), (0.6, 1.5, 1.0))):
        box("print frame", (x, Y0 + 0.02, 1.75), (w + 0.06, 0.03, h + 0.06), metal)
        box("print", (x, Y0 + 0.04, 1.75), (w, 0.01, h), painting(2.3 * k + 0.7, prints[k]))
    box("print frame", (X1 - 0.02, -2.3, 1.75), (0.03, 1.66, 1.06), metal)
    box("print", (X1 - 0.04, -2.3, 1.75), (0.01, 1.6, 1.0), painting(9.1, prints[0]))

    # The paper backdrop: hanging from a roll, sweeping onto the floor.
    width, back, top, bend, front = 2 * px, py1, 2.85, 0.6, py0
    profile = []
    for i in range(12):
        z = top - (top - bend) * i / 12
        profile.append((back, z))
    for i in range(13):
        a = (math.pi / 2) * i / 12
        profile.append((back - bend + bend * math.cos(a), bend - bend * math.sin(a) + 0.002))
    for i in range(1, 13):
        profile.append((back - bend - (back - bend - front) * i / 12, 0.002))
    mesh = bpy.data.meshes.new("paper")
    verts, faces = [], []
    for k, (y, z) in enumerate(profile):
        verts += [(-width / 2, y, z), (width / 2, y, z)]
        if k:
            faces.append((2 * k - 2, 2 * k, 2 * k + 1, 2 * k - 1))  # facing the set
    mesh.from_pydata(verts, [], faces)
    o = bpy.data.objects.new("paper", mesh)
    bpy.context.collection.objects.link(o)
    o.data.materials.append(paper)
    cylinder("paper roll", (0, back + 0.07, top + 0.06), 0.08, width + 0.02, paper, 32, (0, math.pi / 2, 0))
    rod("backdrop bar", (-1.55, back + 0.07, top + 0.17), (1.55, back + 0.07, top + 0.17), 0.016, metal)
    for x in (-1.5, 1.5):
        stand(x, back + 0.12, top + 0.2, metal)

    # Key: an octabox front left, high, aimed at the subject.
    key_at = mathutils.Vector((-1.7, -0.3, 2.05))
    softbox("octabox", key_at, 0.6, 0.55, SUBJECT, fabric, softlight, metal)
    stand(-2.05, -0.62, 1.85, metal, 0.6)
    rod("octabox arm", (-2.05, -0.62, 1.85), key_at, 0.015, metal)
    area_light("key", key_at + (SUBJECT - key_at).normalized() * 0.03, SUBJECT, "DISK", 1.2, 1.2, 420)
    # Rim: a strip box behind right.
    rim_at = mathutils.Vector((1.55, 2.05, 1.55))
    strip = box("strip front", rim_at, (0.3, 0.01, 1.4), softlight)
    aim_y = (SUBJECT - rim_at)
    strip.rotation_euler[2] = math.atan2(aim_y.y, aim_y.x) - math.pi / 2
    hide_from_rays(strip)
    body = box("strip body", rim_at - aim_y.normalized() * 0.13, (0.34, 0.24, 1.44), fabric)
    body.rotation_euler[2] = strip.rotation_euler[2]
    stand(1.75, 2.35, 1.0, metal)
    area_light("rim", rim_at + aim_y.normalized() * 0.02, SUBJECT, "RECTANGLE", 0.3, 1.4, 140)
    # Background light: a small reflector on the floor behind the person, lighting the paper.
    bg_at = mathutils.Vector((0.0, 1.75, 0.35))
    head = cylinder("bg light head", bg_at, 0.09, 0.2, metal, 20)
    aim(head, (0, back, 1.6))
    stand(0.0, 1.95, 0.3, metal, 0.25)
    bpy.ops.object.light_add(type="SPOT", location=bg_at + mathutils.Vector((0, 0.1, 0.05)))
    spot = bpy.context.object
    spot.data.energy = 260
    spot.data.spot_size = math.radians(75)
    spot.data.spot_blend = 0.8
    spot.data.shadow_soft_size = 0.05
    spot.data.color = (1.0, 0.95, 0.88)
    aim(spot, (0, back, 1.7))

    # Fill: a V-flat front right, white inside.
    hinge = mathutils.Vector((2.25, -0.25, 1.2))
    for a in (math.radians(100), math.radians(190)):
        d = mathutils.Vector((math.cos(a), math.sin(a), 0)) * 0.6
        board = box("vflat white", hinge + d, (1.2, 0.02, 2.4), white_board)
        board.rotation_euler[2] = a
        normal = mathutils.Vector((-math.sin(a), math.cos(a), 0))
        if normal.dot(SUBJECT - hinge) > 0:
            normal = -normal  # the black side faces away from the set
        backing = box("vflat black", hinge + d + normal * 0.03, (1.2, 0.04, 2.4), fabric)
        backing.rotation_euler[2] = a

    # Camera on a tripod, facing the set.
    cam = mathutils.Vector((0.0, -2.6, 1.45))
    for k in range(3):
        a = k * 2 * math.pi / 3 + math.pi / 2
        rod("tripod leg", cam - mathutils.Vector((0, 0, 0.08)),
            (cam.x + 0.5 * math.cos(a), cam.y + 0.5 * math.sin(a), 0.01), 0.013, metal, 8)
    box("tripod head", cam - mathutils.Vector((0, 0, 0.04)), (0.08, 0.08, 0.06), metal)
    box("camera body", cam + mathutils.Vector((0, 0, 0.05)), (0.15, 0.09, 0.11), metal, bevel=0.01)
    cylinder("camera lens", cam + mathutils.Vector((0, 0.13, 0.04)), 0.042, 0.16, metal, 24, (math.pi / 2, 0, 0))
    # A tether cart with a laptop.
    cx, cy = -1.4, -2.9
    box("cart top", (cx, cy, 0.95), (0.65, 0.45, 0.03), metal)
    box("cart shelf", (cx, cy, 0.3), (0.65, 0.45, 0.03), metal)
    for dx in (-0.3, 0.3):
        for dy in (-0.2, 0.2):
            rod("cart leg", (cx + dx, cy + dy, 0.05), (cx + dx, cy + dy, 0.95), 0.012, chrome, 8)
            cylinder("cart wheel", (cx + dx, cy + dy, 0.035), 0.035, 0.025, metal, 12, (0, math.pi / 2, 0))
    box("laptop base", (cx, cy, 0.975), (0.34, 0.24, 0.018), chrome)
    screen = box("laptop screen", (cx, cy + 0.12, 1.08), (0.34, 0.012, 0.22), emissive("screen", (0.75, 0.82, 0.95), 2.0))
    screen.rotation_euler[0] = math.radians(-15)
    # Apple boxes by the paper.
    for k, (x, y, z, r) in enumerate(((1.25, 0.15, 0.1, 0.1), (1.27, 0.12, 0.3, -0.05), (-1.3, -0.1, 0.1, 0.4))):
        b = box("apple box", (x, y, z), (0.5, 0.3, 0.2), pine, bevel=0.006)
        b.rotation_euler[2] = r

    # The make-up corner on the -x wall: a table, a mirror ringed with bulbs, a stool.
    mx, my = X0 + 0.3, -1.6
    box("makeup table", (mx, my, 0.74), (0.5, 1.6, 0.04), principled("white table", (0.85, 0.84, 0.82), 0.4))
    for dy in (-0.75, 0.75):
        box("makeup table side", (mx, my + dy, 0.37), (0.5, 0.03, 0.74), principled("white table", (0.85, 0.84, 0.82), 0.4))
    box("mirror", (X0 + 0.015, my, 1.45), (0.02, 1.1, 0.85), principled("mirror", (0.20, 0.21, 0.22), 0.15))
    bulb = emissive("bulb", (1.0, 0.85, 0.62), 6.0)
    for k in range(7):
        y = my - 0.6 + 1.2 * k / 6
        for z in (1.0, 1.9):
            bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=0.035, location=(X0 + 0.05, y, z))
            bpy.context.object.name = "bulb"
            bpy.context.object.data.materials.append(bulb)
    for k in range(1, 4):
        for y in (my - 0.6, my + 0.6):
            bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=0.035,
                                                 location=(X0 + 0.05, y, 1.0 + 0.9 * k / 4))
            bpy.context.object.name = "bulb"
            bpy.context.object.data.materials.append(bulb)
    bpy.ops.object.light_add(type="AREA", location=(X0 + 0.15, my, 1.45))
    glow = bpy.context.object
    glow.data.size, glow.data.size_y, glow.data.shape = 1.3, 1.0, "RECTANGLE"
    glow.data.energy = 60
    glow.data.color = (1.0, 0.85, 0.65)
    aim(glow, (0, my, 1.2))
    cylinder("stool seat", (mx + 0.75, my + 0.1, 0.62), 0.18, 0.05, principled("stool", (0.12, 0.10, 0.09), 0.5), 24)
    for k in range(4):
        a = k * math.pi / 2 + 0.3
        rod("stool leg", (mx + 0.75 + 0.12 * math.cos(a), my + 0.1 + 0.12 * math.sin(a), 0.6),
            (mx + 0.75 + 0.2 * math.cos(a), my + 0.1 + 0.2 * math.sin(a), 0.0), 0.012, chrome, 8)

    # A clothes rack on the +x side.
    rx, ry = X1 - 0.8, 0.9
    rod("rack rail", (rx, ry - 0.8, 1.6), (rx, ry + 0.8, 1.6), 0.014, chrome)
    for dy in (-0.8, 0.8):
        rod("rack upright", (rx, ry + dy, 0.05), (rx, ry + dy, 1.6), 0.014, chrome)
        rod("rack foot", (rx - 0.25, ry + dy, 0.03), (rx + 0.25, ry + dy, 0.03), 0.016, chrome)
    colors = [(0.75, 0.70, 0.62), (0.12, 0.13, 0.18), (0.55, 0.18, 0.15), (0.85, 0.84, 0.82), (0.20, 0.32, 0.30),
              (0.62, 0.50, 0.35), (0.10, 0.10, 0.10)]
    y = ry - 0.65
    for k, c in enumerate(colors):
        length = random.uniform(0.75, 1.05)
        box("garment", (rx, y, 1.56 - length / 2), (0.42, 0.06, length), principled(f"cloth {k}", c, 0.85, 0.3),
            bevel=0.02)
        y += random.uniform(0.17, 0.23)

    # A sofa for clients along the +x wall, facing the set.
    kit.asset(ASSETS, "SheenWoodLeatherSofa", (X1 - 0.55, -2.3, 0), yaw_deg=90)
    # Cables from the lights to the wall.
    for (a, b) in (((-2.05, -0.62), (X0, -0.2)), ((1.75, 2.35), (X1, 2.6)), ((0.0, 1.95), (1.0, Y1))):
        curve = bpy.data.curves.new("cable", "CURVE")
        curve.dimensions = "3D"
        curve.bevel_depth = 0.006
        spline = curve.splines.new("BEZIER")
        spline.bezier_points.add(2)
        mid = ((a[0] + b[0]) / 2 + random.uniform(-0.4, 0.4), (a[1] + b[1]) / 2 + random.uniform(-0.4, 0.4))
        for p, xy in zip(spline.bezier_points, (a, mid, b)):
            p.co = (xy[0], xy[1], 0.006)
            p.handle_left_type = p.handle_right_type = "AUTO"
        o = bpy.data.objects.new("cable", curve)
        bpy.context.collection.objects.link(o)
        for other in bpy.context.selected_objects:
            other.select_set(False)
        bpy.context.view_layer.objects.active = o
        o.select_set(True)
        bpy.ops.object.convert(target="MESH")
        o.select_set(False)
        o.data.materials.append(metal)


def light():
    scene = bpy.context.scene
    world = bpy.data.worlds.new("dark")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)
    # Dim house lights, so the room around the set is not black.
    for x in (-2.5, 2.5):
        area_light("house", (x, -1.0, H - 0.4), (x, -1.0, 0), "RECTANGLE", 2.0, 3.0, 35, (1.0, 0.92, 0.82))


def spacing_for(o):
    name = o.name
    if name.startswith(("ceiling", "wall")):
        return SPACING * 3  # dark and plain; sparser than the floor
    if name.startswith("floor"):
        return SPACING * 1.4
    return SPACING


VIEWS = [((0.0, -3.4, 1.5), (0.0, 1.5, 1.0)), ((3.8, 2.9, 1.9), (-0.8, -0.8, 1.0)),
         ((-0.6, 1.9, 1.6), (1.2, -3.0, 1.0))]

kit.clear()
os.makedirs(OUT, exist_ok=True)
build()
light()
if MODE == "bake":
    kit.bake_and_export(os.path.join(OUT, "studio.ply"), int(os.environ.get("BAKE_SAMPLES", "128")), spacing_for,
                        median_grey=0.3, expose=lambda o: o.name.startswith("paper"))
else:
    kit.preview(OUT, VIEWS, exposure=-1.0, samples=32)
