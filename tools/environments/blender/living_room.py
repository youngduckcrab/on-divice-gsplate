"""A living room built in Blender from code, lit by sun and sky through a big
window, for the viewer's 3D places.

Run with Blender's Python module (pip install bpy, Python 3.11):
  python living_room.py <assets_dir> <out_dir> [preview|bake]

  preview  renders a few camera views with Cycles (for checking the look)
  bake     bakes the lighting (diffuse, direct + indirect) into the vertex
           colours of a densified copy of every surface and writes every
           vertex as a flat splat: <out_dir>/living_room.ply (3DGS layout,
           RDF axes: x right, y down, z forward)

Furniture from the Khronos glTF sample assets (credited in
docs/environments.md): GlamVelvetSofa, SheenChair, ChairDamaskPurplegold,
SpecularSilkPouf, DiffuseTransmissionPlant, GlassVaseFlowers. Everything
else is modelled here. Units are metres; Blender's z is up.
"""
import math
import os
import random
import sys

import bpy  # first: it makes mathutils importable
import mathutils

import place_kit as kit
from place_kit import box, cylinder, nodes, painting, panel, plane, principled, world_position

ASSETS = sys.argv[1]
OUT = sys.argv[2]
MODE = sys.argv[3] if len(sys.argv) > 3 else "preview"
random.seed(7)

W, D, H = 6.4, 5.2, 2.75  # room: x from -W/2, y from -D/2, floor at z = 0
WIN_W, WIN_SILL, WIN_TOP = 2.8, 0.45, 2.35  # window in the +y wall
SPACING = float(os.environ.get("SPACING", "0.024"))  # splat spacing on surfaces (m)


# --- Materials ------------------------------------------------------------------

def plaster(name, color):
    return kit.mottled(name, color)


def oak_floor():
    m = principled("oak floor", (0.45, 0.30, 0.18), 0.35)
    n, l, b = nodes(m)
    pos = world_position(m)
    mp = n.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1.0, 1.0, 1.0)
    brick = n.new("ShaderNodeTexBrick")
    brick.inputs["Scale"].default_value = 1.0
    brick.inputs["Brick Width"].default_value = 1.35
    brick.inputs["Row Height"].default_value = 0.16
    brick.inputs["Mortar Size"].default_value = 0.0025
    brick.inputs["Color1"].default_value = (0.34, 0.20, 0.10, 1)
    brick.inputs["Color2"].default_value = (0.46, 0.29, 0.15, 1)
    brick.inputs["Mortar"].default_value = (0.18, 0.11, 0.06, 1)
    brick.offset = 0.37
    brick.squash = 1.0
    grain = n.new("ShaderNodeTexNoise")
    grain.inputs["Scale"].default_value = 3.0
    grain.inputs["Detail"].default_value = 8.0
    stretch = n.new("ShaderNodeMapping")
    stretch.inputs["Scale"].default_value = (0.6, 30.0, 1.0)
    mix = n.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.inputs["Factor"].default_value = 0.55
    gray = n.new("ShaderNodeValToRGB")
    gray.color_ramp.elements[0].color = (0.75, 0.72, 0.70, 1)
    gray.color_ramp.elements[1].color = (1.1, 1.05, 1.0, 1)
    l.new(pos, mp.inputs["Vector"])
    l.new(mp.outputs["Vector"], brick.inputs["Vector"])
    l.new(pos, stretch.inputs["Vector"])
    l.new(stretch.outputs["Vector"], grain.inputs["Vector"])
    l.new(grain.outputs["Fac"], gray.inputs["Fac"])
    l.new(brick.outputs["Color"], mix.inputs["A"])
    l.new(gray.outputs["Color"], mix.inputs["B"])
    l.new(mix.outputs["Result"], b.inputs["Base Color"])
    return m


def walnut():
    m = principled("walnut", (0.25, 0.14, 0.08), 0.4)
    n, l, b = nodes(m)
    mp = n.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1.0, 14.0, 1.0)
    wave = n.new("ShaderNodeTexWave")
    wave.inputs["Scale"].default_value = 3.0
    wave.inputs["Distortion"].default_value = 4.0
    wave.inputs["Detail"].default_value = 4.0
    ramp = n.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.17, 0.09, 0.05, 1)
    ramp.color_ramp.elements[1].color = (0.33, 0.19, 0.11, 1)
    l.new(world_position(m), mp.inputs["Vector"])
    l.new(mp.outputs["Vector"], wave.inputs["Vector"])
    l.new(wave.outputs["Fac"], ramp.inputs["Fac"])
    l.new(ramp.outputs["Color"], b.inputs["Base Color"])
    return m


def rug(centre, size):
    """A wool rug: a navy border, a thin rust line, a cream field with a soft
    mottle; laid out for a rug of `size` (x, y) at `centre`."""
    m = principled("rug", (0.7, 0.66, 0.6), 1.0, sheen=0.5)
    n, l, b = nodes(m)
    local = n.new("ShaderNodeMapping")  # world -> 0..1 across the rug
    local.inputs["Scale"].default_value = (1 / size[0], 1 / size[1], 1.0)
    local.inputs["Location"].default_value = (0.5 - centre[0] / size[0], 0.5 - centre[1] / size[1], 0.0)
    l.new(world_position(m), local.inputs["Vector"])
    sep = n.new("ShaderNodeSeparateXYZ")
    l.new(local.outputs["Vector"], sep.inputs["Vector"])
    # Distance to the nearest edge, in the rug's 0..1 coordinates (x across 2.6 m, y across 1.8 m).
    def edge(axis, size):
        sub = n.new("ShaderNodeMath")
        sub.operation = "SUBTRACT"
        sub.inputs[1].default_value = 0.5
        l.new(sep.outputs[axis], sub.inputs[0])
        ab = n.new("ShaderNodeMath")
        ab.operation = "ABSOLUTE"
        l.new(sub.outputs[0], ab.inputs[0])
        inv = n.new("ShaderNodeMath")
        inv.operation = "MULTIPLY_ADD"
        inv.inputs[1].default_value = -size
        inv.inputs[2].default_value = 0.5 * size
        l.new(ab.outputs[0], inv.inputs[0])
        return inv
    ex, ey = edge("X", size[0]), edge("Y", size[1])
    near = n.new("ShaderNodeMath")
    near.operation = "MINIMUM"
    l.new(ex.outputs[0], near.inputs[0])
    l.new(ey.outputs[0], near.inputs[1])
    ramp = n.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    ramp.color_ramp.elements[0].color = (0.10, 0.13, 0.22, 1)  # navy border
    ramp.color_ramp.elements[1].position = 0.17
    ramp.color_ramp.elements[1].color = (0.45, 0.20, 0.12, 1)  # rust line
    e = ramp.color_ramp.elements.new(0.20)
    e.color = (0.10, 0.13, 0.22, 1)
    e = ramp.color_ramp.elements.new(0.24)
    e.color = (0.80, 0.76, 0.68, 1)  # cream field
    l.new(near.outputs[0], ramp.inputs["Fac"])
    mottle = n.new("ShaderNodeTexNoise")
    l.new(world_position(m), mottle.inputs["Vector"])
    mottle.inputs["Scale"].default_value = 12.0
    mottle.inputs["Detail"].default_value = 6.0
    tint = n.new("ShaderNodeValToRGB")
    tint.color_ramp.elements[0].color = (0.85, 0.85, 0.85, 1)
    tint.color_ramp.elements[1].color = (1.05, 1.05, 1.05, 1)
    l.new(mottle.outputs["Fac"], tint.inputs["Fac"])
    mix = n.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.inputs["Factor"].default_value = 1.0
    l.new(ramp.outputs["Color"], mix.inputs["A"])
    l.new(tint.outputs["Color"], mix.inputs["B"])
    l.new(mix.outputs["Result"], b.inputs["Base Color"])
    return m


def foliage():
    m = principled("foliage", (0.06, 0.13, 0.04), 0.9)
    n, l, b = nodes(m)
    noise = n.new("ShaderNodeTexNoise")
    l.new(world_position(m), noise.inputs["Vector"])
    noise.inputs["Scale"].default_value = 6.0
    noise.inputs["Detail"].default_value = 10.0
    ramp = n.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.02, 0.05, 0.015, 1)
    ramp.color_ramp.elements[1].color = (0.13, 0.22, 0.06, 1)
    l.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    l.new(ramp.outputs["Color"], b.inputs["Base Color"])
    return m


def tree(x, y, r, leaves):
    """A trunk with a few limbs and a crown of many small leafy clumps (an
    ellipsoid of radius `r`), which reads as a tree from indoors."""
    bark = principled("bark", (0.16, 0.12, 0.09), 0.9)
    rnd = random.Random(int(x * 10 + y))
    top = mathutils.Vector((x, y, r * 1.5))
    kit.rod("garden trunk", (x, y, 0), top, r * 0.07, bark)
    for k in range(4):
        a = k * math.pi / 2 + rnd.uniform(-0.4, 0.4)
        end = top + mathutils.Vector((math.cos(a) * r * 0.6, math.sin(a) * r * 0.6, r * rnd.uniform(0.3, 0.7)))
        kit.rod("garden limb", top - mathutils.Vector((0, 0, r * 0.3)), end, r * 0.035, bark, 8)
    centre = top + mathutils.Vector((0, 0, r * 0.45))
    for k in range(22):
        while True:
            d = mathutils.Vector((rnd.uniform(-1, 1), rnd.uniform(-1, 1), rnd.uniform(-1, 1)))
            if d.length <= 1:
                break
        c = centre + mathutils.Vector((d.x * r, d.y * r, d.z * r * 0.8))
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=r * rnd.uniform(0.28, 0.42), location=c)
        blob = bpy.context.object
        blob.name = "garden crown"
        for v in blob.data.vertices:
            v.co *= 1.0 + 0.18 * math.sin(9 * v.co.x / r + k) * math.cos(7 * v.co.y / r - k) + 0.1 * math.sin(13 * v.co.z / r)
        bpy.ops.object.shade_smooth()
        blob.data.materials.append(leaves)


def lawn():
    m = principled("lawn", (0.16, 0.30, 0.08), 0.9)
    n, l, b = nodes(m)
    noise = n.new("ShaderNodeTexNoise")
    l.new(world_position(m), noise.inputs["Vector"])
    noise.inputs["Scale"].default_value = 0.6
    ramp = n.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.10, 0.22, 0.05, 1)
    ramp.color_ramp.elements[1].color = (0.26, 0.42, 0.12, 1)
    l.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    l.new(ramp.outputs["Color"], b.inputs["Base Color"])
    return m


# --- Scene ------------------------------------------------------------------------

def asset(name, location, yaw_deg=0.0, scale=1.0, drop_materials=()):
    return kit.asset(ASSETS, name, location, yaw_deg, scale, drop_materials)


def build():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    wall = plaster("wall", (0.86, 0.84, 0.80))
    ceiling = plaster("ceiling", (0.90, 0.90, 0.89))
    trim = principled("trim", (0.90, 0.90, 0.88), 0.35)
    frame = principled("window frame", (0.06, 0.06, 0.065), 0.4)
    linen = principled("curtain", (0.88, 0.86, 0.80), 0.95, sheen=0.3)
    metal = principled("black metal", (0.03, 0.03, 0.03), 0.35)
    brass = principled("brass", (0.70, 0.52, 0.25), 0.3)
    shelf_wood = principled("white oak", (0.66, 0.53, 0.38), 0.45)
    book_colors = [(0.55, 0.20, 0.15), (0.15, 0.25, 0.40), (0.80, 0.75, 0.62), (0.20, 0.35, 0.25),
                   (0.70, 0.55, 0.30), (0.30, 0.30, 0.32), (0.85, 0.82, 0.78), (0.45, 0.15, 0.25)]
    books = [principled(f"book {i}", c, 0.6) for i, c in enumerate(book_colors)]

    # Floor and ceiling, walls, the window opening in +y.
    panel("floor", (0, 0, 0), (W, D, 0), "+z", oak_floor())
    panel("ceiling", (0, 0, H), (W, D, 0), "-z", ceiling)
    panel("wall -x", (-W / 2, 0, H / 2), (0, D, H), "+x", wall)
    panel("wall +x", (W / 2, 0, H / 2), (0, D, H), "-x", wall)
    panel("wall -y", (0, -D / 2, H / 2), (W, 0, H), "+y", wall)
    side = (W - WIN_W) / 2
    panel("wall +y left", (-W / 2 + side / 2, D / 2, H / 2), (side, 0, H), "-y", wall)
    panel("wall +y right", (W / 2 - side / 2, D / 2, H / 2), (side, 0, H), "-y", wall)
    panel("wall +y below", (0, D / 2, WIN_SILL / 2), (WIN_W, 0, WIN_SILL), "-y", wall)
    panel("wall +y above", (0, D / 2, (WIN_TOP + H) / 2), (WIN_W, 0, H - WIN_TOP), "-y", wall)
    # The sides of the window opening (the wall is 10 cm thick).
    mid = (WIN_SILL + WIN_TOP) / 2
    panel("wall reveal", (-WIN_W / 2, D / 2 + 0.05, mid), (0, 0.1, WIN_TOP - WIN_SILL), "+x", wall)
    panel("wall reveal", (WIN_W / 2, D / 2 + 0.05, mid), (0, 0.1, WIN_TOP - WIN_SILL), "-x", wall)
    panel("wall reveal", (0, D / 2 + 0.05, WIN_TOP), (WIN_W, 0.1, 0), "-z", wall)
    # Window frame: a border and two mullions, a sill.
    for x in (-WIN_W / 2, WIN_W / 2, -WIN_W / 6, WIN_W / 6):
        box("mullion", (x, D / 2 + 0.05, (WIN_SILL + WIN_TOP) / 2), (0.05, 0.12, WIN_TOP - WIN_SILL), frame)
    for z in (WIN_SILL, WIN_TOP):
        box("transom", (0, D / 2 + 0.05, z), (WIN_W, 0.12, 0.05), frame)
    box("sill", (0, D / 2 - 0.03, WIN_SILL - 0.02), (WIN_W + 0.2, 0.16, 0.04), trim)
    # Baseboards.
    box("base -y", (0, -D / 2 + 0.008, 0.05), (W, 0.016, 0.1), trim)
    box("base -x", (-W / 2 + 0.008, 0, 0.05), (0.016, D, 0.1), trim)
    box("base +x", (W / 2 - 0.008, 0, 0.05), (0.016, D, 0.1), trim)
    # Sheer curtains either side of the window, gently folded.
    for sx in (-1, 1):
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=60, y_subdivisions=40, size=1,
                                        location=(sx * (WIN_W / 2 + 0.15), D / 2 - 0.12, (WIN_TOP + 0.25) / 2 + 0.05),
                                        rotation=(math.radians(90), 0, 0))
        c = bpy.context.object
        c.name = "curtain"
        c.scale = (0.7, WIN_TOP + 0.1, 1)
        bpy.ops.object.transform_apply(scale=True, rotation=True)
        for v in c.data.vertices:
            v.co.y += 0.035 * math.sin((v.co.x - c.location.x) * 2 * math.pi / 0.16)
        c.data.materials.append(linen)

    # Outside: a lawn and a low garden wall beyond.
    plane("lawn", (0, D / 2 + 9, -0.3), (30, 18, 1), lawn())
    box("garden wall", (0, D / 2 + 7, 0.3), (14, 0.3, 1.2), plaster("stone", (0.40, 0.37, 0.33)))
    leaves = foliage()
    box("garden hedge", (0, D / 2 + 6.4, 0.45), (12, 0.9, 1.5), leaves, bevel=0.3)
    for (x, y, r) in ((-4.5, D / 2 + 9.5, 2.0), (1.5, D / 2 + 11, 2.6), (6.0, D / 2 + 8.5, 1.7),
                      (-1.8, D / 2 + 15, 2.4), (4.0, D / 2 + 16, 2.2)):
        tree(x, y, r, leaves)

    # The sofa against the -y wall, a rug, a coffee table with flowers.
    asset("GlamVelvetSofa", (-0.4, -D / 2 + 0.55, 0))
    box("rug", (-0.4, -0.6, 0.005), (2.6, 1.8, 0.01), rug((-0.4, -0.6), (2.6, 1.8)))
    box("table top", (-0.4, -0.85, 0.40), (1.2, 0.6, 0.035), walnut(), bevel=0.008)
    for dx in (-0.55, 0.55):
        for dy in (-0.25, 0.25):
            box("table leg", (-0.4 + dx, -0.85 + dy, 0.19), (0.03, 0.03, 0.38), metal)
    asset("GlassVaseFlowers", (-0.25, -0.85, 0.42), drop_materials=("Glass",))
    asset("SpecularSilkPouf", (1.05, -1.0, 0))
    # Two chairs facing the sofa by the window, the plant in the corner.
    asset("SheenChair", (-1.3, 1.05, 0), yaw_deg=200)
    asset("ChairDamaskPurplegold", (0.6, 1.0, 0), yaw_deg=160)
    asset("DiffuseTransmissionPlant", (2.55, 1.95, 0), scale=1.6)
    # A bookshelf on the +x wall.
    sx = W / 2 - 0.2
    for z in (0.02, 0.42, 0.82, 1.22, 1.62, 2.0):
        box("shelf", (sx, -0.6, z), (0.34, 1.5, 0.03), shelf_wood)
    for y in (-1.35, 0.15):
        box("shelf side", (sx, y, 1.01), (0.34, 0.03, 2.02), shelf_wood)
    for z in (0.04, 0.44, 0.84, 1.24, 1.64):
        y = -1.3
        while y < 0.05:
            w = random.uniform(0.025, 0.05)
            h = random.uniform(0.22, 0.32)
            if random.random() < 0.12:
                y += random.uniform(0.08, 0.2)
                continue
            box("book", (sx + 0.02, y + w / 2, z + h / 2), (0.24, w, h), random.choice(books))
            y += w + 0.002
    # A sideboard on the -x wall with two paintings above, a floor lamp.
    box("sideboard", (-W / 2 + 0.25, -0.2, 0.37), (0.45, 1.8, 0.6), walnut(), bevel=0.01)
    for dy in (-0.75, 0.75):
        box("sideboard leg", (-W / 2 + 0.25, -0.2 + dy, 0.035), (0.04, 0.04, 0.07), metal)
    paintings = [((0.80, 0.72, 0.58), (0.35, 0.40, 0.48), (0.75, 0.45, 0.30), (0.92, 0.90, 0.85)),
                 ((0.90, 0.88, 0.82), (0.20, 0.22, 0.25), (0.62, 0.55, 0.42), (0.55, 0.30, 0.25))]
    for i, (dy, colors) in enumerate(zip((-0.65, 0.35), paintings)):
        box("painting frame", (-W / 2 + 0.02, -0.2 + dy, 1.55), (0.03, 0.66, 0.86), metal)
        box("painting", (-W / 2 + 0.04, -0.2 + dy, 1.55), (0.01, 0.6, 0.8), painting(i * 3.1 + 1, colors))
    # A large painting above the sofa, crown moulding, a pendant over the table.
    box("big painting frame", (-0.4, -D / 2 + 0.015, 1.62), (1.34, 0.03, 0.84), metal)
    box("big painting", (-0.4, -D / 2 + 0.035, 1.62), (1.26, 0.01, 0.76),
        painting(7.3, ((0.86, 0.84, 0.78), (0.42, 0.48, 0.44), (0.80, 0.62, 0.40), (0.25, 0.28, 0.32))))
    for (cx, cy, sx_, sy_) in ((0, -D / 2 + 0.03, W, 0.06), (-W / 2 + 0.03, 0, 0.06, D), (W / 2 - 0.03, 0, 0.06, D)):
        box("crown", (cx, cy, H - 0.04), (sx_, sy_, 0.08), trim)
    cylinder("pendant cord", (-0.4, -0.85, H - 0.35), 0.004, 0.7, metal, 8)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.24, location=(-0.4, -0.85, H - 0.72))
    dome = bpy.context.object
    dome.name = "pendant"
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    for v in dome.data.vertices:
        v.select = v.co.z < -0.01
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.delete(type="VERT")
    bpy.ops.object.mode_set(mode="OBJECT")
    dome.data.materials.append(brass)
    # A round side table at the sofa's end with a small lamp; books on the coffee table.
    cylinder("side table", (1.0, -D / 2 + 0.45, 0.52), 0.24, 0.03, walnut(), 40)
    cylinder("side table stem", (1.0, -D / 2 + 0.45, 0.26), 0.025, 0.5, metal)
    cylinder("side table foot", (1.0, -D / 2 + 0.45, 0.01), 0.16, 0.02, metal, 32)
    cylinder("table lamp base", (1.05, -D / 2 + 0.42, 0.62), 0.06, 0.18, principled("ceramic", (0.82, 0.78, 0.70), 0.3))
    cylinder("table lamp shade", (1.05, -D / 2 + 0.42, 0.83), 0.14, 0.2, linen, 32)
    for k, (h, col) in enumerate(((0.03, books[1]), (0.025, books[2]), (0.02, books[0]))):
        box("coffee book", (-0.75, -0.8, 0.42 + 0.03 * k + h / 2), (0.32 - 0.03 * k, 0.24 - 0.02 * k, h), col)
    cylinder("lamp base", (2.2, -2.0, 0.015), 0.16, 0.03, brass)
    cylinder("lamp pole", (2.2, -2.0, 0.8), 0.012, 1.55, brass, 12)
    cylinder("lamp shade", (2.2, -2.0, 1.6), 0.22, 0.3, linen)


def light():
    scene = bpy.context.scene
    world = bpy.data.worlds.new("sky")
    scene.world = world
    world.use_nodes = True
    sky = world.node_tree.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "MULTIPLE_SCATTERING"
    sky.sun_elevation = math.radians(32)
    sky.sun_rotation = math.radians(160)
    world.node_tree.links.new(sky.outputs["Color"], world.node_tree.nodes["Background"].inputs["Color"])
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.35
    # The sun comes in through the window (from +y), down onto the floor.
    d = mathutils.Vector((0.35, -0.8, -0.5)).normalized()
    bpy.ops.object.light_add(type="SUN")
    sun = bpy.context.object
    sun.rotation_mode = "QUATERNION"
    sun.rotation_quaternion = mathutils.Vector((0, 0, -1)).rotation_difference(d)
    sun.data.energy = 3.2
    sun.data.angle = math.radians(1.5)
    sun.data.color = (1.0, 0.94, 0.86)
    # Ceiling fill: soft, warm, as recessed lights would give.
    bpy.ops.object.light_add(type="AREA", location=(0, -0.3, H - 0.02))
    a = bpy.context.object
    a.data.shape = "RECTANGLE"
    a.data.size = 3.0
    a.data.size_y = 2.0
    a.data.energy = 90
    a.data.color = (1.0, 0.88, 0.75)


def spacing_for(o):
    if o.name.startswith(("lawn", "garden")):
        return 0.15  # outside, seen only through the window
    if o.name.startswith("ceiling"):
        return SPACING * 2  # sparser than the floor, so the viewer finds the floor
    if o.name.startswith("wall"):
        return SPACING * 1.5  # plain plaster
    return SPACING


VIEWS = [((2.6, -2.2, 1.55), (0.0, 0.6, 0.9)), ((-2.6, 2.0, 1.6), (0.6, -1.0, 0.7)),
         ((0.3, 2.1, 1.5), (-0.4, -1.5, 0.6))]

kit.clear()
os.makedirs(OUT, exist_ok=True)
build()
light()
if MODE == "bake":
    kit.bake_and_export(os.path.join(OUT, "living_room.ply"), int(os.environ.get("BAKE_SAMPLES", "96")), spacing_for,
                        median_grey=0.5, expose=lambda o: o.name.startswith("wall"),
                        gain_for=lambda o: 0.2 if o.name.startswith(("lawn", "garden")) else 1.0)
else:
    kit.preview(OUT, VIEWS, exposure=-0.4, samples=32)
