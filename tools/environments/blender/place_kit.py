"""Shared pieces for the places built in Blender (living_room.py, studio.py):
materials, simple geometry, Khronos sample assets, Cycles previews, and the
bake of the lighting into splats.

How a place becomes splats: every surface is densified until no edge is
longer than the splat spacing, Cycles bakes the diffuse light (direct +
indirect, times the surface colour) into the vertex colours, and every
vertex becomes one flat Gaussian lying in the surface, sized to the edges
around it. Glowing surfaces (softboxes, lamp shades) take their emission
instead. Colours are tone mapped (ACES) to display values, as captured
splats are.
"""
import math
import os
import time

import bpy  # first: it makes bmesh and mathutils importable
import bmesh
import mathutils
import numpy as np


# --- Materials ------------------------------------------------------------------

def principled(name, color, rough=0.6, sheen=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    if sheen > 0 and "Sheen Weight" in b.inputs:
        b.inputs["Sheen Weight"].default_value = sheen
    return m


def nodes(m):
    return m.node_tree.nodes, m.node_tree.links, m.node_tree.nodes["Principled BSDF"]


def mottled(name, color, scale=6.0, low=0.95, high=1.03, rough=0.9):
    """A flat colour with a soft noise in it (plaster, paint, paper, concrete)."""
    m = principled(name, color, rough)
    n, l, b = nodes(m)
    noise = n.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = 6.0
    ramp = n.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*(c * low for c in color), 1)
    ramp.color_ramp.elements[1].color = (*(min(1, c * high) for c in color), 1)
    l.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    l.new(ramp.outputs["Color"], b.inputs["Base Color"])
    return m


def emissive(name, color, strength):
    """A glowing surface; the bake takes `color * strength` as its light."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    n = m.node_tree.nodes
    for node in list(n):
        if node.type != "OUTPUT_MATERIAL":
            n.remove(node)
    e = n.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = (*color, 1)
    e.inputs["Strength"].default_value = strength
    m.node_tree.links.new(e.outputs["Emission"], n["Material Output"].inputs["Surface"])
    m["glow"] = [c * strength for c in color]
    return m


def painting(seed, colors):
    """Soft blobs of a few flat colours (an abstract painting or print)."""
    m = principled(f"painting {seed}", colors[0], 0.8)
    n, l, b = nodes(m)
    noise = n.new("ShaderNodeTexNoise")
    noise.noise_dimensions = "4D"
    noise.inputs["Scale"].default_value = 1.6
    noise.inputs["Detail"].default_value = 3.0
    noise.inputs["W"].default_value = seed
    ramp = n.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    for i, c in enumerate(colors):
        e = ramp.color_ramp.elements[i] if i < 2 else ramp.color_ramp.elements.new(i / len(colors))
        e.position = i / len(colors)
        e.color = (*c, 1)
    l.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    l.new(ramp.outputs["Color"], b.inputs["Base Color"])
    return m


# --- Geometry ---------------------------------------------------------------------

def box(name, centre, size, material, bevel=0.0):
    bpy.ops.mesh.primitive_cube_add(location=centre)
    o = bpy.context.object
    o.name = name
    o.scale = (size[0] / 2, size[1] / 2, size[2] / 2)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)  # pivot stays put
    if bevel > 0:
        md = o.modifiers.new("bevel", "BEVEL")
        md.width = bevel
        md.segments = 3
    o.data.materials.append(material)
    return o


def cylinder(name, centre, radius, depth, material, vertices=24, rotation=(0, 0, 0)):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth, location=centre,
                                        rotation=rotation)
    o = bpy.context.object
    o.name = name
    o.data.materials.append(material)
    return o


def plane(name, centre, size, material, rotation=(0, 0, 0)):
    bpy.ops.mesh.primitive_plane_add(location=centre, rotation=rotation)
    o = bpy.context.object
    o.name = name
    o.scale = (size[0] / 2, size[1] / 2, 1)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)  # pivot stays put
    o.data.materials.append(material)
    return o


def rod(name, a, b, radius, material, vertices=12):
    """A cylinder from point a to point b."""
    a, b = mathutils.Vector(a), mathutils.Vector(b)
    d = b - a
    o = cylinder(name, (a + b) / 2, radius, d.length, material, vertices)
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = mathutils.Vector((0, 0, 1)).rotation_difference(d.normalized())
    return o


def asset(assets_dir, name, location, yaw_deg=0.0, scale=1.0, drop_materials=()):
    """Imports a Khronos glTF sample asset, standing at `location`."""
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=os.path.join(assets_dir, name + ".glb"))
    new = [o for o in bpy.data.objects if o not in before]
    roots = [o for o in new if o.parent is None]
    for r in roots:
        r.location = mathutils.Vector(location) + r.location
        r.rotation_mode = "XYZ"
        r.rotation_euler[2] += math.radians(yaw_deg)
        r.scale = (scale, scale, scale)
    for o in new:
        if o.type == "MESH" and any(s.material and any(d in s.material.name for d in drop_materials)
                                    for s in o.material_slots):
            bpy.data.objects.remove(o, do_unlink=True)
    return new


def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


# --- Rendering ----------------------------------------------------------------------

def render_settings(samples, exposure=0.0):
    s = bpy.context.scene
    s.render.engine = "CYCLES"
    s.cycles.device = "CPU"
    s.cycles.samples = samples
    s.cycles.max_bounces = 6
    s.view_settings.view_transform = "AgX"
    s.view_settings.exposure = exposure
    looks = [i.identifier for i in s.view_settings.bl_rna.properties["look"].enum_items]
    s.view_settings.look = "AgX - Medium High Contrast" if "AgX - Medium High Contrast" in looks else "None"


def preview(out_dir, views, exposure=0.0, samples=48, lens=18):
    """Renders `views` [(eye, target)] to <out_dir>/preview_<i>.png."""
    render_settings(samples, exposure)
    s = bpy.context.scene
    s.cycles.use_denoising = True
    s.render.resolution_x, s.render.resolution_y = 640, 480
    for i, (eye, target) in enumerate(views):
        bpy.ops.object.camera_add(location=eye)
        cam = bpy.context.object
        cam.data.lens = lens
        cam.rotation_mode = "QUATERNION"
        cam.rotation_quaternion = (mathutils.Vector(target) - mathutils.Vector(eye)).to_track_quat("-Z", "Y")
        s.camera = cam
        s.render.filepath = os.path.join(out_dir, f"preview_{i}.png")
        t = time.time()
        bpy.ops.render.render(write_still=True)
        print("preview", i, round(time.time() - t, 1), "s", flush=True)


# --- Baking into splats ------------------------------------------------------------

def densify(o, spacing):
    """Splits edges longer than `spacing` until none are (world units)."""
    bm = bmesh.new()
    bm.from_mesh(o.data)
    scale = max(o.matrix_world.to_scale())
    for _ in range(10):
        long_edges = [e for e in bm.edges if e.calc_length() * scale > spacing]
        if not long_edges:
            break
        bmesh.ops.subdivide_edges(bm, edges=long_edges, cuts=1, use_grid_fill=True)
        bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
    bm.to_mesh(o.data)
    bm.free()


def bake_and_export(path, samples, spacing_for, median_grey=0.32, smooth_light=4, expose=None):
    """Bakes every mesh's light into splats and writes them to `path` (PLY).
    `spacing_for(object)` gives the splat spacing for an object (metres);
    `median_grey` is the display value the median surface is exposed to (of
    the objects `expose(object)` picks, if given);
    `smooth_light` how many rings of neighbours the baked light is averaged
    over (it is path traced, so noisy)."""
    s = bpy.context.scene
    render_settings(samples)
    meshes = []
    for o in list(bpy.data.objects):
        if o.type != "MESH":
            continue
        bpy.context.view_layer.objects.active = o
        o.select_set(True)
        for md in list(o.modifiers):
            bpy.ops.object.modifier_apply(modifier=md.name)
        o.select_set(False)
        densify(o, spacing_for(o))
        if "bake" not in o.data.color_attributes:
            o.data.color_attributes.new("bake", "FLOAT_COLOR", "POINT")
        o.data.color_attributes.active_color = o.data.color_attributes["bake"]
        meshes.append(o)
    print("vertices", sum(len(o.data.vertices) for o in meshes), flush=True)
    heavy = {}
    for o in meshes:
        key = o.name.split(".")[0]
        heavy[key] = heavy.get(key, 0) + len(o.data.vertices)
    print("heaviest", sorted(heavy.items(), key=lambda kv: -kv[1])[:12], flush=True)
    for o in bpy.data.objects:
        o.select_set(False)
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    s.render.bake.target = "VERTEX_COLORS"
    # The light (noisy: path traced) and the surface colour (exact) apart, so
    # the light alone can be smoothed.
    t = time.time()
    bpy.ops.object.bake(type="DIFFUSE", pass_filter={"DIRECT", "INDIRECT"})
    print("bake light", round(time.time() - t, 1), "s", flush=True)
    light = {o.name: np.array([d.color[:3] for d in o.data.color_attributes["bake"].data]).reshape(-1, 3)
             for o in meshes}
    s.cycles.samples = 4
    bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"})
    s.cycles.samples = samples

    pos, nrm, col, spc, lit = [], [], [], [], []
    for o in meshes:
        m = o.data
        mw = o.matrix_world
        nw = mw.to_3x3().inverted().transposed()
        lengths = np.zeros(len(m.vertices))
        counts = np.zeros(len(m.vertices))
        edges = np.array([e.vertices[:] for e in m.edges], dtype=np.int64).reshape(-1, 2)
        world = np.array([(mw @ v.co)[:] for v in m.vertices]).reshape(-1, 3)
        normals = np.array([(nw @ v.normal).normalized()[:] for v in m.vertices]).reshape(-1, 3)
        if len(edges):
            length = np.linalg.norm(world[edges[:, 0]] - world[edges[:, 1]], axis=1)
            np.add.at(lengths, edges[:, 0], length)
            np.add.at(lengths, edges[:, 1], length)
            np.add.at(counts, edges[:, 0], 1)
            np.add.at(counts, edges[:, 1], 1)
        albedo = np.array([d.color[:3] for d in m.color_attributes["bake"].data]).reshape(-1, 3)
        # Glowing surfaces show their emission, where the bake has none.
        glow = [slot.material.get("glow") for slot in o.material_slots if slot.material]
        glow = [g for g in glow if g is not None]
        if glow:
            colours = np.tile(np.array(glow[0], dtype=float), (len(world), 1))
        else:
            # Smooth the light over a few rings of neighbours (a few cm): the
            # noise goes, shadow edges stay about as soft as the sun makes them.
            e = light[o.name]
            for _ in range(smooth_light):
                if not len(edges):
                    break
                neighbour = np.zeros_like(e)
                np.add.at(neighbour, edges[:, 0], e[edges[:, 1]])
                np.add.at(neighbour, edges[:, 1], e[edges[:, 0]])
                e = 0.4 * e + 0.6 * np.where(counts[:, None] > 0, neighbour / np.maximum(counts, 1)[:, None], e)
            colours = albedo * e
        pos.append(world)
        nrm.append(normals)
        col.append(colours)
        spc.append(lengths / np.maximum(counts, 1))
        lit.append(np.full(len(world), (expose(o) if expose else True) and not glow))
    write_splats(path, np.concatenate(pos), np.concatenate(nrm), np.concatenate(col), np.concatenate(spc),
                 np.concatenate(lit), median_grey)


def tone(linear, exposure):
    x = linear * exposure
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    y = np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)
    return np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055)


def write_splats(path, pos, nrm, lin, spacing, lit, median_grey):
    lum = lin @ np.array([0.2126, 0.7152, 0.0722])
    # Exposure from the lit surfaces (glowing ones are far brighter).
    exposure = median_grey / max(np.median(lum[lit]), 1e-6)
    rgb = tone(lin, exposure)
    # Blender (x, y, z up) to RDF (x right, y down, z forward).
    p = np.stack([pos[:, 0], -pos[:, 2], pos[:, 1]], 1)
    n = np.stack([nrm[:, 0], -nrm[:, 2], nrm[:, 1]], 1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    # A frame with z along the normal.
    helper = np.where(np.abs(n[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    x = np.cross(helper, n)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = np.cross(n, x)
    m = np.stack([x, y, n], 2)  # columns
    w = np.sqrt(np.maximum(0, 1 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2])) / 2
    w = np.maximum(w, 1e-6)
    q = np.stack([w, (m[:, 2, 1] - m[:, 1, 2]) / (4 * w), (m[:, 0, 2] - m[:, 2, 0]) / (4 * w),
                  (m[:, 1, 0] - m[:, 0, 1]) / (4 * w)], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    sigma = np.clip(spacing, 0.004, 0.2) * 0.62
    scales = np.log(np.stack([sigma, sigma, sigma * 0.15], 1))
    dc = (rgb - 0.5) / 0.28209479177387814
    opacity = np.full((len(p), 1), math.log(0.97 / 0.03))
    n_pts = len(p)
    props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.concatenate([p, np.zeros((n_pts, 3)), dc, opacity, scales, q], 1).astype("<f4")
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n_pts).encode())
        f.write("".join(f"property float {k}\n" for k in props).encode())
        f.write(b"end_header\n")
        f.write(data.tobytes())
    print("wrote", n_pts, "splats to", path, "exposure", round(exposure, 3), flush=True)
