"""Renders a textured glTF subject inside a procedural room with headless OpenGL.

Produces, per camera: an sRGB image, metric depth along the viewing axis and a
soft subject mask. Rendering is supersampled 2x per axis and box-filtered, so
edges look like a camera image rather than a rasteriser.
"""

from __future__ import annotations

import dataclasses
import math

import moderngl
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter

from camera_path import Intrinsics

VERTEX_SHADER = """
#version 330
uniform mat4 u_view_proj;
uniform mat4 u_view;
in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;
out vec3 v_world;
out vec3 v_normal;
out vec2 v_uv;
out float v_depth;
void main() {
    v_world = in_position;
    v_normal = in_normal;
    v_uv = in_uv;
    v_depth = -(u_view * vec4(in_position, 1.0)).z;
    gl_Position = u_view_proj * vec4(in_position, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330
uniform sampler2D u_albedo;
uniform vec3 u_camera;
uniform float u_subject;
uniform float u_specular;
uniform vec3 u_tint;
in vec3 v_world;
in vec3 v_normal;
in vec2 v_uv;
in float v_depth;
layout(location = 0) out vec4 out_color;
layout(location = 1) out float out_depth;
layout(location = 2) out float out_mask;

vec3 to_linear(vec3 c) { return pow(c, vec3(2.2)); }

void main() {
    vec3 n = normalize(v_normal);
    if (!gl_FrontFacing) n = -n;
    vec3 albedo = to_linear(texture(u_albedo, v_uv).rgb * u_tint);
    vec3 view = normalize(u_camera - v_world);

    // Soft indoor lighting: hemisphere ambient, a key light and a fill light.
    vec3 sky = vec3(0.62, 0.64, 0.68), ground = vec3(0.30, 0.27, 0.24);
    vec3 light = mix(ground, sky, 0.5 + 0.5 * n.y) * 0.85;
    vec3 key = normalize(vec3(0.45, 0.85, 0.55));
    vec3 fill = normalize(vec3(-0.7, 0.35, -0.4));
    light += vec3(1.0, 0.96, 0.9) * 0.95 * max(dot(n, key), 0.0);
    light += vec3(0.8, 0.85, 1.0) * 0.30 * max(dot(n, fill), 0.0);
    vec3 color = albedo * light;
    vec3 h = normalize(key + view);
    color += u_specular * pow(max(dot(n, h), 0.0), 40.0) * vec3(1.0);

    out_color = vec4(pow(color, vec3(1.0 / 2.2)), 1.0);
    out_depth = v_depth;
    out_mask = u_subject;
}
"""


@dataclasses.dataclass
class MeshPart:
    positions: np.ndarray  # (n, 3) float32, world space
    normals: np.ndarray    # (n, 3)
    uvs: np.ndarray        # (n, 2)
    indices: np.ndarray    # (m, 3) uint32
    texture: Image.Image
    subject: bool
    specular: float = 0.0
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0)


# --- Subjects -----------------------------------------------------------------------


def load_gltf_subject(path: str, texture_override: str | None, rotate_x_deg: float, height_m: float,
                      base: str, specular: float) -> list[MeshPart]:
    """Loads a glTF, rotates it upright, scales it to `height_m` and places it.

    base="floor": feet on y=0 at the origin. base="center": centred at the origin.
    """
    scene = trimesh.load(path, force="scene")
    parts = []
    meshes = scene.dump() if hasattr(scene, "dump") else [scene]
    rot = trimesh.transformations.rotation_matrix(math.radians(rotate_x_deg), [1, 0, 0])
    all_points = np.concatenate([m.vertices for m in meshes]) @ rot[:3, :3].T
    lo, hi = all_points.min(0), all_points.max(0)
    scale = height_m / (hi[1] - lo[1])
    centre = (lo + hi) / 2
    for mesh in meshes:
        v = (mesh.vertices @ rot[:3, :3].T - centre) * scale
        if base == "floor":
            v[:, 1] += height_m / 2
        normals = mesh.vertex_normals @ rot[:3, :3].T
        uv = np.asarray(mesh.visual.uv, dtype=np.float32)
        if texture_override:
            image = Image.open(texture_override).convert("RGB")
        else:
            material = mesh.visual.material
            image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
            image = image.convert("RGB") if image is not None else Image.new("RGB", (4, 4), (180, 160, 150))
        parts.append(MeshPart(v.astype(np.float32), normals.astype(np.float32), uv,
                              np.asarray(mesh.faces, dtype=np.uint32), image, True, specular))
    return parts


# --- Procedural room ----------------------------------------------------------------


def _noise(rng: np.random.Generator, size: int, octaves: int = 4) -> np.ndarray:
    out = np.zeros((size, size), np.float32)
    for o in range(octaves):
        cells = 4 * 2 ** o
        grid = rng.standard_normal((cells, cells)).astype(np.float32)
        layer = np.asarray(Image.fromarray(grid).resize((size, size), Image.BICUBIC))
        out += layer / 2 ** o
    return (out - out.min()) / (np.ptp(out) + 1e-6)


def wood_floor(rng: np.random.Generator, size: int = 1024) -> Image.Image:
    img = np.zeros((size, size, 3), np.float32)
    planks = 8
    grain = _noise(rng, size, 5)
    for p in range(planks):
        y0, y1 = p * size // planks, (p + 1) * size // planks
        base = np.array([0.52, 0.37, 0.24]) * rng.uniform(0.8, 1.15)
        streaks = np.sin(np.linspace(0, rng.uniform(30, 60), size) + 6 * grain[y0:y1, :]) * 0.06
        img[y0:y1] = base * (0.85 + 0.25 * grain[y0:y1, :, None]) + streaks[..., None]
        img[y0:y0 + 3] *= 0.6  # seam
    return Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))


def plaster_wall(rng: np.random.Generator, size: int = 1024, posters: int = 3) -> Image.Image:
    n = _noise(rng, size, 5)
    base = np.array([0.80, 0.78, 0.74]) * rng.uniform(0.9, 1.05)
    img = Image.fromarray(np.clip((base * (0.9 + 0.12 * n[..., None])) * 255, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    for _ in range(posters):  # framed pictures give the background trackable texture
        w, h = rng.integers(140, 260), rng.integers(180, 320)
        x, y = rng.integers(40, size - w - 40), rng.integers(200, size - h - 120)
        draw.rectangle([x - 8, y - 8, x + w + 8, y + h + 8], fill=(40, 34, 30))
        draw.rectangle([x, y, x + w, y + h], fill=tuple(int(c) for c in rng.integers(150, 255, 3)))
        for _ in range(12):
            cx, cy, r = x + rng.integers(0, w), y + rng.integers(0, h), rng.integers(8, 50)
            color = tuple(int(c) for c in rng.integers(20, 235, 3))
            if rng.uniform() < 0.5:
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
            else:
                draw.rectangle([cx - r, cy - r // 2, cx + r, cy + r // 2], fill=color)
    return img.filter(ImageFilter.GaussianBlur(0.6))


def fabric(rng: np.random.Generator, color: tuple[float, float, float], size: int = 512) -> Image.Image:
    n = _noise(rng, size, 6)
    weave = 0.5 + 0.5 * np.sin(np.arange(size)[:, None] * 0.9) * np.sin(np.arange(size)[None, :] * 0.9)
    img = np.array(color) * (0.8 + 0.25 * n[..., None] + 0.05 * weave[..., None])
    return Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))


def quad(corner: np.ndarray, u: np.ndarray, v: np.ndarray, uv_repeat: tuple[float, float], texture: Image.Image,
         normal: np.ndarray) -> MeshPart:
    p = np.array([corner, corner + u, corner + u + v, corner + v], np.float32)
    uv = np.array([[0, 0], [uv_repeat[0], 0], [uv_repeat[0], uv_repeat[1]], [0, uv_repeat[1]]], np.float32)
    n = np.repeat(normal[None].astype(np.float32), 4, 0)
    return MeshPart(p, n, uv, np.array([[0, 1, 2], [0, 2, 3]], np.uint32), texture, False)


def box(centre, size, texture: Image.Image, tint=(1.0, 1.0, 1.0)) -> list[MeshPart]:
    c, s = np.array(centre, float), np.array(size, float) / 2
    parts = []
    for axis in range(3):
        for sign in (-1, 1):
            n = np.zeros(3)
            n[axis] = sign
            a, b = [i for i in range(3) if i != axis]
            u, v = np.zeros(3), np.zeros(3)
            u[a], v[b] = 2 * s[a], 2 * s[b]
            corner = c + n * s - u / 2 - v / 2
            part = quad(corner, u, v, (1, 1), texture, n)
            part.tint = tint
            parts.append(part)
    return parts


def room(seed: int, half_extent: float = 3.5, height: float = 2.8) -> list[MeshPart]:
    rng = np.random.default_rng(seed)
    e = half_extent
    floor = wood_floor(rng)
    parts = [
        quad(np.array([-e, 0, -e]), np.array([2 * e, 0, 0]), np.array([0, 0, 2 * e]), (3, 3), floor,
             np.array([0, 1, 0])),
        quad(np.array([-e, height, -e]), np.array([2 * e, 0, 0]), np.array([0, 0, 2 * e]), (1, 1),
             fabric(rng, (0.9, 0.9, 0.88)), np.array([0, -1, 0])),
    ]
    walls = [
        (np.array([-e, 0, -e]), np.array([2 * e, 0, 0]), np.array([0, 0, 1])),
        (np.array([e, 0, e]), np.array([-2 * e, 0, 0]), np.array([0, 0, -1])),
        (np.array([-e, 0, e]), np.array([0, 0, -2 * e]), np.array([1, 0, 0])),
        (np.array([e, 0, -e]), np.array([0, 0, 2 * e]), np.array([-1, 0, 0])),
    ]
    for corner, u, normal in walls:
        parts.append(quad(corner, u, np.array([0, height, 0]), (1, 1), plaster_wall(rng), normal))
    # Furniture-sized blocks near the walls add parallax and features.
    wood = fabric(rng, (0.45, 0.32, 0.22))
    cloth = fabric(rng, (0.25, 0.35, 0.55))
    parts += box((-2.6, 0.45, -2.4), (1.4, 0.9, 0.7), wood)
    parts += box((2.5, 0.4, -1.8), (0.8, 0.8, 1.6), cloth, (0.9, 0.5, 0.4))
    parts += box((2.3, 0.6, 2.5), (1.6, 1.2, 0.6), wood, (0.8, 0.8, 0.8))
    parts += box((-2.7, 0.25, 1.9), (0.9, 0.5, 0.9), cloth)
    parts += box((-2.9, 1.0, 0.2), (0.4, 2.0, 0.4), wood, (1.0, 0.95, 0.85))
    return parts


# --- Renderer -----------------------------------------------------------------------


def projection_from_intrinsics(k: Intrinsics, near: float, far: float, sample_scale: int) -> np.ndarray:
    """OpenGL projection for an OpenCV-style pinhole camera."""
    w, h = k.width * sample_scale, k.height * sample_scale
    fx, fy, cx, cy = k.fx * sample_scale, k.fy * sample_scale, k.cx * sample_scale, k.cy * sample_scale
    p = np.zeros((4, 4))
    p[0, 0] = 2 * fx / w
    p[1, 1] = 2 * fy / h
    p[0, 2] = 1 - 2 * cx / w
    p[1, 2] = 2 * cy / h - 1
    p[2, 2] = -(far + near) / (far - near)
    p[2, 3] = -2 * far * near / (far - near)
    p[3, 2] = -1
    return p


class MeshRenderer:
    def __init__(self, parts: list[MeshPart], intrinsics: Intrinsics, sample_scale: int = 2):
        self.k = intrinsics
        self.s = sample_scale
        self.ctx = moderngl.create_standalone_context(backend="egl")
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.program = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        self.items = []
        self.buffers = []
        for part in parts:
            data = np.hstack([part.positions, part.normals, part.uvs]).astype("f4")
            vbo = self.ctx.buffer(data.tobytes())
            ibo = self.ctx.buffer(part.indices.astype("u4").tobytes())
            vao = self.ctx.vertex_array(self.program, [(vbo, "3f 3f 2f", "in_position", "in_normal", "in_uv")],
                                        index_buffer=ibo)
            image = part.texture.transpose(Image.FLIP_TOP_BOTTOM)  # glTF / our UVs have v pointing down the image
            tex = self.ctx.texture(image.size, 3, image.tobytes())
            tex.build_mipmaps()
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            tex.repeat_x = tex.repeat_y = True
            tex.anisotropy = 8.0
            self.items.append((vao, tex, part))
            self.buffers.append(vbo)
        w, h = intrinsics.width * sample_scale, intrinsics.height * sample_scale
        self.color = self.ctx.texture((w, h), 4)
        self.depth = self.ctx.texture((w, h), 1, dtype="f4")
        self.mask = self.ctx.texture((w, h), 1, dtype="f4")
        self.zbuffer = self.ctx.depth_renderbuffer((w, h))
        self.fbo = self.ctx.framebuffer([self.color, self.depth, self.mask], self.zbuffer)
        self.projection = projection_from_intrinsics(intrinsics, 0.05, 50.0, sample_scale)

    def update_part(self, index: int, positions: np.ndarray, normals: np.ndarray) -> None:
        """Replaces a part's vertex positions and normals (same count), e.g. a deformed subject."""
        part = self.items[index][2]
        data = np.hstack([positions, normals, part.uvs]).astype("f4")
        self.buffers[index].write(data.tobytes())

    def render(self, c2w_gl: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (rgb uint8 HxWx3, depth float32 HxW in metres, mask float32 HxW in [0,1])."""
        view = np.linalg.inv(c2w_gl)
        self.fbo.use()
        self.fbo.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        self.program["u_view_proj"].write((self.projection @ view).T.astype("f4").tobytes())
        self.program["u_view"].write(view.T.astype("f4").tobytes())
        self.program["u_camera"].value = tuple(c2w_gl[:3, 3])
        for vao, tex, part in self.items:
            tex.use(0)
            self.program["u_albedo"].value = 0
            self.program["u_subject"].value = 1.0 if part.subject else 0.0
            self.program["u_specular"].value = part.specular
            self.program["u_tint"].value = part.tint
            vao.render()
        w, h = self.k.width * self.s, self.k.height * self.s
        rgb = np.frombuffer(self.color.read(), np.uint8).reshape(h, w, 4)[::-1, :, :3]
        depth = np.frombuffer(self.depth.read(), np.float32).reshape(h, w)[::-1]
        mask = np.frombuffer(self.mask.read(), np.float32).reshape(h, w)[::-1]
        s = self.s
        # Box-filter the supersampled colour and mask; take the centre sample for depth.
        rgb = rgb.reshape(h // s, s, w // s, s, 3).astype(np.float32).mean((1, 3))
        mask = mask.reshape(h // s, s, w // s, s).mean((1, 3))
        depth = depth[s // 2::s, s // 2::s].copy()
        return np.clip(rgb + 0.5, 0, 255).astype(np.uint8), depth, mask
