"""Builds a virtual phone capture: images + exact camera poses (+ masks and depth).

Subjects:
  head    Lee Perry-Smith head scan on a pedestal (real human face), in a room
  body    full-body character standing in a room (stands in for a person)
  lizard  Niantic's hornedlizard Scaniverse scan (statue in a park), rendered
          with the engine's own splat renderer

Output layout (nerfstudio-compatible transforms.json, OpenGL camera-to-world):
  images/frame_00000.jpg     colour, portrait
  masks/frame_00000.png      soft subject mask (mesh subjects only)
  depth/frame_00000.png      ARCore-like depth: low-res, noisy, 16-bit millimetres
  depth_confidence/...png    8-bit confidence for that depth
  depth_gt/frame_00000.png   exact depth at 1/4 resolution, 16-bit millimetres
  points3d.ply               sparse coloured points, like SfM output (lizard only)
  transforms.json, preview.mp4, README.txt

--motion S (body only) makes the person fidget while the camera goes round:
the head turns and nods, the arms sway and the upper body leans, smoothly and
about the rest pose (S = 1: head up to ~8 deg, arms ~5 deg, torso ~2 deg). With
the same --frames, a run without --motion renders the rest pose from the same
cameras, for judging a model trained on the moving one.

Example:
  python make_dataset.py head --assets assets/ --out out/head
  python make_dataset.py lizard --spz hornedlizard.spz --gs-render build/engine/gs_render --out out/lizard
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import tempfile
import zipfile

import numpy as np
from PIL import Image

from camera_path import Intrinsics, OrbitSpec, handheld_orbit, world_to_camera_opencv

ATTRIBUTION = {
    "head": "Lee Perry-Smith head scan by Infinite-Realities, CC BY 3.0 "
            "(https://creativecommons.org/licenses/by/3.0/), via the three.js examples.",
    "body": "'Michelle' character from Adobe Mixamo (https://www.mixamo.com), via the three.js examples. "
            "For internal testing only; do not redistribute the model.",
    "lizard": "hornedlizard.spz by Niantic (Scaniverse), MIT License, from https://github.com/nianticlabs/spz.",
}


def arcore_like_depth(depth: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Degrades exact depth into something like ARCore's depth-from-motion output.

    ARCore depth images are about 160x90; error grows roughly with distance
    squared, there are holes, and object boundaries are unreliable.
    """
    h, w = depth.shape
    out_h, out_w = 160, 90  # portrait orientation
    ys = ((np.arange(out_h) + 0.5) * h / out_h).astype(int)
    xs = ((np.arange(out_w) + 0.5) * w / out_w).astype(int)
    d = depth[np.ix_(ys, xs)].astype(np.float64)
    valid = (d > 0.05) & (d < 8.0)

    # Depth discontinuities: neighbours that differ by more than 10 %.
    pad = np.pad(d, 1, mode="edge")
    neighbour_max = np.max([pad[1:-1, :-2], pad[1:-1, 2:], pad[:-2, 1:-1], pad[2:, 1:-1]], axis=0)
    neighbour_min = np.min([pad[1:-1, :-2], pad[1:-1, 2:], pad[:-2, 1:-1], pad[2:, 1:-1]], axis=0)
    edge = (neighbour_max - neighbour_min) > 0.1 * np.maximum(d, 1e-3)

    sigma = 0.004 + 0.012 * d * d
    low_freq = np.asarray(Image.fromarray(rng.standard_normal((8, 5)).astype(np.float32))
                          .resize((out_w, out_h), Image.BICUBIC))
    noisy = d * (1 + 0.01 * low_freq) + rng.standard_normal(d.shape) * sigma
    noisy[edge] += rng.standard_normal(edge.sum()) * 0.15  # smeared boundaries
    holes = rng.uniform(size=d.shape) < 0.04
    holes |= edge & (rng.uniform(size=d.shape) < 0.3)
    keep = valid & ~holes
    confidence = np.clip(255 * np.exp(-sigma / 0.05) * np.where(edge, 0.3, 1.0), 0, 255)
    noisy = np.where(keep, noisy, 0.0)
    confidence = np.where(keep, confidence, 0)
    return (np.clip(noisy * 1000, 0, 65535).astype(np.uint16), confidence.astype(np.uint8))


def rotation(axis_angle: np.ndarray) -> np.ndarray:
    """Rodrigues: rotation matrix of an axis-angle vector (radians)."""
    th = float(np.linalg.norm(axis_angle))
    if th < 1e-12:
        return np.eye(3)
    k = axis_angle / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# Joints of the body subject (Michelle in a T-pose, 1.65 m, facing +z, feet on y = 0).
NECK = np.array([0.0, 1.33, -0.01])
SHOULDERS = (np.array([0.24, 1.26, -0.01]), np.array([-0.24, 1.26, -0.01]))
WAIST = np.array([0.0, 0.95, 0.0])


def body_motion(frames: int, scale: float, seed: int) -> np.ndarray:
    """Per frame, axis-angle rotations (radians) of head, right arm, left arm and torso: (frames, 4, 3).

    Each angle is a sum of two sines with random phases and periods between a
    sixth and two thirds of the capture, minus its mean, so the rest pose is
    the average pose.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(frames, dtype=np.float64)
    # Amplitudes in degrees per joint, axes x (nod / lean forward), y (turn), z (tilt / raise).
    amplitude = np.array([[4.0, 8.0, 3.0],    # head: nod, turn, tilt
                          [0.0, 4.0, 5.0],    # right arm: swing forward/back, raise/lower
                          [0.0, 4.0, 5.0],    # left arm
                          [2.0, 3.0, 1.5]])   # torso: lean, twist, side lean
    out = np.zeros((frames, 4, 3))
    for j in range(4):
        for a in range(3):
            if amplitude[j, a] == 0:
                continue
            wave = np.zeros(frames)
            for weight in (0.65, 0.35):
                period = rng.uniform(frames / 6, frames * 2 / 3)
                wave += weight * np.sin(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi))
            wave -= wave.mean()
            wave /= max(np.abs(wave).max(), 1e-9)
            out[:, j, a] = np.radians(amplitude[j, a] * scale) * wave
    return out


def deform_body(positions: np.ndarray, normals: np.ndarray, angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Poses the T-pose body: head about the neck, arms about the shoulders, then
    everything above the waist about the waist, each blended in over a few cm
    (linear blend skinning with smooth weights)."""
    p, n = positions.astype(np.float64), normals.astype(np.float64)
    x, y = p[:, 0], p[:, 1]

    def blend(p, n, weight, pivot, rot):
        moved = (p - pivot) @ rot.T + pivot
        turned = n @ rot.T
        w = weight[:, None]
        return p + w * (moved - p), n + w * (turned - n)

    head = smoothstep(1.30, 1.38, y)
    p, n = blend(p, n, head, NECK, rotation(angles[0]))
    arm_band = smoothstep(1.05, 1.15, y) * (1 - smoothstep(1.30, 1.36, y))
    for side, (shoulder, sign) in enumerate(zip(SHOULDERS, (1.0, -1.0))):
        weight = smoothstep(0.24, 0.34, sign * x) * arm_band
        a = angles[1 + side].copy()
        a[2] *= sign  # the same raise on both sides
        p, n = blend(p, n, weight, shoulder, rotation(a))
    p, n = blend(p, n, smoothstep(0.85, 1.05, y), WAIST, rotation(angles[3]))
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    return p.astype(np.float32), n.astype(np.float32)


def save_png16(path: str, data: np.ndarray) -> None:
    Image.fromarray(data.astype(np.uint16)).save(path)


def subject_setup(name: str, assets: str):
    """Returns (scene parts, orbit spec) for a mesh subject."""
    import mesh_capture as mc

    if name == "head":
        bust_height = 0.42
        parts = mc.load_gltf_subject(os.path.join(assets, "LeePerrySmith.glb"),
                                     os.path.join(assets, "Map-COL.jpg"), 0.0, bust_height, "center", 0.12)
        centre_y = 1.42
        for p in parts:
            p.positions[:, 1] += centre_y
        rng = np.random.default_rng(7)
        stone = mc.fabric(rng, (0.55, 0.55, 0.57))
        pedestal = mc.box((0, (centre_y - bust_height / 2) / 2, 0.02), (0.34, centre_y - bust_height / 2, 0.30), stone)
        parts += pedestal + mc.room(seed=11)
        spec = OrbitSpec(target=(0.0, centre_y + 0.05, 0.0), radius=0.7, heights=(centre_y + 0.18, centre_y - 0.12),
                         seed=1)
    elif name == "body":
        parts = mc.load_gltf_subject(os.path.join(assets, "Michelle.glb"), None, -90.0, 1.65, "floor", 0.04)
        parts += mc.room(seed=23)
        spec = OrbitSpec(target=(0.0, 0.88, 0.0), radius=2.3, heights=(1.5, 0.95), seed=2)
    else:
        raise ValueError(name)
    return parts, spec


def render_mesh_subject(name: str, assets: str, k: Intrinsics, frames: int, out: str,
                        radius: float | None = None, heights: tuple[float, float] | None = None,
                        motion: float = 0.0, motion_seed: int = 5):
    import mesh_capture as mc

    parts, spec = subject_setup(name, assets)
    spec.frames = frames
    if radius is not None:
        spec.radius = radius
    if heights is not None:
        spec.heights = heights
    poses, timestamps = handheld_orbit(spec)
    renderer = mc.MeshRenderer(parts, k)
    rng = np.random.default_rng(spec.seed + 100)
    moving = [i for i, p in enumerate(parts) if p.subject] if motion > 0 else []
    if moving and name != "body":
        raise ValueError("--motion needs the body subject")
    angles = body_motion(frames, motion, motion_seed) if moving else None
    records = []
    for i, (c2w, ts) in enumerate(zip(poses, timestamps)):
        for m in moving:
            renderer.update_part(m, *deform_body(parts[m].positions, parts[m].normals, angles[i]))
        rgb, depth, mask = renderer.render(c2w)
        stem = f"frame_{i:05d}"
        Image.fromarray(rgb).save(os.path.join(out, "images", stem + ".jpg"), quality=92)
        Image.fromarray(np.clip(mask * 255 + 0.5, 0, 255).astype(np.uint8)).save(os.path.join(out, "masks", stem + ".png"))
        noisy, confidence = arcore_like_depth(depth, rng)
        save_png16(os.path.join(out, "depth", stem + ".png"), noisy)
        Image.fromarray(confidence).save(os.path.join(out, "depth_confidence", stem + ".png"))
        gt = depth[2::4, 2::4]
        save_png16(os.path.join(out, "depth_gt", stem + ".png"), np.clip(gt * 1000, 0, 65535))
        records.append({
            "file_path": f"images/{stem}.jpg",
            "mask_path": f"masks/{stem}.png",
            "depth_file_path": f"depth/{stem}.png",
            "depth_confidence_path": f"depth_confidence/{stem}.png",
            "depth_gt_path": f"depth_gt/{stem}.png",
            "transform_matrix": c2w.tolist(),
            "timestamp": ts,
            **({"subject_motion_deg": np.degrees(angles[i]).round(3).tolist()} if moving else {}),
        })
        print(f"\r{name}: {i + 1}/{frames}", end="", flush=True)
    print()
    return records, spec


def sparse_points_from_spz(spz: str, out_ply: str, target, radius: float, count: int, seed: int) -> None:
    """Samples opaque splat centres as a coloured PLY with SfM-like noise.

    Half the points are drawn uniformly (most splats sit on the subject, which
    keeps its detail) and half with probability growing with the squared
    distance from the subject, so the surroundings get points too, as they
    would from SfM. Without them the background cannot be trained at all.
    """
    with gzip.open(spz, "rb") as f:
        raw = f.read()
    magic, version, n = np.frombuffer(raw[:12], dtype="<u4")
    if magic != 0x5053474E or version not in (2, 3):
        raise ValueError(f"unsupported SPZ (version {version})")
    frac_bits = raw[13]
    pos = np.frombuffer(raw, dtype=np.uint8, count=n * 9, offset=16).reshape(n, 3, 3).astype(np.int32)
    fixed = pos[..., 0] | (pos[..., 1] << 8) | (pos[..., 2] << 16)
    fixed = np.where(fixed & 0x800000, fixed - (1 << 24), fixed)
    xyz = fixed.astype(np.float64) / float(1 << frac_bits)
    alpha = np.frombuffer(raw, dtype=np.uint8, count=n, offset=16 + n * 9) / 255.0
    color = np.frombuffer(raw, dtype=np.uint8, count=n * 3, offset=16 + n * 10).reshape(n, 3) / 255.0
    rgb = np.clip(0.5 + 0.28209479 * (color - 0.5) / 0.15, 0.0, 1.0)

    dist = np.linalg.norm(xyz - np.asarray(target), axis=1)
    candidates = np.flatnonzero((alpha > 0.5) & (dist < radius))
    rng = np.random.default_rng(seed)
    count = min(count, len(candidates))
    uniform = rng.choice(candidates, size=count // 2, replace=False)
    rest = np.setdiff1d(candidates, uniform)
    weight = np.clip(dist[rest], 0.5, None) ** 2
    far = rng.choice(rest, size=count - len(uniform), replace=False, p=weight / weight.sum())
    idx = np.concatenate([uniform, far])
    pts = (xyz[idx] + rng.normal(scale=0.003, size=(len(idx), 3))).astype("<f4")
    cols = np.round(rgb[idx] * 255).astype(np.uint8)

    vertex = np.empty(len(idx), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                       ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = cols[:, 0], cols[:, 1], cols[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(idx)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(out_ply, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertex.tobytes())


def render_splat_subject(spz: str, gs_render: str, k: Intrinsics, frames: int, out: str):
    # The statue stands near (0.15, y, 0.37), about 0.5 m tall, ground near y = -0.1.
    spec = OrbitSpec(target=(0.15, 0.2, 0.37), radius=1.3, heights=(1.0, 0.55), frames=frames, seed=3)
    poses, timestamps = handheld_orbit(spec)
    with tempfile.TemporaryDirectory() as tmp:
        cameras = os.path.join(tmp, "cameras.txt")
        with open(cameras, "w") as f:
            for c2w in poses:
                w2c = world_to_camera_opencv(c2w)
                values = [k.width, k.height, k.fx, k.fy, k.cx, k.cy] + w2c[:3, :4].reshape(-1).tolist()
                f.write(" ".join(f"{v:.9g}" for v in values) + "\n")
        subprocess.run([gs_render, spz, "--views", cameras, tmp], check=True)
        records = []
        for i, (c2w, ts) in enumerate(zip(poses, timestamps)):
            stem = f"frame_{i:05d}"
            Image.open(os.path.join(tmp, f"view_{i:05d}.ppm")).save(os.path.join(out, "images", stem + ".jpg"),
                                                                    quality=92)
            records.append({"file_path": f"images/{stem}.jpg", "transform_matrix": c2w.tolist(), "timestamp": ts})
    sparse_points_from_spz(spz, os.path.join(out, "points3d.ply"), spec.target, radius=20.0, count=60000, seed=7)
    return records, spec


def write_readme(out: str, name: str, k: Intrinsics, frames: int) -> None:
    text = f"""GSplat Lab virtual capture: {name}

Rendered images that imitate walking around a subject with a phone held upright,
with exact camera poses. Use it to test training without a camera.

- {frames} frames, {k.width}x{k.height}, pinhole fx={k.fx:.2f} fy={k.fy:.2f} cx={k.cx} cy={k.cy}
- transforms.json: nerfstudio format. transform_matrix is camera-to-world,
  OpenGL convention (x right, y up, camera looks down -z). World is Y-up, metres.
- masks/: soft subject mask (255 = subject). depth/: ARCore-like (160x90 portrait,
  noisy, holes), 16-bit millimetres, 0 = missing. depth_gt/: exact depth at 1/4
  resolution, 16-bit millimetres. Mesh subjects only.
- points3d.ply: 60k coloured points with ~3 mm noise on the subject and its
  surroundings, standing in for SfM output (transforms.json "ply_file_path").
  Lizard only.
- preview.mp4: the frames as a video.

Source: {ATTRIBUTION[name]}
"""
    with open(os.path.join(out, "README.txt"), "w") as f:
        f.write(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subject", choices=["head", "body", "lizard"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--assets", default="assets", help="folder with the glTF models (head/body)")
    parser.add_argument("--spz", help="hornedlizard.spz (lizard)")
    parser.add_argument("--gs-render", help="path to the engine's gs_render tool (lizard)")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=1280)
    parser.add_argument("--zip", action="store_true", help="also write <out>.zip")
    parser.add_argument("--radius", type=float, help="orbit radius in metres (mesh subjects; default per subject)")
    parser.add_argument("--heights", type=float, nargs=2, help="camera heights of the two laps (mesh subjects)")
    parser.add_argument("--motion", type=float, default=0.0, help="body only: how much the person fidgets (0 = still)")
    parser.add_argument("--motion-seed", type=int, default=5)
    args = parser.parse_args()

    k = Intrinsics.portrait_phone(args.width, args.height)
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    for sub in ["images", "masks", "depth", "depth_confidence", "depth_gt"]:
        os.makedirs(os.path.join(args.out, sub))

    if args.subject == "lizard":
        records, spec = render_splat_subject(args.spz, args.gs_render, k, args.frames, args.out)
        for sub in ["masks", "depth", "depth_confidence", "depth_gt"]:
            os.rmdir(os.path.join(args.out, sub))
    else:
        records, spec = render_mesh_subject(args.subject, args.assets, k, args.frames, args.out, args.radius,
                                            tuple(args.heights) if args.heights else None, args.motion,
                                            args.motion_seed)

    transforms = {
        "camera_model": "OPENCV",
        "w": k.width, "h": k.height,
        "fl_x": k.fx, "fl_y": k.fy, "cx": k.cx, "cy": k.cy,
        "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
        "depth_unit_scale_factor": 0.001,
        "frames": records,
        **({"ply_file_path": "points3d.ply"} if args.subject == "lizard" else {}),
        "gsplat_lab": {
            "subject": args.subject,
            "virtual_capture": True,
            "world_up": [0.0, 1.0, 0.0],
            "orbit_target": list(spec.target),
            "orbit_radius": spec.radius,
            **({"subject_motion": args.motion, "subject_motion_seed": args.motion_seed} if args.motion > 0 else {}),
            "source": ATTRIBUTION[args.subject],
        },
    }
    with open(os.path.join(args.out, "transforms.json"), "w") as f:
        json.dump(transforms, f, indent=1)
    write_readme(args.out, args.subject, k, args.frames)

    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "15", "-i",
                    os.path.join(args.out, "images", "frame_%05d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", "-movflags", "+faststart", os.path.join(args.out, "preview.mp4")], check=True)

    if args.zip:
        base = os.path.basename(os.path.normpath(args.out))
        with zipfile.ZipFile(args.out.rstrip("/") + ".zip", "w", zipfile.ZIP_STORED) as z:
            for root, _, files in os.walk(args.out):
                for name in sorted(files):
                    path = os.path.join(root, name)
                    z.write(path, os.path.join(base, os.path.relpath(path, args.out)))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
