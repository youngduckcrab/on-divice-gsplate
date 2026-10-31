"""Simulated phone capture (tools/datasets/simulate_capture.py). Turns a portrait virtual dataset into what the app's capture tab writes:
sensor-oriented (landscape) JPEGs, ARCore-convention poses (OpenGL axes along
the sensor image), 160x90 raw depth + confidence, half-resolution masks, some
frames without depth. Optional degradations: pose noise/drift, mask
erosion/dilation, no depth at all."""
import argparse, json, math, os
import numpy as np
from PIL import Image, ImageFilter

p = argparse.ArgumentParser()
p.add_argument("src"); p.add_argument("dst")
p.add_argument("--pose-noise-cm", type=float, default=0.0)   # per-frame jitter + slow drift (same size)
p.add_argument("--rot-noise-deg", type=float, default=0.0)
p.add_argument("--mask", choices=["exact", "erode", "dilate", "soft"], default="exact")
p.add_argument("--no-depth", action="store_true")
p.add_argument("--missing-depth", type=float, default=0.3)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--noise", choices=["both", "drift", "jitter"], default="both")
a = p.parse_args()
rng = np.random.default_rng(a.seed)
src = json.load(open(os.path.join(a.src, "transforms.json")))
W, H = src["w"], src["h"]  # portrait
for sub in ["images", "masks", "depth", "confidence"]:
    os.makedirs(os.path.join(a.dst, sub), exist_ok=True)

# Upright (portrait) -> sensor (landscape): rotate 90 degrees counter-clockwise
# (sensor orientation 90: the app turns sensor images clockwise to stand them up).
# A point (x, y) goes to (y, W - x), so fx' = fy, fy' = fx, cx' = cy, cy' = W - cx,
# and the camera turns about its z axis: in OpenGL axes c2w' = c2w . R.
R = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)

def rotvec(axis_angle):
    th = np.linalg.norm(axis_angle)
    if th < 1e-12: return np.eye(3)
    k = axis_angle / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K

n = len(src["frames"])
# Slow drift (a random walk smoothed over the loop) plus jitter, like ARCore.
def smooth_walk(scale, dims):
    steps = rng.standard_normal((n, dims))
    walk = np.cumsum(steps, axis=0)
    walk -= np.linspace(0, 1, n)[:, None] * walk[-1]  # loop closes roughly
    walk /= max(np.abs(walk).max(), 1e-9)
    return scale * walk
drift_t = smooth_walk(a.pose_noise_cm / 100.0, 3)
drift_r = smooth_walk(math.radians(a.rot_noise_deg), 3)

frames = []
for i, f in enumerate(src["frames"]):
    stem = "frame_%05d" % i
    img = Image.open(os.path.join(a.src, f["file_path"])).convert("RGB").rotate(90, expand=True)
    img.save(os.path.join(a.dst, "images", stem + ".jpg"), quality=92)
    mask = Image.open(os.path.join(a.src, f["mask_path"])).convert("L")
    if a.mask == "erode": mask = mask.filter(ImageFilter.MinFilter(9))
    if a.mask == "dilate": mask = mask.filter(ImageFilter.MaxFilter(9))
    if a.mask == "soft": mask = mask.filter(ImageFilter.GaussianBlur(6))
    mask = mask.rotate(90, expand=True)
    mask = mask.resize((mask.width // 2, mask.height // 2), Image.BILINEAR)
    mask.save(os.path.join(a.dst, "masks", stem + ".png"))
    entry = {"file_path": f"images/{stem}.jpg", "mask_path": f"masks/{stem}.png"}
    if not a.no_depth and rng.uniform() >= a.missing_depth:
        d = Image.open(os.path.join(a.src, f["depth_file_path"]))
        c = Image.open(os.path.join(a.src, f["depth_confidence_path"]))
        d.rotate(90, expand=True).save(os.path.join(a.dst, "depth", stem + ".png"))
        c.rotate(90, expand=True).save(os.path.join(a.dst, "confidence", stem + ".png"))
        entry["depth_file_path"] = f"depth/{stem}.png"
        entry["depth_confidence_path"] = f"confidence/{stem}.png"
    M = np.array(f["transform_matrix"], dtype=np.float64) @ R
    if a.pose_noise_cm > 0 or a.rot_noise_deg > 0:
        jitter_t = rng.standard_normal(3) * a.pose_noise_cm / 100.0 / 3
        jitter_r = rng.standard_normal(3) * math.radians(a.rot_noise_deg) / 3
        if a.noise == "drift": jitter_t, jitter_r = 0 * jitter_t, 0 * jitter_r
        dt, dr = (0 * drift_t[i], 0 * drift_r[i]) if a.noise == "jitter" else (drift_t[i], drift_r[i])
        M[:3, :3] = rotvec(dr + jitter_r) @ M[:3, :3]
        M[:3, 3] += dt + jitter_t
    entry["transform_matrix"] = M.tolist()
    entry["timestamp"] = i * 100_000_000
    frames.append(entry)

out = {
    "camera_model": "OPENCV", "w": H, "h": W,
    "fl_x": src["fl_y"], "fl_y": src["fl_x"], "cx": src["cy"], "cy": W - src["cx"],
    "k1": 0, "k2": 0, "p1": 0, "p2": 0, "depth_unit_scale_factor": 0.001,
    "gsplat_lab": {"subject": "person", "capture": "simulated arcore", "world_up": [0, 1, 0],
                   "image_orientation": "sensor", "upright_degrees": 90},
    "frames": frames,
}
json.dump(out, open(os.path.join(a.dst, "transforms.json"), "w"), indent=1)
print(f"{len(frames)} frames, {sum('depth_file_path' in f for f in frames)} with depth, "
      f"{out['w']}x{out['h']}, fx {out['fl_x']:.1f} cx {out['cx']:.1f} cy {out['cy']:.1f}")
