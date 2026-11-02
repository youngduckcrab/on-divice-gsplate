"""Reads PlayCanvas SOG (version 2) splat files into 3DGS arrays and writes
standard 3DGS PLY files.

SOG is a zip of meta.json and WebP images, one texel per splat (row major):
means as 16-bit log-encoded positions split over two images, quaternions in
"smallest three" form, scales and SH DC through 256-entry codebooks, opacity
in the DC image's alpha, and optional higher SH bands as palette entries.
"""
import io
import json
import zipfile

import numpy as np
from PIL import Image


def _image(z, name):
    with z.open(name) as f:
        return np.asarray(Image.open(io.BytesIO(f.read())).convert("RGBA"))


def read_sog(path):
    with zipfile.ZipFile(path) as z:
        meta = json.loads(z.read("meta.json"))
        if meta.get("version") != 2:
            raise ValueError("only SOG version 2")
        n = meta["count"]
        m = meta["means"]
        lo = _image(z, m["files"][0]).reshape(-1, 4)[:n].astype(np.uint32)
        hi = _image(z, m["files"][1]).reshape(-1, 4)[:n].astype(np.uint32)
        t = ((hi[:, :3] << 8) | lo[:, :3]).astype(np.float64) / 65535.0
        mins, maxs = np.array(m["mins"]), np.array(m["maxs"])
        v = mins + (maxs - mins) * t
        positions = np.sign(v) * (np.exp(np.abs(v)) - 1.0)

        q = _image(z, meta["quats"]["files"][0]).reshape(-1, 4)[:n].astype(np.float64)
        abc = (q[:, :3] / 255.0 - 0.5) * np.sqrt(2.0)
        d = np.sqrt(np.clip(1.0 - (abc ** 2).sum(1), 0.0, None))
        mode = (q[:, 3] - 252).astype(np.int64)
        rot = np.zeros((n, 4))
        for k in range(4):
            sel = mode == k
            others = [j for j in range(4) if j != k]
            rot[sel, k] = d[sel]
            rot[np.ix_(sel, others)] = abc[sel]

        cb = np.array(meta["scales"]["codebook"])
        s = _image(z, meta["scales"]["files"][0]).reshape(-1, 4)[:n]
        log_scales = cb[s[:, :3]]

        cb = np.array(meta["sh0"]["codebook"])
        c = _image(z, meta["sh0"]["files"][0]).reshape(-1, 4)[:n]
        dc = cb[c[:, :3]]
        alpha = np.clip(c[:, 3] / 255.0, 1e-4, 1 - 1e-4)
        opacity = np.log(alpha / (1 - alpha))

        rest = np.zeros((n, 0, 3))
        if "shN" in meta:
            sh = meta["shN"]
            coeffs = {1: 3, 2: 8, 3: 15}[sh["bands"]]
            cb = np.array(sh["codebook"])
            cent = _image(z, sh["files"][0])
            labels = _image(z, sh["files"][1]).reshape(-1, 4)[:n].astype(np.int64)
            idx = labels[:, 0] | (labels[:, 1] << 8)
            row = idx // 64
            col = (idx % 64) * coeffs
            rest = np.zeros((n, coeffs, 3))
            for j in range(coeffs):
                rest[:, j, :] = cb[cent[row, col + j, :3]]
    return {
        "positions": positions.astype(np.float32), "rotations": rot.astype(np.float32),
        "log_scales": log_scales.astype(np.float32), "dc": dc.astype(np.float32),
        "opacity": opacity.astype(np.float32), "rest": rest.astype(np.float32),
        "antialias": bool(meta.get("antialias", False)),
    }


def write_ply(path, s):
    n = len(s["positions"])
    coeffs = s["rest"].shape[1]
    props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    props += [f"f_rest_{i}" for i in range(coeffs * 3)]
    props += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n
    header += "".join(f"property float {p}\n" for p in props) + "end_header\n"
    rest = s["rest"].transpose(0, 2, 1).reshape(n, -1)  # PLY: all of red's coefficients, then green, blue
    cols = [s["positions"], np.zeros((n, 3), np.float32), s["dc"], rest, s["opacity"][:, None],
            s["log_scales"], s["rotations"]]
    data = np.concatenate(cols, axis=1).astype("<f4")
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(data.tobytes())
