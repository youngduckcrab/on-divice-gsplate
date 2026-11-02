#!/usr/bin/env python3
"""Builds the viewer's 3D places: real captured Gaussian splat scenes the
model is put into, so it stands among the furniture and walls with true
parallax and occlusion instead of in front of a picture.

Sources are CC BY 4.0 splat captures shipped with the PlayCanvas engine
examples (credited in docs/environments.md and in the app). Each is read
from SOG, cleaned (nearly transparent splats dropped, cropped to the area
around its spots, thinned to a size a phone draws comfortably) and written
as <out>/<id>.gsenv, plus <out>/manifest.json describing the spots a model
can stand at.

.gsenv is a gzip stream of
  "GSEV", u32 version (1), u32 meta length, meta JSON
  (count, means mins/maxs, scale and colour codebooks, antialias)
  then per splat, in planes: means low bytes (3), means high bytes (3),
  quaternion (4, SOG's smallest-three), scale indices (3), colour indices
  and opacity (4). Positions decode as in SOG: lerp(mins, maxs, 16 bits),
  then sign(v) * (exp(|v|) - 1). Higher SH bands are left out: the scene is
  turned around the model, and view-dependent colour does not survive that
  without rotating it.

usage: make_environments.py <download_dir> <out_dir>
"""
import gzip
import io
import json
import os
import struct
import sys
import urllib.request
import zipfile

import numpy as np
from PIL import Image

PC = "https://raw.githubusercontent.com/playcanvas/engine/main/examples/assets/splats"

ENVIRONMENTS = [
    {
        "id": "apartment",
        "source": f"{PC}/apartment.sog",
        "title": "SA3D_R&D_XP47",
        "author": "Stephane Agullo (https://sa3d.fr)",
        "link": "https://superspl.at/view?id=cdcec084",
        "license": "CC BY 4.0",
        "scale": 1.0,              # metres per unit (table tops 0.72 above the floor)
        "crop": [-14.5, 0.5, -9.0, 0.5],  # x0, x1, z0, z1 (the apartment and its courtyard)
        "max_splats": 700000,
        "sky": [[0.62, 0.74, 0.90], [0.86, 0.90, 0.95]],  # zenith, horizon (above the courtyard)
        "spots": [
            {"id": "apartment_living", "label": "아파트 거실", "category": "indoor", "x": -9.6, "z": -4.3, "yaw": 180},
            {"id": "apartment_courtyard", "label": "아파트 안뜰", "category": "outdoor", "x": -4.4, "z": -4.4, "yaw": 270},
        ],
    },
    {
        "id": "hall",
        "source": f"{PC}/knock-community-hall.sog",
        "title": "Knock Community Hall",
        "author": "scbenoit (https://superspl.at/user?id=scbenoit)",
        "link": "https://superspl.at/scene/0ff2e6dc",
        "license": "CC BY 4.0",
        "scale": 1.0,
        "crop": [-12.0, 10.0, -10.0, 10.0],
        "max_splats": 1100000,
        "sky": [[0.40, 0.60, 0.88], [0.78, 0.86, 0.94]],
        "spots": [
            {"id": "hall_lawn", "label": "마을회관 잔디밭", "category": "outdoor", "x": -0.5, "z": 4.6, "yaw": 180},
        ],
    },
]


def fetch(url, folder):
    path = os.path.join(folder, url.rsplit("/", 1)[1])
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


def planes(path):
    """The SOG's raw byte planes and metadata (no decoding: the same codes are written out)."""
    with zipfile.ZipFile(path) as z:
        meta = json.loads(z.read("meta.json"))
        if meta.get("version") != 2:
            raise ValueError("only SOG version 2")
        n = meta["count"]

        def image(name):
            return np.asarray(Image.open(io.BytesIO(z.read(name))).convert("RGBA")).reshape(-1, 4)[:n]

        return meta, {
            "means_l": image(meta["means"]["files"][0])[:, :3],
            "means_u": image(meta["means"]["files"][1])[:, :3],
            "quats": image(meta["quats"]["files"][0]),
            "scales": image(meta["scales"]["files"][0])[:, :3],
            "sh0": image(meta["sh0"]["files"][0]),
        }


def positions(meta, p):
    t = ((p["means_u"].astype(np.uint32) << 8) | p["means_l"]).astype(np.float64) / 65535.0
    mins, maxs = np.array(meta["means"]["mins"]), np.array(meta["means"]["maxs"])
    v = mins + (maxs - mins) * t
    return np.sign(v) * (np.exp(np.abs(v)) - 1.0)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    downloads, out = sys.argv[1], sys.argv[2]
    os.makedirs(downloads, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    manifest = []
    for env in ENVIRONMENTS:
        meta, p = planes(fetch(env["source"], downloads))
        pos = positions(meta, p)
        alpha = p["sh0"][:, 3].astype(np.float64) / 255.0
        x0, x1, z0, z1 = env["crop"]
        keep = (alpha > 0.03) & (pos[:, 0] > x0) & (pos[:, 0] < x1) & (pos[:, 2] > z0) & (pos[:, 2] < z1)
        idx = np.nonzero(keep)[0]
        if len(idx) > env["max_splats"]:
            # Keep the splats that cover most: opacity times area.
            scale_cb = np.array(meta["scales"]["codebook"])
            ls = scale_cb[p["scales"][idx]]
            area = np.exp(np.sort(ls, axis=1)[:, 1:].sum(1))
            weight = alpha[idx] * area
            idx = idx[np.argsort(weight)[::-1][: env["max_splats"]]]
            idx.sort()
        n = len(idx)
        header = {
            "count": n,
            "means": {"mins": meta["means"]["mins"], "maxs": meta["means"]["maxs"]},
            "scales": {"codebook": meta["scales"]["codebook"]},
            "sh0": {"codebook": meta["sh0"]["codebook"]},
            "antialias": bool(meta.get("antialias", False)),
        }
        text = json.dumps(header).encode()
        body = b"".join(np.ascontiguousarray(p[k][idx]).astype(np.uint8).tobytes()
                        for k in ("means_l", "means_u", "quats", "scales", "sh0"))
        blob = b"GSEV" + struct.pack("<II", 1, len(text)) + text + body
        path = os.path.join(out, env["id"] + ".gsenv")
        with gzip.open(path, "wb", compresslevel=9) as f:
            f.write(blob)
        print(f"{env['id']}: {meta['count']} -> {n} splats, {os.path.getsize(path) / 1e6:.1f} MB")
        entry = {k: env[k] for k in ("id", "title", "author", "link", "license", "scale", "sky", "spots")}
        entry["file"] = env["id"] + ".gsenv"
        manifest.append(entry)
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
