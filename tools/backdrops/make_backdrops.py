#!/usr/bin/env python3
"""Builds the viewer's photo backdrops from HDR panoramas.

Every source is a CC0 HDRI from Poly Haven (https://polyhaven.com), fetched
from public GitHub mirrors (Poly Haven itself may be unreachable from build
machines). For each one this writes

  <out>/<id>.jpg      the panorama, exposed and tone-mapped for display
  <out>/manifest.json how to place it and the light it casts:
      sh            irradiance as 9 real SH coefficients per colour, already
                    convolved with the cosine lobe, in the viewer's display
                    space (y up, the default camera looking along -z), scaled
                    so the average over all directions has luminance
                    `brightness` (night scenes are darker than day ones)
      key           the direction most of the light comes from (the sun, a
                    window), what share of the light it carries (shadows,
                    rim light) and its colour (key_color, luminance 1)
      yaw           turn of the panorama (degrees) so the chosen view is
                    behind the model at the default camera
      camera_height how high the panorama was taken above its floor (m)
      radius        distance (m) at which the surroundings are placed

usage: make_backdrops.py <download_dir> <out_dir>
"""
import json
import math
import os
import sys
import urllib.request

import cv2
import numpy as np

GH = "https://raw.githubusercontent.com"
DEMO = f"{GH}/gkjohnson/3d-demo-data/main/hdri"
FILAMENT = f"{GH}/google/filament/main/third_party/environments"
PLAYCANVAS = f"{GH}/playcanvas/engine/main/examples/assets/hdri"
THREE = f"{GH}/mrdoob/three.js/r170/examples/textures/equirectangular"

# id, label, category, HDR source, display source (None: from the HDR), key (exposure), yaw, camera height, radius
SCENES = [
    ("cafe", "카페", "indoor", f"{DEMO}/comfy_cafe_2k.hdr", None, 0.20, 0, 1.5, 5),
    ("studio_bright", "밝은 스튜디오", "indoor", f"{DEMO}/brown_photostudio_02_2k.hdr", None, 0.22, -90, 1.5, 6),
    ("vintage_room", "빈티지 방", "indoor", f"{DEMO}/photo_studio_loft_hall_2k.hdr", None, 0.20, -18, 1.5, 5),
    ("palace_hall", "궁전 홀", "indoor", f"{DEMO}/vestibule_2k.hdr", None, 0.20, 0, 1.6, 8),
    ("museum", "박물관", "indoor", f"{DEMO}/hall_of_mammals_2k.hdr", None, 0.20, -126, 1.5, 6),
    ("christmas", "크리스마스", "indoor", f"{DEMO}/christmas_photo_studio_04_2k.hdr", None, 0.20, 0, 1.5, 6),
    ("sunset_sea", "노을 바다", "outdoor", f"{FILAMENT}/the_sky_is_on_fire_2k.hdr", None, 0.15, 43, 1.7, 40),
    ("sunrise_field", "해돋이 들판", "outdoor", f"{THREE}/spruit_sunrise_1k.hdr", f"{THREE}/spruit_sunrise_4k.hdr.jpg", 0.16, 36, 1.7, 60),
    ("garden", "정원", "outdoor", f"{DEMO}/chinese_garden_2k.hdr", None, 0.18, 36, 1.7, 15),
    ("venice", "베네치아 골목", "outdoor", f"{FILAMENT}/venetian_crossroads_2k.hdr", None, 0.18, -18, 1.7, 8),
    ("market", "런던 시장", "outdoor", f"{DEMO}/leadenhall_market_2k.hdr", None, 0.18, 0, 1.7, 10),
    ("night_square", "밤 광장", "outdoor", f"{PLAYCANVAS}/st-peters-square.hdr", None, 0.07, 0, 1.7, 60),
    ("forest", "숲길", "outdoor", f"{DEMO}/rainforest_trail_2k.hdr", None, 0.18, 0, 1.7, 15),
    ("country_road", "시골길", "outdoor", f"{FILAMENT}/flower_road_2k.hdr", None, 0.18, 0, 1.7, 60),
]

LUMA = np.array([0.2126, 0.7152, 0.0722])


def fetch(url, folder):
    path = os.path.join(folder, url.rsplit("/", 1)[1])
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


def read_rgb(path):
    """Linear RGB float for .hdr, display RGB 0..1 for .jpg."""
    flags = cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR if path.endswith(".hdr") else cv2.IMREAD_COLOR
    img = cv2.imread(path, flags)
    if img is None:
        raise RuntimeError("cannot read " + path)
    img = img[:, :, ::-1].astype(np.float32)
    return img if path.endswith(".hdr") else img / 255.0


def tone_map(x):
    """ACES filmic (Narkowicz), then sRGB."""
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    y = np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)
    return np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055)


def directions(w, h, yaw_deg):
    """Display-space unit direction of every pixel (the viewer's mapping:
    u = atan2(x, -z) / 2pi + 0.5 + yaw / 360, v = acos(y) / pi)."""
    u = (np.arange(w) + 0.5) / w
    v = (np.arange(h) + 0.5) / h
    phi = (u - 0.5 - yaw_deg / 360.0) * 2 * math.pi
    theta = v * math.pi
    st = np.sin(theta)[:, None]
    x = st * np.sin(phi)[None, :]
    z = -st * np.cos(phi)[None, :]
    y = np.cos(theta)[:, None] * np.ones((1, w))
    solid = (2 * math.pi / w) * (math.pi / h) * np.sin(theta)[:, None] * np.ones((1, w))
    return x, y, z, solid


def sh_basis(x, y, z):
    return np.stack([
        0.282095 * np.ones_like(x),
        0.488603 * y, 0.488603 * z, 0.488603 * x,
        1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3 * z * z - 1),
        1.092548 * x * z, 0.546274 * (x * x - y * y),
    ])


def irradiance_sh(radiance, yaw_deg):
    """Ramamoorthi-Hanrahan: project radiance on 9 SH, convolve with the cosine lobe."""
    small = cv2.resize(radiance, (256, 128), interpolation=cv2.INTER_AREA)
    x, y, z, solid = directions(256, 128, yaw_deg)
    basis = sh_basis(x, y, z)  # 9 x h x w
    coeffs = np.einsum("khw,hwc,hw->kc", basis, small, solid)
    band = np.array([math.pi] + [2 * math.pi / 3] * 3 + [math.pi / 4] * 5)
    return coeffs * band[:, None]


def key_light(radiance, yaw_deg):
    small = cv2.resize(radiance, (256, 128), interpolation=cv2.INTER_AREA)
    x, y, z, solid = directions(256, 128, yaw_deg)
    lum = small @ LUMA
    power = lum * solid
    # The brightest 2% of the sphere by area, weighted by power.
    order = np.argsort(lum.ravel())[::-1]
    area = np.cumsum(solid.ravel()[order])
    top = order[: max(1, int(np.searchsorted(area, 0.02 * 4 * math.pi)))]
    p = power.ravel()[top]
    d = np.array([x.ravel()[top] @ p, y.ravel()[top] @ p, z.ravel()[top] @ p])
    d /= max(np.linalg.norm(d), 1e-9)
    # Share of the light from the brightest part: high for sun, low for overcast or soft rooms.
    share = float(p.sum() / max(power.sum(), 1e-9))
    # Its colour, at luminance 1.
    rgb = (small.reshape(-1, 3)[top] * solid.ravel()[top, None]).sum(0)
    color = rgb / max(float(rgb @ LUMA), 1e-9)
    return d, share, color


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    downloads, out = sys.argv[1], sys.argv[2]
    os.makedirs(downloads, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    manifest = []
    for sid, label, category, hdr_url, ldr_url, key, yaw, cam_h, radius in SCENES:
        hdr = read_rgb(fetch(hdr_url, downloads))
        lum = hdr @ LUMA
        exposure = key / math.exp(np.mean(np.log(lum + 1e-4)))
        radiance = hdr * exposure
        if ldr_url:
            display = read_rgb(fetch(ldr_url, downloads))
        else:
            display = tone_map(radiance)
        if display.shape[1] > 4096:
            display = cv2.resize(display, (4096, 2048), interpolation=cv2.INTER_AREA)
        jpg = (np.clip(display, 0, 1) * 255 + 0.5).astype(np.uint8)[:, :, ::-1]
        cv2.imwrite(os.path.join(out, sid + ".jpg"), jpg, [cv2.IMWRITE_JPEG_QUALITY, 88,
                                                          cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
                                                          cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
        sh = irradiance_sh(radiance, yaw)
        mean = float(sh[0] @ LUMA) * 0.282095  # luminance of the average irradiance
        brightness = (key / 0.18) ** 1.0
        sh = sh / max(mean, 1e-9) * brightness
        d, share, color = key_light(radiance, yaw)
        manifest.append({
            "id": sid, "label": label, "category": category, "file": sid + ".jpg",
            "width": int(display.shape[1]), "height": int(display.shape[0]),
            "yaw": yaw, "camera_height": cam_h, "radius": radius,
            "sh": [round(float(v), 5) for v in sh.ravel()],
            "key": [round(float(v), 4) for v in d], "key_share": round(share, 3),
            "key_color": [round(float(v), 3) for v in color],
            "brightness": round(brightness, 3),
            "source": hdr_url, "license": "CC0 (Poly Haven)",
        })
        print(f"{sid:14s} {display.shape[1]}x{display.shape[0]} exposure {exposure:.3g} "
              f"key ({d[0]:+.2f} {d[1]:+.2f} {d[2]:+.2f}) share {share:.2f}")
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
