"""Handheld orbit camera paths that mimic someone walking around a subject with a phone.

World space is Y-up in metres (like ARCore). Poses are returned as camera-to-world
matrices in the OpenGL convention (x right, y up, camera looks down -z), which is
also what nerfstudio's transforms.json expects.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np


@dataclasses.dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @staticmethod
    def portrait_phone(width: int = 720, height: int = 1280, vertical_fov_deg: float = 65.0) -> "Intrinsics":
        """Main camera of a phone held upright (about a 26 mm equivalent lens)."""
        f = 0.5 * height / math.tan(math.radians(vertical_fov_deg) / 2)
        return Intrinsics(width, height, f, f, width / 2, height / 2)


@dataclasses.dataclass
class OrbitSpec:
    target: tuple[float, float, float]  # point the operator keeps in frame
    radius: float                       # distance from the subject's vertical axis
    heights: tuple[float, ...]          # camera height for each loop
    frames: int = 180
    seconds: float = 30.0
    seed: int = 0


def look_at(eye: np.ndarray, target: np.ndarray, roll_rad: float = 0.0) -> np.ndarray:
    """Camera-to-world matrix (OpenGL convention) looking from eye at target."""
    forward = target - eye
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    # Roll about the viewing axis.
    c, s = math.cos(roll_rad), math.sin(roll_rad)
    right, up = c * right + s * up, -s * right + c * up
    m = np.eye(4)
    m[:3, 0] = right
    m[:3, 1] = up
    m[:3, 2] = -forward
    m[:3, 3] = eye
    return m


def handheld_orbit(spec: OrbitSpec) -> tuple[list[np.ndarray], list[float]]:
    """One loop per entry in spec.heights, joined into a single continuous walk.

    Adds the small imperfections of a real capture: uneven walking speed, a
    wobbling radius, step bounce, aim jitter and slight roll.
    """
    rng = np.random.default_rng(spec.seed)
    loops = len(spec.heights)
    t = np.linspace(0.0, 1.0, spec.frames, endpoint=False)

    # Uneven walking speed: progress along the walk is a perturbed ramp.
    speed = 1.0 + 0.25 * np.sin(2 * math.pi * (3 * t + rng.uniform())) + 0.1 * rng.standard_normal(spec.frames)
    speed = np.clip(speed, 0.4, None)
    progress = np.cumsum(speed)
    progress = (progress - progress[0]) / (progress[-1] - progress[0] + 1e-9) * loops * (1 - 1 / spec.frames)

    target = np.array(spec.target, dtype=float)
    poses, timestamps = [], []
    phase = rng.uniform(0, 2 * math.pi, size=4)
    for i, p in enumerate(progress):
        loop = min(int(p), loops - 1)
        frac = p - loop
        # Ease the height change over the first 15 % of each loop after the first.
        if loop > 0 and frac < 0.15:
            k = frac / 0.15
            k = k * k * (3 - 2 * k)
            height = spec.heights[loop - 1] + (spec.heights[loop] - spec.heights[loop - 1]) * k
        else:
            height = spec.heights[loop]
        yaw = 2 * math.pi * p + math.pi / 2
        steps = 2 * math.pi * 1.6 * spec.seconds * t[i]  # ~1.6 steps per second
        radius = spec.radius * (1 + 0.04 * math.sin(2 * math.pi * 2 * t[i] + phase[0]))
        bounce = 0.012 * math.sin(steps + phase[1])
        eye = np.array([
            target[0] + radius * math.cos(yaw),
            height + bounce,
            target[2] + radius * math.sin(yaw),
        ])
        aim = target + rng.normal(0, 0.012 * spec.radius, size=3)
        roll = math.radians(1.5) * math.sin(steps / 2 + phase[2]) + math.radians(0.4) * rng.standard_normal()
        poses.append(look_at(eye, aim, roll))
        timestamps.append(float(t[i] * spec.seconds))
    return poses, timestamps


def world_to_camera_opencv(c2w_gl: np.ndarray) -> np.ndarray:
    """4x4 world-to-camera in the OpenCV convention (x right, y down, z forward)."""
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return np.linalg.inv(c2w_gl @ flip)
