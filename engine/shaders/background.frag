#version 450
// A backdrop behind the splats: procedural rooms and skies with a floor under
// the subject and a soft shadow, a photo behind, or a 360-degree panorama
// around. Rays come from the same camera the splats are drawn with, in the
// viewer's display space (y up), so the backdrop turns with the model.
layout(binding = 0) uniform sampler2D picture;

layout(push_constant) uniform Params {
  vec4 axis_x;  // camera x axis in display space; w: fx
  vec4 axis_y;  // camera y (down) axis; w: fy
  vec4 axis_z;  // camera forward; w: cx
  vec4 eye;     // camera centre; w: cy
  vec4 ground;  // floor height, shadow centre x, shadow centre z, shadow radius
  vec4 extra;   // picture aspect (width / height), scene scale (subject height), unused, unused
  uvec4 info;   // logical width, height, surface rotation (quarter turns), backdrop
} pc;

layout(location = 0) out vec4 out_color;

const uint kPlain = 0u, kStudio = 1u, kDark = 2u, kPark = 3u, kChecker = 4u, kSunset = 5u, kNight = 6u,
           kPhoto = 7u, kPanorama = 8u;
const float kPi = 3.14159265;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float hash3(vec3 p) { return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453); }
float noise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
}

// Checkerboard filtered over the pixel's footprint (no shimmer towards the horizon).
float checker(vec2 p) {
  vec2 w = fwidth(p) + 1e-4;
  vec2 i = 2.0 * (abs(fract((p - 0.5 * w) * 0.5) - 0.5) - abs(fract((p + 0.5 * w) * 0.5) - 0.5)) / w;
  return 0.5 - 0.5 * i.x * i.y;
}

// Darker under the subject, fading out over the shadow radius.
float shadow(vec2 xz, float strength) {
  vec2 q = (xz - pc.ground.yz) / max(pc.ground.w, 1e-4);
  return 1.0 - strength * exp(-dot(q, q));
}

vec3 sky(vec3 d, vec3 horizon, vec3 zenith, float power) {
  return mix(horizon, zenith, pow(clamp(d.y, 0.0, 1.0), power));
}

void main() {
  const vec2 size = vec2(pc.info.xy);
  const uint rotation = pc.info.z & 3u;
  // Undo the surface pre-rotation (splat.vert applies it to splats).
  const vec2 fb = (rotation & 1u) != 0u ? size.yx : size;
  const vec2 f = gl_FragCoord.xy / fb * 2.0 - 1.0;
  vec2 ndc = f;
  if (rotation == 1u) ndc = vec2(f.y, -f.x);
  else if (rotation == 2u) ndc = -f;
  else if (rotation == 3u) ndc = vec2(-f.y, f.x);
  const vec2 px = (ndc + 1.0) * 0.5 * size;
  const vec3 d = normalize(pc.axis_x.xyz * ((px.x - pc.axis_z.w) / pc.axis_x.w) +
                           pc.axis_y.xyz * ((px.y - pc.eye.w) / pc.axis_y.w) + pc.axis_z.xyz);
  const uint kind = pc.info.w;
  const float scale = max(pc.extra.y, 1e-3);

  if (kind == kPhoto) {
    // Fills the screen (cropping the picture), and stays put as the model turns.
    vec2 uv = px / size;
    const float screen = size.x / size.y, aspect = max(pc.extra.x, 1e-3);
    if (aspect > screen) uv.x = 0.5 + (uv.x - 0.5) * screen / aspect;
    else uv.y = 0.5 + (uv.y - 0.5) * aspect / screen;
    out_color = vec4(texture(picture, uv).rgb, 1.0);
    return;
  }
  if (kind == kPlain) {
    out_color = vec4(0.08, 0.09, 0.11, 1.0);
    return;
  }

  // Where the ray meets the floor, if it does (looking down, from above it).
  bool on_floor = false;
  vec2 xz = vec2(0.0);
  float dist = 0.0;
  if (d.y < -1e-4) {
    const float t = (pc.ground.x - pc.eye.y) / d.y;
    if (t > 0.0) {
      on_floor = true;
      xz = pc.eye.xz + d.xz * t;
      dist = t / scale;  // in subject heights
    }
  }

  if (kind == kPanorama) {
    const vec2 uv = vec2(atan(d.x, -d.z) / (2.0 * kPi) + 0.5, acos(clamp(d.y, -1.0, 1.0)) / kPi);
    vec3 p = textureLod(picture, uv, 0.0).rgb;  // one level: no seam where u wraps
    // A soft shadow where the floor would be, so the model stands in the picture.
    if (on_floor) p *= shadow(xz, 0.35);
    out_color = vec4(p, 1.0);
    return;
  }

  vec3 c;
  if (kind == kStudio) {
    // A white cyclorama: the floor fades into the wall with distance.
    const vec3 wall = sky(d, vec3(0.97), vec3(0.86, 0.87, 0.89), 0.6);
    c = wall;
    if (on_floor) {
      vec3 floor_c = vec3(0.93) * shadow(xz, 0.45);
      c = mix(floor_c, vec3(0.97), 1.0 - exp(-dist / 6.0));
    }
  } else if (kind == kDark) {
    c = sky(d, vec3(0.06, 0.06, 0.07), vec3(0.015), 0.5);
    if (on_floor) {
      vec2 q = (xz - pc.ground.yz) / scale;
      const float pool = exp(-dot(q, q) / 0.6);
      vec3 floor_c = (vec3(0.07) + vec3(0.30, 0.29, 0.27) * pool) * shadow(xz, 0.6);
      c = mix(floor_c, vec3(0.06, 0.06, 0.07), 1.0 - exp(-dist / 5.0));
    }
  } else if (kind == kPark) {
    const vec3 sun = normalize(vec3(0.45, 0.55, -0.7));
    c = sky(d, vec3(0.78, 0.86, 0.95), vec3(0.30, 0.52, 0.88), 0.55);
    const float s = max(dot(d, sun), 0.0);
    c += vec3(1.0, 0.9, 0.7) * (pow(s, 600.0) * 2.0 + pow(s, 12.0) * 0.18);
    if (on_floor) {
      const vec2 p = xz / scale;
      const float n = 0.55 * noise(p * 6.0) + 0.3 * noise(p * 23.0) + 0.15 * noise(p * 80.0);
      vec3 grass = mix(vec3(0.20, 0.36, 0.12), vec3(0.36, 0.52, 0.20), n);
      grass *= shadow(xz, 0.5);
      c = mix(grass, vec3(0.72, 0.80, 0.86), 1.0 - exp(-dist / 12.0));
    }
  } else if (kind == kChecker) {
    c = sky(d, vec3(0.82, 0.84, 0.87), vec3(0.55, 0.60, 0.70), 0.7);
    if (on_floor) {
      const float k = checker(xz / (0.3 * scale));
      vec3 floor_c = mix(vec3(0.55), vec3(0.88), k) * shadow(xz, 0.45);
      c = mix(floor_c, vec3(0.82, 0.84, 0.87), 1.0 - exp(-dist / 10.0));
    }
  } else if (kind == kSunset) {
    const vec3 sun = normalize(vec3(-0.25, 0.06, -1.0));
    const float h = clamp(d.y, 0.0, 1.0);
    c = mix(vec3(1.0, 0.58, 0.28), vec3(0.88, 0.38, 0.42), smoothstep(0.0, 0.18, h));
    c = mix(c, vec3(0.20, 0.16, 0.36), smoothstep(0.15, 0.7, h));
    const float s = max(dot(d, sun), 0.0);
    c += vec3(1.0, 0.75, 0.45) * (pow(s, 900.0) * 2.5 + pow(s, 20.0) * 0.35);
    if (on_floor) {
      // Dark water-like ground with the sun's glow on it.
      const vec3 r = reflect(d, vec3(0, 1, 0));
      const float g = pow(max(dot(r, sun), 0.0), 40.0);
      vec3 floor_c = (vec3(0.11, 0.07, 0.10) + vec3(0.9, 0.5, 0.3) * g * 0.6) * shadow(xz, 0.5);
      c = mix(floor_c, vec3(0.95, 0.55, 0.32), 1.0 - exp(-dist / 9.0));
    }
  } else {  // kNight
    c = sky(d, vec3(0.06, 0.08, 0.18), vec3(0.01, 0.015, 0.05), 0.45);
    if (d.y > 0.0) {
      const vec3 q = d * 260.0;
      const vec3 cell = floor(q);
      const float h = hash3(cell);
      if (h > 0.985) {
        const vec3 at = cell + vec3(hash3(cell + 1.7), hash3(cell + 3.1), hash3(cell + 5.3));
        const float star = smoothstep(0.45, 0.0, length(q - at)) * (h - 0.985) / 0.015;
        c += vec3(0.9, 0.92, 1.0) * star * smoothstep(0.0, 0.15, d.y);
      }
    }
    if (on_floor) {
      vec2 q = (xz - pc.ground.yz) / scale;
      const float glow = exp(-dot(q, q) / 0.4);
      vec3 floor_c = (vec3(0.025, 0.03, 0.055) + vec3(0.10, 0.22, 0.45) * glow) * shadow(xz, 0.6);
      c = mix(floor_c, vec3(0.06, 0.08, 0.18), 1.0 - exp(-dist / 6.0));
    }
  }
  out_color = vec4(c, 1.0);
}
