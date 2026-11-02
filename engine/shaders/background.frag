#version 450
// A backdrop behind the splats: a 360-degree photo around the model (with its
// floor under the model's feet), a photo behind it, or a white studio. Rays
// come from the camera the splats are drawn with, in the viewer's display
// space (y up), so the backdrop turns with the model.
//
// Panoramas are "ground projected": the photo is taken to have been shot
// `camera_height` above a flat floor, with the surroundings `radius` away. A
// ray that meets the floor near the model samples the photo's floor where
// the photo saw that spot, so the model stands on it as the view turns.
// Like a camera focused on the model, the farther something is from the
// model's distance the blurrier it gets (the picture's mip levels). A soft
// shadow pools under the model and falls away from the main light.
layout(binding = 0) uniform sampler2D picture;

layout(push_constant) uniform Params {
  vec4 axis_x;  // camera x axis in display space; w: fx
  vec4 axis_y;  // camera y (down) axis; w: fy
  vec4 axis_z;  // camera forward; w: cx
  vec4 eye;     // camera centre; w: cy
  vec4 ground;  // floor height, model centre x, model centre z, model height
  vec4 env;     // panorama turn (radians), camera height, radius (display units), blur
  vec4 key;     // direction the main light comes from (unit), shadow strength
  uvec4 info;   // logical width, height, rotation | backdrop << 8, picture aspect (float bits)
} pc;

layout(location = 0) out vec4 out_color;

const uint kPlain = 0u, kStudio = 1u, kPhoto = 2u, kPanorama = 3u;
const float kPi = 3.14159265;

// How much light reaches the floor at xz: a pool of shade under the model and
// its shadow stretched away from the main light.
float shadow(vec2 xz) {
  const float h = max(pc.ground.w, 1e-3);
  const vec2 q = (xz - pc.ground.yz) / h;
  float lit = 1.0 - 0.42 * exp(-dot(q, q) / 0.035);
  const vec2 away = -pc.key.xz;
  const float horizontal = length(away);
  if (pc.key.w > 0.0 && horizontal > 1e-3 && pc.key.y > 0.05) {
    const vec2 dir = away / horizontal;
    // A standing model's shadow is about height / tan(elevation) long.
    const float len = clamp(horizontal / pc.key.y, 0.25, 2.5);
    const float along = dot(q, dir), across = dot(q, vec2(-dir.y, dir.x));
    const float a = (along - 0.5 * len) / (0.5 * len + 0.08);
    const float width = 0.16 + 0.04 * len;
    lit *= 1.0 - pc.key.w * exp(-a * a * 1.6 - across * across / (width * width));
  }
  return lit;
}

vec3 sample_panorama(vec3 dir, float lod) {
  const vec2 uv = vec2(atan(dir.x, -dir.z) / (2.0 * kPi) + 0.5 + pc.env.x / (2.0 * kPi),
                       acos(clamp(dir.y, -1.0, 1.0)) / kPi);
  return textureLod(picture, uv, lod).rgb;  // explicit level: no seam where u wraps
}

void main() {
  const vec2 size = vec2(pc.info.xy);
  const uint rotation = pc.info.z & 3u;
  const uint kind = pc.info.z >> 8;
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
  const vec3 eye = pc.eye.xyz;
  const float floor_y = pc.ground.x;
  const float height = max(pc.ground.w, 1e-3);
  const vec3 model = vec3(pc.ground.y, floor_y + 0.5 * height, pc.ground.z);
  const float focus = max(length(model - eye), 1e-3);
  const float levels = float(textureQueryLevels(picture));

  if (kind == kPlain) {
    out_color = vec4(0.08, 0.09, 0.11, 1.0);
    return;
  }
  if (kind == kPhoto) {
    // Fills the screen (cropping the picture) and stays put as the model
    // turns, a little out of focus behind it.
    vec2 uv = px / size;
    const float screen = size.x / size.y, aspect = max(uintBitsToFloat(pc.info.w), 1e-3);
    if (aspect > screen) uv.x = 0.5 + (uv.x - 0.5) * screen / aspect;
    else uv.y = 0.5 + (uv.y - 0.5) * aspect / screen;
    out_color = vec4(textureLod(picture, uv, min(pc.env.w * 1.2, levels - 1.0)).rgb, 1.0);
    return;
  }

  // Where the ray meets the floor (looking down, from above it).
  bool on_floor = false;
  float t_floor = 0.0;
  if (d.y < -1e-4) {
    t_floor = (floor_y - eye.y) / d.y;
    on_floor = t_floor > 0.0;
  }

  if (kind == kStudio) {
    // A white cyclorama: the floor fades into the wall with distance.
    const vec3 wall = mix(vec3(0.97), vec3(0.86, 0.87, 0.89), pow(clamp(d.y, 0.0, 1.0), 0.6));
    vec3 c = wall;
    if (on_floor) {
      const vec3 p = eye + d * t_floor;
      const vec3 floor_c = vec3(0.93) * shadow(p.xz);
      c = mix(floor_c, vec3(0.97), 1.0 - exp(-(t_floor / height) / 6.0));
    }
    out_color = vec4(c, 1.0);
    return;
  }

  // Panorama, ground projected around a centre `camera height` above the floor under the model.
  const vec3 centre = vec3(model.x, floor_y + pc.env.y, model.z);
  const float radius = pc.env.z;
  vec3 sample_dir = d;
  float dist = 1e4 * height;
  bool floor_hit = false;
  vec3 hit = vec3(0.0);
  if (on_floor) {
    hit = eye + d * t_floor;
    if (length(hit.xz - centre.xz) < radius) {
      floor_hit = true;
      sample_dir = normalize(hit - centre);
      dist = t_floor;
    }
  }
  if (!floor_hit) {
    // The surroundings: a sphere of `radius` around the centre (if the camera is inside it).
    const vec3 oc = eye - centre;
    const float b = dot(oc, d), c = dot(oc, oc) - radius * radius;
    const float disc = b * b - c;
    if (c < 0.0 && disc >= 0.0) {
      const float t = -b + sqrt(disc);
      sample_dir = normalize(eye + d * t - centre);
      dist = t;
    }
  }
  const float lod = clamp(pc.env.w * abs(log2(dist / focus)), 0.0, min(levels - 1.0, 3.5));
  vec3 c = sample_panorama(sample_dir, lod);
  if (floor_hit) c *= shadow(hit.xz);
  out_color = vec4(c, 1.0);
}
