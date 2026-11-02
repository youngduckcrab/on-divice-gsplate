#version 450
// What shows where no splat covers the screen: the viewer's dark grey, or a
// sky for places open to it. Rays come from the camera the splats are drawn
// with, in display space (y up), so the sky turns with the model.
layout(push_constant) uniform Params {
  vec4 axis_x;   // camera x axis in display space; w: fx
  vec4 axis_y;   // camera y (down) axis; w: fy
  vec4 axis_z;   // camera forward; w: cx
  vec4 eye;      // unused xyz; w: cy
  vec4 zenith;   // sky colour overhead
  vec4 horizon;  // sky colour at the horizon
  uvec4 info;    // logical width, height, rotation | backdrop << 8
} pc;

layout(location = 0) out vec4 out_color;

const uint kPlain = 0u, kSky = 1u;

void main() {
  const vec2 size = vec2(pc.info.xy);
  const uint rotation = pc.info.z & 3u;
  const uint kind = pc.info.z >> 8;
  if (kind == kPlain) {
    out_color = vec4(0.08, 0.09, 0.11, 1.0);
    return;
  }
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
  vec3 c = mix(pc.horizon.rgb, pc.zenith.rgb, pow(clamp(d.y, 0.0, 1.0), 0.6));
  if (d.y < 0.0) c = mix(pc.horizon.rgb, pc.horizon.rgb * 0.7, clamp(-d.y * 3.0, 0.0, 1.0));
  out_color = vec4(c, 1.0);
}
