#version 450
// One instanced quad per visible splat, in back-to-front order.
layout(std430, binding = 0) readonly buffer Projected { uvec4 projected[]; };
layout(std430, binding = 1) readonly buffer Order { uint order[]; };

layout(push_constant) uniform Params {
  vec2 viewport;  // logical (camera) size in pixels
  uint rotation;  // surface pre-rotation in quarter turns
} pc;

layout(location = 0) out vec2 v_local;
layout(location = 1) out vec4 v_color;

void main() {
  uint idx = order[gl_InstanceIndex];
  uvec4 a = projected[idx * 2u];
  uvec4 b = projected[idx * 2u + 1u];
  // Triangle strip corners: (-2,-2), (2,-2), (-2,2), (2,2).
  vec2 corner = vec2(float(gl_VertexIndex & 1), float(gl_VertexIndex >> 1)) * 4.0 - 2.0;
  vec2 center = uintBitsToFloat(a.xy);
  vec2 axis1 = uintBitsToFloat(a.zw);
  vec2 axis2 = uintBitsToFloat(b.xy);
  vec2 px = center + corner.x * axis1 + corner.y * axis2;
  vec2 ndc = px / pc.viewport * 2.0 - 1.0;
  if (pc.rotation == 1u) {
    ndc = vec2(-ndc.y, ndc.x);
  } else if (pc.rotation == 2u) {
    ndc = -ndc;
  } else if (pc.rotation == 3u) {
    ndc = vec2(ndc.y, -ndc.x);
  }
  gl_Position = vec4(ndc, 0.0, 1.0);
  v_local = corner;
  v_color = vec4(unpackHalf2x16(b.z), unpackHalf2x16(b.w));
}
