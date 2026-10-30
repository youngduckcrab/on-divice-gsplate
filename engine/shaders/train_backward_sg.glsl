// Backward pass with subgroup adds, built from only the subgroup features
// that compile on their own on Fold6 (basic builtins, scalar float adds):
// no gl_SubgroupID, gl_NumSubgroups, subgroupElect or vector adds, which
// appear only in shaders the driver rejected. The subgroup index is taken
// as invocation / subgroup size and lane 0 leads; if a device packs
// invocations differently, the gradient check before training rejects this
// variant. Shape as in train_backward_serial.glsl (one loop with a
// buffer-derived trip count, stopping at the tile's last contributor), with
// GS_BATCH splats per round of barriers. Included by train_backward_sg1.comp
// and train_backward_sg4.comp.
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
layout(local_size_x = 16, local_size_y = 16) in;

#include "train_common.glsl"

layout(std430, binding = 0) readonly buffer FrameBuf { FrameData frame; };
layout(std430, binding = 1) readonly buffer Projected { vec4 projected[]; };
layout(std430, binding = 2) readonly buffer Ranges { uvec2 ranges[]; };
layout(std430, binding = 3) readonly buffer Values { uint values[]; };
layout(std430, binding = 4) readonly buffer PairGaussian { uint pair_gaussian[]; };
layout(std430, binding = 5) readonly buffer PixelGrad { vec4 pixel_grad[]; };
layout(std430, binding = 6) readonly buffer PixelCount { uint pixel_count[]; };
layout(std430, binding = 7) writeonly buffer PairGrads { vec4 pair_grads[]; };
layout(std430, binding = 8) readonly buffer TileCount { uint tile_count[]; };

// Same layout as the other variants; unused here.
layout(push_constant) uniform Push {
  uint batch;
  uint max_subgroups;
} pc;

const uint B = GS_BATCH;
const uint MAX_SG = 32u;                 // subgroups of 8 or more invocations
shared vec4 s_part[MAX_SG * B * 3u];     // [subgroup][splat][3]

void main() {
  uint lid = gl_LocalInvocationIndex;
  uvec2 tile = gl_WorkGroupID.xy;
  uvec2 pix = tile * kTile + gl_LocalInvocationID.xy;
  uint width = frame.image.x, height = frame.image.y;
  bool inside = pix.x < width && pix.y < height;
  uint pix_id = pix.y * width + pix.x;
  vec2 center = vec2(pix) + 0.5;
  uint tile_id = tile.y * frame.image.z + tile.x;
  uvec2 range = ranges[tile_id];

  vec4 pg = inside ? pixel_grad[pix_id] : vec4(0.0);
  vec3 dL_dC = pg.rgb;
  float T_final = pg.w;
  uint last = inside ? pixel_count[pix_id] : 0u;
  float bg_dot = dot(frame.background.rgb, dL_dC);
  PixelState st = PixelState(T_final, vec3(0.0), 0.0, vec3(0.0));

  uint n = min(range.y - range.x, tile_count[tile_id]);
  for (uint k = range.x + n + lid; k < range.y; k += kTile * kTile) {
    uint p = values[k] * 3u;
    pair_grads[p] = vec4(0.0);
    pair_grads[p + 1u] = vec4(0.0);
    pair_grads[p + 2u] = vec4(0.0);
  }

  uint sg_size = gl_SubgroupSize;
  uint sg = lid / sg_size;
  uint num_sg = min((kTile * kTile + sg_size - 1u) / sg_size, MAX_SG);
  bool leader = gl_SubgroupInvocationID == 0u && sg < MAX_SG;

  for (uint i0 = 0u; i0 < n; i0 += B) {
    for (uint b = 0u; b < B; ++b) {
      uint i = i0 + b;
      vec4 g0 = vec4(0.0), g1 = vec4(0.0), g2 = vec4(0.0);
      if (i < n && inside && n - i <= last) {  // n - i: 1-based position in the tile's range
        uint g = pair_gaussian[values[range.x + n - 1u - i]];
        vec4 q0 = projected[g * 4u];
        vec4 q1 = projected[g * 4u + 1u];
        vec4 q2 = projected[g * 4u + 2u];
        splat_pixel_grad(q0.xy - center, q1.xyz, q0.w, q2.rgb, dL_dC, T_final, bg_dot, st, g0, g1, g2);
      }
      float a0 = subgroupAdd(g0.x), a1 = subgroupAdd(g0.y), a2 = subgroupAdd(g0.z), a3 = subgroupAdd(g0.w);
      float a4 = subgroupAdd(g1.x), a5 = subgroupAdd(g1.y), a6 = subgroupAdd(g1.z), a7 = subgroupAdd(g1.w);
      float a8 = subgroupAdd(g2.x), a9 = subgroupAdd(g2.y), a10 = subgroupAdd(g2.z);
      if (leader) {
        uint r = (sg * B + b) * 3u;
        s_part[r] = vec4(a0, a1, a2, a3);
        s_part[r + 1u] = vec4(a4, a5, a6, a7);
        s_part[r + 2u] = vec4(a8, a9, a10, 0.0);
      }
    }
    memoryBarrierShared();
    barrier();
    if (lid < B * 3u) {
      uint b = lid / 3u, c = lid - b * 3u;
      if (i0 + b < n) {
        vec4 sum = vec4(0.0);
        for (uint s = 0u; s < num_sg; ++s) sum += s_part[(s * B + b) * 3u + c];
        pair_grads[values[range.x + n - 1u - (i0 + b)] * 3u + c] = sum;
      }
    }
    memoryBarrierShared();
    barrier();
  }
}
