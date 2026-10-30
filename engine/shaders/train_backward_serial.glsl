// Backward pass without subgroup operations, shaped like the forward pass:
// one loop over the tile's splats (back to front) whose trip count comes
// from a buffer, every thread reading the splat directly, and the sum over
// the tile's pixels in two shared-memory stages (rows of 16, then the 16
// rows). On Fold6 the batched shared-memory variant (train_backward.glsl
// with barriers in nested loops) returned zero gradients.
//
// With GS_TILE_CUTOFF the loop stops at the tile's furthest last contributor
// (tile_count, written by the forward pass); the pairs behind it get zeros.
// GS_BATCH splats (1 or 2) share each round of barriers.
// Included by train_backward_cut.comp and train_backward_cut2.comp
// (train_backward_serial.comp is the v0.3.5 original, kept as verified on Fold6).
#ifndef GS_BATCH
#define GS_BATCH 1
#endif
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
shared vec4 s_red[B * 256u * 3u];  // [splat][pixel][3]
shared vec4 s_row[B * 16u * 3u];   // [splat][row of the tile][3]

void main() {
  uint lid = gl_LocalInvocationIndex;
  uvec2 tile = gl_WorkGroupID.xy;
  uvec2 pix = tile * kTile + gl_LocalInvocationID.xy;
  uint width = frame.image.x, height = frame.image.y;
  bool inside = pix.x < width && pix.y < height;
  uint pix_id = pix.y * width + pix.x;
  vec2 center = vec2(pix) + 0.5;
  uvec2 range = ranges[tile.y * frame.image.z + tile.x];

  vec4 pg = inside ? pixel_grad[pix_id] : vec4(0.0);
  vec3 dL_dC = pg.rgb;
  float T_final = pg.w;
  uint last = inside ? pixel_count[pix_id] : 0u;
  float bg_dot = dot(frame.background.rgb, dL_dC);
  PixelState st = PixelState(T_final, vec3(0.0), 0.0, vec3(0.0));

  uint n = range.y - range.x;
#ifdef GS_TILE_CUTOFF
  n = min(n, tile_count[tile.y * frame.image.z + tile.x]);
  for (uint k = range.x + n + lid; k < range.y; k += kTile * kTile) {
    uint p = values[k] * 3u;
    pair_grads[p] = vec4(0.0);
    pair_grads[p + 1u] = vec4(0.0);
    pair_grads[p + 2u] = vec4(0.0);
  }
#endif
  for (uint i0 = 0u; i0 < n; i0 += B) {
    // Each pixel's gradients for the next B splats, back to front.
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
      uint r = (b * 256u + lid) * 3u;
      s_red[r] = g0;
      s_red[r + 1u] = g1;
      s_red[r + 2u] = g2;
    }
    memoryBarrierShared();
    barrier();
    for (uint t = lid; t < B * 48u; t += kTile * kTile) {
      uint b = t / 48u, rc = t - b * 48u, row = rc / 3u, c = rc - row * 3u;
      vec4 sum = vec4(0.0);
      for (uint x = 0u; x < 16u; ++x) sum += s_red[(b * 256u + row * 16u + x) * 3u + c];
      s_row[(b * 16u + row) * 3u + c] = sum;
    }
    memoryBarrierShared();
    barrier();
    if (lid < B * 3u) {
      uint b = lid / 3u, c = lid - b * 3u;
      if (i0 + b < n) {
        vec4 sum = vec4(0.0);
        for (uint r = 0u; r < 16u; ++r) sum += s_row[(b * 16u + r) * 3u + c];
        pair_grads[values[range.x + n - 1u - (i0 + b)] * 3u + c] = sum;
      }
    }
  }
}
