// Training pass 5: back-to-front gradient of every (tile, gaussian) pair with
// respect to the projected splat, one 16x16 tile per workgroup. Included by
// train_backward.comp and train_backward_shared.comp.
//
// Mobile GPUs lack float atomics, so instead of accumulating into the
// gaussian, each pair's gradient is summed over the tile's pixels and stored
// at the pair's pre-sort index; train_step then sums each gaussian's
// contiguous pairs. The sum over pixels uses subgroup adds (vec4, or scalar
// with GS_SUBGROUP_SCALAR) and then shared memory across subgroups, or, with
// GS_SHARED_REDUCTION (drivers without usable subgroup arithmetic), a slower
// shared-memory tree per gaussian. No atomics: Fold6 returned all-zero
// gradients when the tile's pair count came from a shared atomicMax.
#ifndef GS_SHARED_REDUCTION
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#endif
layout(local_size_x = 16, local_size_y = 16) in;

#include "train_common.glsl"

layout(push_constant) uniform Push {
  uint batch;          // gaussians per round, <= MAX_BATCH
  uint max_subgroups;  // subgroups that fit the reduction slots (batch * 3 each)
} pc;

const uint MAX_BATCH = 32u;

layout(std430, binding = 0) readonly buffer FrameBuf { FrameData frame; };
layout(std430, binding = 1) readonly buffer Projected { vec4 projected[]; };
layout(std430, binding = 2) readonly buffer Ranges { uvec2 ranges[]; };
layout(std430, binding = 3) readonly buffer Values { uint values[]; };
layout(std430, binding = 4) readonly buffer PairGaussian { uint pair_gaussian[]; };
layout(std430, binding = 5) readonly buffer PixelGrad { vec4 pixel_grad[]; };
layout(std430, binding = 6) readonly buffer PixelCount { uint pixel_count[]; };
// Per pair, 3 x vec4: [0] dL/d(centre xy, conic a, conic b), [1] dL/d(conic c,
// opacity, colour r, colour g), [2] dL/d(colour b), sum |dL/dcentre xy|, unused.
layout(std430, binding = 7) writeonly buffer PairGrads { vec4 pair_grads[]; };

shared vec4 s_g0[MAX_BATCH];  // centre xy, opacity
shared vec4 s_g1[MAX_BATCH];  // conic abc
shared vec4 s_g2[MAX_BATCH];  // colour
shared uint s_pair[MAX_BATCH];
#ifdef GS_SHARED_REDUCTION
shared vec4 s_red[256u * 3u];  // one gaussian at a time, every pixel
#else
shared vec4 s_red[512u];  // [subgroup][gaussian][3]
#endif
shared uint s_last[256];

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

  // The tile's furthest last contributor, by a shared-memory tree.
  s_last[lid] = last;
  barrier();
  for (uint stride = 128u; stride > 0u; stride >>= 1u) {
    if (lid < stride) s_last[lid] = max(s_last[lid], s_last[lid + stride]);
    barrier();
  }
  uint count = s_last[0];
  uint end = range.x + count;

  // Pairs behind every pixel's last contributor get no gradient.
  for (uint k = end + lid; k < range.y; k += kTile * kTile) {
    uint p = values[k] * 3u;
    pair_grads[p] = vec4(0.0);
    pair_grads[p + 1u] = vec4(0.0);
    pair_grads[p + 2u] = vec4(0.0);
  }

  float T = T_final;
  vec3 accum = vec3(0.0);
  float last_alpha = 0.0;
  vec3 last_color = vec3(0.0);
  uint batch = min(pc.batch, MAX_BATCH);
#ifndef GS_SHARED_REDUCTION
  uint num_subgroups = min(gl_NumSubgroups, pc.max_subgroups);
#endif

  for (uint done_n = 0u; done_n < count; done_n += batch) {
    uint nb = min(batch, count - done_n);
    barrier();
    if (lid < nb) {
      uint p = values[end - 1u - done_n - lid];
      uint g = pair_gaussian[p];
      vec4 q0 = projected[g * 4u];
      s_g0[lid] = vec4(q0.xy, q0.w, 0.0);
      s_g1[lid] = projected[g * 4u + 1u];
      s_g2[lid] = projected[g * 4u + 2u];
      s_pair[lid] = p;
    }
    barrier();

    for (uint j = 0u; j < nb; ++j) {
      uint rel = count - done_n - j;  // 1-based position in the tile's range
      vec4 g0 = vec4(0.0), g1 = vec4(0.0), g2 = vec4(0.0);
      if (inside && rel <= last) {
        vec2 d = s_g0[j].xy - center;
        vec3 con = s_g1[j].xyz;
        float opacity = s_g0[j].z;
        float power = -0.5 * (con.x * d.x * d.x + con.z * d.y * d.y) - con.y * d.x * d.y;
        float G = exp(power);
        float raw_alpha = opacity * G;
        float alpha = min(kMaxAlpha, raw_alpha);
        if (power <= 0.0 && alpha >= kMinAlpha) {
          vec3 color = s_g2[j].rgb;
          T = T / (1.0 - alpha);
          accum = last_alpha * last_color + (1.0 - last_alpha) * accum;
          last_color = color;
          float dL_dalpha = dot(color - accum, dL_dC) * T;
          last_alpha = alpha;
          dL_dalpha += (-T_final / (1.0 - alpha)) * bg_dot;
          vec3 dL_dcolor = (alpha * T) * dL_dC;
          bool clamped = raw_alpha >= kMaxAlpha;
          float dL_dopacity = clamped ? 0.0 : G * dL_dalpha;
          float dL_dpower = clamped ? 0.0 : raw_alpha * dL_dalpha;
          vec2 dL_dmean = dL_dpower * vec2(-(con.x * d.x + con.y * d.y), -(con.z * d.y + con.y * d.x));
          vec3 dL_dconic = dL_dpower * vec3(-0.5 * d.x * d.x, -d.x * d.y, -0.5 * d.y * d.y);
          g0 = vec4(dL_dmean, dL_dconic.xy);
          g1 = vec4(dL_dconic.z, dL_dopacity, dL_dcolor.rg);
          g2 = vec4(dL_dcolor.b, abs(dL_dmean), 0.0);
        }
      }
#ifdef GS_SHARED_REDUCTION
      s_red[lid * 3u] = g0;
      s_red[lid * 3u + 1u] = g1;
      s_red[lid * 3u + 2u] = g2;
      barrier();
      for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
          for (uint c = 0u; c < 3u; ++c) s_red[lid * 3u + c] += s_red[(lid + stride) * 3u + c];
        }
        barrier();
      }
      if (lid < 3u) pair_grads[s_pair[j] * 3u + lid] = s_red[lid];
      barrier();
#else
#ifdef GS_SUBGROUP_SCALAR
      g0 = vec4(subgroupAdd(g0.x), subgroupAdd(g0.y), subgroupAdd(g0.z), subgroupAdd(g0.w));
      g1 = vec4(subgroupAdd(g1.x), subgroupAdd(g1.y), subgroupAdd(g1.z), subgroupAdd(g1.w));
      g2 = vec4(subgroupAdd(g2.x), subgroupAdd(g2.y), subgroupAdd(g2.z), 0.0);
#else
      g0 = subgroupAdd(g0);
      g1 = subgroupAdd(g1);
      g2 = subgroupAdd(g2);
#endif
      if (subgroupElect() && gl_SubgroupID < num_subgroups) {
        uint r = (gl_SubgroupID * batch + j) * 3u;
        s_red[r] = g0;
        s_red[r + 1u] = g1;
        s_red[r + 2u] = g2;
      }
#endif
    }
    barrier();

#ifndef GS_SHARED_REDUCTION
    for (uint idx = lid; idx < nb * 3u; idx += kTile * kTile) {
      uint j = idx / 3u, c = idx - j * 3u;
      vec4 sum = vec4(0.0);
      for (uint sg = 0u; sg < num_subgroups; ++sg) sum += s_red[(sg * batch + j) * 3u + c];
      pair_grads[s_pair[j] * 3u + c] = sum;
    }
#endif
  }
}
