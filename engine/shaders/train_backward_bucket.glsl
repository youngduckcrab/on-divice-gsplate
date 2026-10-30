// Backward pass, one thread per splat ("Taming 3DGS", Mallick et al. 2024):
// the per-pixel variants sum every splat's gradient over the tile's 256
// pixels, which costs barriers and mostly idle threads per splat. Here 32
// lanes take 32 consecutive splats of a tile (a bucket) and sweep its pixels
// as a pipeline: at step `it` lane i handles pixel group it - i (GS_PIXELS
// pixels), front to back, and hands each pixel's state (transmittance,
// remaining colour) to lane i + 1 through shared memory. Each lane sums its
// splat's gradient in registers, so nothing is reduced across threads.
//
// The forward pass stored each pixel's state at the start of every bucket
// after the first (bucket_state), its count of pairs up to the last
// contributor, and D = dL/dC . C (the pixel colour over the background), and
// listed the buckets to process (bucket_list: count, then tile << 16 | bucket).
// Front to back, with R = dL/dC . (colour of this splat and everything
// behind it, background included):
//   dL/dalpha_k = T_k (dL/dC . c_k) - R_{k+1} / (1 - alpha_k).
// Included by train_backward_bucket.comp (1 pixel per step) and
// train_backward_bucket2.comp (2: half the barriers, longer pipeline fill).
#ifndef GS_PIXELS
#define GS_PIXELS 1
#endif
layout(local_size_x = 256) in;

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
layout(std430, binding = 9) readonly buffer BucketState { vec4 bucket_state[]; };  // front colour, T
layout(std430, binding = 10) readonly buffer BucketList { uint bucket_list[]; };
layout(std430, binding = 11) readonly buffer PixelDot { float pixel_dot[]; };

// Same layout as the other variants; unused here.
layout(push_constant) uniform Push {
  uint batch;
  uint max_subgroups;
} pc;

const uint kLanes = 32u;   // splats per bucket
const uint kGroups = 8u;   // buckets per workgroup
const uint kPixels = kTile * kTile;
const uint P = GS_PIXELS;
const uint kSteps = kPixels / P;
const uint kSlots = kGroups * kLanes * P;

// Pixel state between lanes, double-buffered by step parity:
// [0] T, R, dL/dC r, g; [1] dL/dC b, pixel count (bits).
shared vec4 s_a[2u * kSlots];
shared vec2 s_b[2u * kSlots];

// The state of pixel `p` of tile `t` at the start of bucket `k` (lane 0's input).
void load_pixel(uint t, uint k, uint p, uvec2 range, out vec4 a, out vec2 b) {
  a = vec4(0.0);
  b = vec2(0.0, uintBitsToFloat(0u));
  uint tx = t % frame.image.z, ty = t / frame.image.z;
  uvec2 pix = uvec2(tx, ty) * kTile + uvec2(p % kTile, p / kTile);
  if (pix.x >= frame.image.x || pix.y >= frame.image.y) return;
  uint pix_id = pix.y * frame.image.x + pix.x;
  uint count = pixel_count[pix_id];
  if (count <= k * kLanes) return;  // done before this bucket
  vec3 g = pixel_grad[pix_id].rgb;
  vec4 front = k == 0u ? vec4(0.0, 0.0, 0.0, 1.0)
                       : bucket_state[((range.x / kLanes + t) + k - 1u) * kPixels + p];
  a = vec4(front.w, pixel_dot[pix_id] - dot(g, front.rgb), g.rg);
  b = vec2(g.b, uintBitsToFloat(count));
}

void main() {
  uint lid = gl_LocalInvocationIndex;
  uint group = lid / kLanes, lane = lid % kLanes;
  uint total = bucket_list[0];
  if (gl_WorkGroupID.x * kGroups >= total) return;  // the whole workgroup: uniform
  uint id = gl_WorkGroupID.x * kGroups + group;
  bool listed = id < total;
  uint entry = listed ? bucket_list[1u + id] : 0u;
  uint t = entry >> 16u, k = entry & 0xffffu;
  uvec2 range = listed ? ranges[t] : uvec2(0u);
  uint len = range.y - range.x;
  uint n = listed ? min(len, tile_count[t]) : 0u;
  uint q = k * kLanes + lane;  // this lane's splat: position in the tile, front to back
  bool valid = q < n;

  vec4 q0 = vec4(0.0), q1 = vec4(0.0), q2 = vec4(0.0);
  if (valid) {
    uint g = pair_gaussian[values[range.x + q]];
    q0 = projected[g * 4u];
    q1 = projected[g * 4u + 1u];
    q2 = projected[g * 4u + 2u];
  }
  vec2 tile_origin = vec2(uvec2(t % frame.image.z, t / frame.image.z) * kTile) + 0.5;
  vec4 a0 = vec4(0.0), a1 = vec4(0.0), a2 = vec4(0.0);

  // Lane 0 reads one step ahead so the loads overlap the previous step.
  vec4 next_a[P];
  vec2 next_b[P];
  for (uint r = 0u; r < P; ++r) {
    next_a[r] = vec4(0.0);
    next_b[r] = vec2(0.0);
    if (lane == 0u && listed) load_pixel(t, k, r, range, next_a[r], next_b[r]);
  }

  for (uint it = 0u; it < kSteps + kLanes - 1u; ++it) {
    int s = int(it) - int(lane);  // this lane's pixel group
    if (s >= 0 && s < int(kSteps)) {
      for (uint r = 0u; r < P; ++r) {
        uint p = uint(s) * P + r;
        vec4 sa;
        vec2 sb;
        if (lane == 0u) {
          sa = next_a[r];
          sb = next_b[r];
          if (listed && p + P < kPixels) load_pixel(t, k, p + P, range, next_a[r], next_b[r]);
        } else {
          uint src = ((it - 1u) & 1u) * kSlots + (lid - 1u) * P + r;
          sa = s_a[src];
          sb = s_b[src];
        }
        if (valid && q < floatBitsToUint(sb.y)) {
          vec2 d = q0.xy - (tile_origin + vec2(p % kTile, p / kTile));
          vec3 con = q1.xyz;
          float power = -0.5 * (con.x * d.x * d.x + con.z * d.y * d.y) - con.y * d.x * d.y;
          float G = exp(power);
          float raw_alpha = q0.w * G;
          float alpha = min(kMaxAlpha, raw_alpha);
          if (power <= 0.0 && alpha >= kMinAlpha) {
            vec3 dL_dC = vec3(sa.zw, sb.x);
            float T = sa.x;
            float gc = dot(dL_dC, q2.rgb);
            float w = alpha * T;
            float rest = sa.y - gc * w;  // R after this splat
            float dL_dalpha = T * gc - rest / (1.0 - alpha);
            vec3 dL_dcolor = w * dL_dC;
            bool clamped = raw_alpha >= kMaxAlpha;
            float dL_dopacity = clamped ? 0.0 : G * dL_dalpha;
            float dL_dpower = clamped ? 0.0 : raw_alpha * dL_dalpha;
            vec2 dL_dmean = dL_dpower * vec2(-(con.x * d.x + con.y * d.y), -(con.z * d.y + con.y * d.x));
            vec3 dL_dconic = dL_dpower * vec3(-0.5 * d.x * d.x, -d.x * d.y, -0.5 * d.y * d.y);
            a0 += vec4(dL_dmean, dL_dconic.xy);
            a1 += vec4(dL_dconic.z, dL_dopacity, dL_dcolor.rg);
            a2 += vec4(dL_dcolor.b, abs(dL_dmean), 0.0);
            sa.x = T * (1.0 - alpha);
            sa.y = rest;
          }
        }
        uint dst = (it & 1u) * kSlots + lid * P + r;
        s_a[dst] = sa;
        s_b[dst] = sb;
      }
    }
    memoryBarrierShared();
    barrier();
  }

  if (valid) {
    uint o = values[range.x + q] * 3u;
    pair_grads[o] = a0;
    pair_grads[o + 1u] = a1;
    pair_grads[o + 2u] = a2;
  }
  // Pairs behind the tile's last contributor get zeros, from its last bucket.
  uint buckets = max(1u, (n + kLanes - 1u) / kLanes);
  if (listed && k == buckets - 1u) {
    for (uint j = n + lane; j < len; j += kLanes) {
      uint o = values[range.x + j] * 3u;
      pair_grads[o] = vec4(0.0);
      pair_grads[o + 1u] = vec4(0.0);
      pair_grads[o + 2u] = vec4(0.0);
    }
  }
}
