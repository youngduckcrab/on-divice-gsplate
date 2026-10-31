// Shared definitions for the training kernels (trainer.cpp mirrors FrameData).
//
// Per-slot parameters, 4 x vec4:
//   [0] position xyz, opacity logit
//   [1] log scale xyz, alive flag (1 = in use, 0 = free slot)
//   [2] rotation quaternion (w, x, y, z), unnormalised
//   [3] SH DC rgb, unused
// SH beyond DC: `sh_stride` floats per slot, [coeff][rgb].
// Projected splat per slot, 4 x vec4:
//   [0] screen centre (px), camera depth, opacity
//   [1] conic (inverse 2D covariance a, b, c), 3-sigma radius (px)
//   [2] colour rgb, clamped-channel bits (as float bits)
//   [3] footprint: 2D covariance a, b, c (dilated) and threshold (see footprint_threshold)

struct FrameData {
  vec4 rot0;        // world -> camera rotation row 0 (OpenCV axes), w = translation x
  vec4 rot1;
  vec4 rot2;
  vec4 cam_pos;     // camera centre (world)
  vec4 intrinsics;  // fx, fy, cx, cy
  uvec4 image;      // width, height, tiles_x, tiles_y
  uvec4 model;      // slots in use, active SH degree, sh_stride (floats), flags
  vec4 background;  // rgb, w = loss scale (1 / (3 * pixels))
  vec4 lr0;         // position, log scale, rotation, opacity
  vec4 lr1;         // SH DC, SH rest, unused, unused
  vec4 adam;        // beta1, beta2, epsilon, unused
  vec4 bias;        // 1 / (1 - beta1^t), 1 / (1 - beta2^t), unused, unused
  vec4 densify0;    // grad threshold, clone max scale, prune min opacity, prune max world scale
  vec4 densify1;    // prune max screen radius, random seed (uint bits), split scale divisor, unused
  uvec4 limits;     // pair capacity (more pairs: the step is skipped), motion nodes (train_motion.glsl), unused
  vec4 loss;        // D-SSIM weight lambda (loss = (1 - lambda) L1 + lambda (1 - SSIM)), C1, C2, unused
  vec4 head;        // head region in this view: centre x, y and radius (px; 0: none), loss weight
};

// Loss weight of the pixel centred at `p`: `head.w` inside the head circle,
// fading to 1 over its outer fifth.
float pixel_weight(vec4 head, vec2 p) {
  if (head.z <= 0.0) return 1.0;
  float d = length(p - head.xy) / head.z;
  return 1.0 + (head.w - 1.0) * (1.0 - smoothstep(0.8, 1.0, d));
}

// Whether the pixel centred at `p` is in the head circle (head PSNR).
bool in_head(vec4 head, vec2 p) { return head.z > 0.0 && length(p - head.xy) <= head.z; }

const uint kFlagMaskedTarget = 1u;  // composite the target over the background using its alpha
const uint kFlagBucketState = 2u;   // the forward pass saves state for train_backward_bucket
const uint kFlagSsim = 4u;          // the forward pass saves render and target colours for train_ssim
const uint kFlagPoseGrad = 8u;      // train_step sums the gradient of the camera pose, per workgroup
const uint kFlagPoseOnly = 16u;     // only that gradient: no parameter, moment or statistics updates
const uint kFlagMotion = 32u;       // gaussians follow the subject's motion in this view (train_motion.glsl)
const uint kFlagMotionGrad = 64u;   // train_step sums the gradient of that motion per node, per workgroup

// SSIM window: 11x11 Gaussian, sigma 1.5, separable (as in 3D Gaussian Splatting).
const int kSsimRadius = 5;
const float kSsimWindow[11] = float[11](0.001028380, 0.007598758, 0.036000772, 0.109360690, 0.213005538, 0.266011725,
                                        0.213005538, 0.109360690, 0.036000772, 0.007598758, 0.001028380);

const uint kTile = 16u;
const float kNear = 0.02;
const float kFar = 1000.0;
const float kDilation = 0.3;  // px^2 low-pass, as in the viewer
const float kMinAlpha = 1.0 / 255.0;
const float kMaxAlpha = 0.99;
const float kMinTransmittance = 1e-4;

mat3 quat_to_matrix(vec4 q) {
  float w = q.x, x = q.y, y = q.z, z = q.w;
  // Column-major: each vec3 below is one column.
  return mat3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y),
              2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x),
              2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y));
}

int sh_rest_coeffs(uint degree) { return degree == 0u ? 0 : int((degree + 1u) * (degree + 1u)) - 1; }

// Real SH basis up to degree 3 (index 0 is DC), as in 3D Gaussian Splatting,
// with optional derivatives with respect to the (unit) direction.
void sh_basis(vec3 d, uint degree, out float b[16]) {
  float x = d.x, y = d.y, z = d.z;
  b[0] = 0.28209479177387814;
  if (degree >= 1u) {
    b[1] = -0.4886025119029199 * y;
    b[2] = 0.4886025119029199 * z;
    b[3] = -0.4886025119029199 * x;
  }
  if (degree >= 2u) {
    float xx = x * x, yy = y * y, zz = z * z;
    b[4] = 1.0925484305920792 * x * y;
    b[5] = -1.0925484305920792 * y * z;
    b[6] = 0.31539156525252005 * (2.0 * zz - xx - yy);
    b[7] = -1.0925484305920792 * x * z;
    b[8] = 0.5462742152960396 * (xx - yy);
    if (degree >= 3u) {
      b[9] = -0.5900435899266435 * y * (3.0 * xx - yy);
      b[10] = 2.890611442640554 * x * y * z;
      b[11] = -0.4570457994644658 * y * (4.0 * zz - xx - yy);
      b[12] = 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
      b[13] = -0.4570457994644658 * x * (4.0 * zz - xx - yy);
      b[14] = 1.445305721320277 * z * (xx - yy);
      b[15] = -0.5900435899266435 * x * (xx - 3.0 * yy);
    }
  }
}

void sh_basis_grad(vec3 d, uint degree, out vec3 g[16]) {
  float x = d.x, y = d.y, z = d.z;
  g[0] = vec3(0.0);
  if (degree >= 1u) {
    const float C1 = 0.4886025119029199;
    g[1] = vec3(0.0, -C1, 0.0);
    g[2] = vec3(0.0, 0.0, C1);
    g[3] = vec3(-C1, 0.0, 0.0);
  }
  if (degree >= 2u) {
    float xx = x * x, yy = y * y, zz = z * z;
    const float A = 1.0925484305920792, B = 0.31539156525252005, C = 0.5462742152960396;
    g[4] = A * vec3(y, x, 0.0);
    g[5] = -A * vec3(0.0, z, y);
    g[6] = B * vec3(-2.0 * x, -2.0 * y, 4.0 * z);
    g[7] = -A * vec3(z, 0.0, x);
    g[8] = C * vec3(2.0 * x, -2.0 * y, 0.0);
    if (degree >= 3u) {
      const float D = -0.5900435899266435, E = 2.890611442640554, F = -0.4570457994644658;
      const float G = 0.3731763325901154, H = 1.445305721320277;
      g[9] = D * vec3(6.0 * x * y, 3.0 * xx - 3.0 * yy, 0.0);
      g[10] = E * vec3(y * z, x * z, x * y);
      g[11] = F * vec3(-2.0 * x * y, 4.0 * zz - xx - 3.0 * yy, 8.0 * y * z);
      g[12] = G * vec3(-6.0 * x * z, -6.0 * y * z, 6.0 * zz - 3.0 * xx - 3.0 * yy);
      g[13] = F * vec3(4.0 * zz - 3.0 * xx - yy, -2.0 * x * y, 8.0 * x * z);
      g[14] = H * vec3(2.0 * x * z, -2.0 * y * z, xx - yy);
      g[15] = D * vec3(3.0 * xx - 3.0 * yy, -6.0 * x * y, 0.0);
    }
  }
}

float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }

// PCG hash based uniform in [0, 1).
uint pcg_hash(uint v) {
  uint state = v * 747796405u + 2891336453u;
  uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

float uniform01(inout uint state) {
  state = pcg_hash(state);
  return float(state >> 8u) * (1.0 / 16777216.0);
}

vec3 normal3(inout uint state) {
  float u1 = max(uniform01(state), 1e-7), u2 = uniform01(state);
  float u3 = max(uniform01(state), 1e-7), u4 = uniform01(state);
  float r1 = sqrt(-2.0 * log(u1)), r2 = sqrt(-2.0 * log(u3));
  return vec3(r1 * cos(6.283185307 * u2), r1 * sin(6.283185307 * u2), r2 * cos(6.283185307 * u4));
}

// Opacity-aware footprint. A pixel gets alpha >= kMinAlpha exactly where
// d^T cov^-1 d <= 2 ln(255 opacity), d = splat centre - pixel centre (plus a
// hair of slack for rounding). train_project counts the tiles this ellipse
// reaches and train_emit writes them, so the two must agree to the bit: the
// arithmetic is `precise` (no contractions) and both run it on the inputs
// stored in projected[3].
float footprint_threshold(float opacity) {
  precise float t = 2.0 * log(opacity * 255.0);
  return t * 1.0001 + 1e-4;
}

// [first, last] of the pixel range covering [lo, hi], within [0, size); first >
// last if empty. Clamped again after conversion, so even NaN stays in range.
ivec2 footprint_pixels(float lo, float hi, uint size) {
  int n = int(size);
  int first = clamp(int(clamp(ceil(lo), -1.0, float(n))), 0, n);
  int last = clamp(int(clamp(floor(hi), -1.0, float(n))), -1, n - 1);
  return ivec2(first, last);
}

// Pixel rows [x, y] the ellipse (centre m in px, covariance cov, threshold t)
// reaches; x > y if none.
ivec2 footprint_rows(vec2 m, vec3 cov, float t, uint height) {
  precise float ey = sqrt(t * cov.z);
  precise float v = m.y - 0.5;  // pixel centres at integer rows
  precise float lo = v - ey, hi = v + ey;
  return footprint_pixels(lo, hi, height);
}

// Tiles [x, y) of tile row `ty` the ellipse reaches within pixel rows `rows`.
uvec2 footprint_row_tiles(vec2 m, vec3 cov, float t, ivec2 rows, uint ty, uint width) {
  precise float u = m.x - 0.5, v = m.y - 0.5;
  precise float y0 = float(max(int(ty * kTile), rows.x)) - v;
  precise float y1 = float(min(int(ty * kTile + kTile - 1u), rows.y)) - v;
  // q(d) = dy^2 / c + (dx - b dy / c)^2 / (det / c), so at height dy the
  // ellipse spans dx = b dy / c -+ sqrt(det / c (t - dy^2 / c)). The right
  // end is concave in dy and the left convex, so over [y0, y1] their extremes
  // lie at the clamped heights of the rightmost and leftmost points, +-b sqrt(t / a).
  precise float a = cov.x, b = cov.y, c = cov.z;
  precise float k = b / c;
  precise float w = (a * c - b * b) / c;
  precise float side = b * sqrt(t / a);
  precise float dr = clamp(side, y0, y1);
  precise float dl = clamp(-side, y0, y1);
  precise float x_max = k * dr + sqrt(max(w * (t - dr * dr / c), 0.0));
  precise float x_min = k * dl - sqrt(max(w * (t - dl * dl / c), 0.0));
  precise float lo = u + x_min, hi = u + x_max;
  ivec2 cols = footprint_pixels(lo, hi, width);
  if (cols.x > cols.y) return uvec2(0u);
  return uvec2(uint(cols.x) / kTile, uint(cols.y) / kTile + 1u);
}

// Back-to-front state of one pixel in the backward pass.
struct PixelState {
  float T;           // transmittance in front of the current splat
  vec3 accum;        // colour of everything behind it, normalised by T
  float last_alpha;
  vec3 last_color;
};

// Gradient of one pixel's loss with respect to one projected splat (centre
// offset d = centre - pixel, conic, opacity, colour), advancing the pixel's
// back-to-front state. Output layout as in pair_grads: [0] dL/d(centre xy,
// conic a, conic b), [1] dL/d(conic c, opacity, colour r, colour g),
// [2] dL/d(colour b), |dL/dcentre xy|, unused. Zero if the splat is skipped.
void splat_pixel_grad(vec2 d, vec3 con, float opacity, vec3 color, vec3 dL_dC, float T_final, float bg_dot,
                      inout PixelState st, out vec4 g0, out vec4 g1, out vec4 g2) {
  g0 = vec4(0.0);
  g1 = vec4(0.0);
  g2 = vec4(0.0);
  float power = -0.5 * (con.x * d.x * d.x + con.z * d.y * d.y) - con.y * d.x * d.y;
  float G = exp(power);
  float raw_alpha = opacity * G;
  float alpha = min(kMaxAlpha, raw_alpha);
  if (power > 0.0 || alpha < kMinAlpha) return;
  st.T = st.T / (1.0 - alpha);
  st.accum = st.last_alpha * st.last_color + (1.0 - st.last_alpha) * st.accum;
  st.last_color = color;
  float dL_dalpha = dot(color - st.accum, dL_dC) * st.T;
  st.last_alpha = alpha;
  dL_dalpha += (-T_final / (1.0 - alpha)) * bg_dot;
  vec3 dL_dcolor = (alpha * st.T) * dL_dC;
  bool clamped = raw_alpha >= kMaxAlpha;
  float dL_dopacity = clamped ? 0.0 : G * dL_dalpha;
  float dL_dpower = clamped ? 0.0 : raw_alpha * dL_dalpha;
  vec2 dL_dmean = dL_dpower * vec2(-(con.x * d.x + con.y * d.y), -(con.z * d.y + con.y * d.x));
  vec3 dL_dconic = dL_dpower * vec3(-0.5 * d.x * d.x, -d.x * d.y, -0.5 * d.y * d.y);
  g0 = vec4(dL_dmean, dL_dconic.xy);
  g1 = vec4(dL_dconic.z, dL_dopacity, dL_dcolor.rg);
  g2 = vec4(dL_dcolor.b, abs(dL_dmean), 0.0);
}
