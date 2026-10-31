// Training pass 6b, per gaussian: back-propagate its summed pair gradients
// (train_pair_sum) through the projection to the 3D parameters, update the
// densification statistics, and take an Adam step on the 16 parameters. The
// SH coefficients' step is train_step_sh, one thread per float, from the
// colour gradient and view direction left in sh_input.
//
// With kFlagPoseGrad it also sums, per workgroup, the gradient with respect
// to a small change of the camera pose: world -> camera becomes
// [(I + [w]x) W | (I + [w]x) t + d], so a camera-space point t moves to
// t + w x t + d. The trainer adds up the workgroups and moves the camera.
//
// With kFlagMotion the gaussians follow the subject's motion in this view
// (train_motion.glsl). Built with GS_MOTION_GRAD (train_step_motion.comp),
// each workgroup also sums the gradient with respect to every node's motion:
// a shift s of node k moves its gaussians by w_k s, a turn u (about where the
// node is) by w_k u x R_k (p - n_k) and turns their orientation by w_k u.
// train_motion_sum adds up the workgroups. (A variant of its own: its shared
// memory would cost the plain step occupancy.)
layout(local_size_x = 128) in;

#include "train_common.glsl"

layout(std430, binding = 0) readonly buffer FrameBuf { FrameData frame; };
layout(std430, binding = 1) buffer Params { vec4 params[]; };
layout(std430, binding = 2) readonly buffer Sh { float sh[]; };
layout(std430, binding = 3) buffer MomentParams { vec4 m_params[]; };
layout(std430, binding = 4) buffer VarianceParams { vec4 v_params[]; };
layout(std430, binding = 5) readonly buffer Tiles { uint tiles_touched[]; };
layout(std430, binding = 6) readonly buffer GradSum { vec4 grad_sum[]; };  // 3 per slot, pair_grads layout
layout(std430, binding = 7) readonly buffer Projected { vec4 projected[]; };
layout(std430, binding = 8) buffer Stats { vec4 stats[]; };  // grad norm sum, visible count, max radius, abs grad sum
layout(std430, binding = 9) readonly buffer Totals { uint totals[]; };  // [0] pairs this step
layout(std430, binding = 10) writeonly buffer ShInput { vec4 sh_input[]; };  // 2 per slot: dL/dcolour, view direction
layout(std430, binding = 11) writeonly buffer PoseGrad { vec4 pose_grad[]; };  // 2 per workgroup: dL/dd, dL/dw
layout(std430, binding = 12) readonly buffer Motion { vec4 motion[]; };
layout(std430, binding = 13) readonly buffer Bind { uvec4 bind[]; };
// Per workgroup and node: dL/dshift, dL/dturn (world axes).
layout(std430, binding = 14) writeonly buffer MotionGrad { vec4 motion_grad[]; };

#include "train_motion.glsl"

shared vec3 s_grad_d[128];
shared vec3 s_grad_w[128];
#ifdef GS_MOTION_GRAD
// Per thread, for the motion gradient: rest position, dL/d(position) and
// dL/d(turn of the orientation) in world axes, nodes and weights.
shared vec3 s_rest[128];
shared vec3 s_dpos[128];
shared vec3 s_dturn[128];
shared uvec4 s_nodes[128];
shared vec4 s_weights[128];
#endif

struct MotionShare {
  vec3 dpos;   // dL/d(position), world
  vec3 dturn;  // dL/d(turn of the orientation), world
  bool any;
};

vec4 adam(vec4 p, vec4 g, uint index, vec4 lr) {
  vec4 m = mix(g, m_params[index], frame.adam.x);
  vec4 v = mix(g * g, v_params[index], frame.adam.y);
  m_params[index] = m;
  v_params[index] = v;
  return p - lr * (m * frame.bias.x) / (sqrt(v * frame.bias.y) + frame.adam.z);
}

// One gaussian's step; its share of the camera pose gradient goes to grad_d,
// grad_w and its share of the motion gradient to `share`.
void step_gaussian(uint i, bool pose_only, inout vec3 grad_d, inout vec3 grad_w, inout MotionShare share) {
  vec4 p0 = params[i * 4u], p1 = params[i * 4u + 1u], q = params[i * 4u + 2u], p3 = params[i * 4u + 3u];
  if (p1.w < 0.5) return;  // free slot

  vec4 dp0 = vec4(0.0), dp1 = vec4(0.0), dq = vec4(0.0), dp3 = vec4(0.0);
  uint degree = frame.model.y;
  int coeffs = sh_rest_coeffs(degree);
  vec4 sh_colour = vec4(0.0), sh_dir = vec4(0.0);  // zero colour gradient: the SH step only decays the moments

  if (tiles_touched[i] > 0u) {
    vec4 a0 = grad_sum[i * 3u], a1 = grad_sum[i * 3u + 1u], a2 = grad_sum[i * 3u + 2u];
    vec2 dL_dmean = a0.xy;
    vec3 dL_dconic = vec3(a0.zw, a1.x);
    float dL_dopacity = a1.y;
    vec3 dL_dcolor = vec3(a1.zw, a2.x);

    // Densification statistics: the view-space positional gradient in NDC units.
    float width = float(frame.image.x), height = float(frame.image.y);
    vec2 ndc_scale = 0.5 * vec2(width, height);
    if (!pose_only) {
      vec4 st = stats[i];
      st.x += length(dL_dmean * ndc_scale);
      st.y += 1.0;
      st.z = max(st.z, projected[i * 4u + 1u].w);
      st.w += length(a2.yz * ndc_scale);
      stats[i] = st;
    }

    // Recompute the projection, of the gaussian moved with the subject: at pos,
    // its orientation turned by `turn` (identities without motion).
    vec3 pos = p0.xyz;
    mat3 turn = mat3(1.0), dxdp = mat3(1.0);
    if ((frame.model.w & kFlagMotion) != 0u) deform(i, p0.xyz, pos, turn, dxdp);
    mat3 V = transpose(mat3(frame.rot0.xyz, frame.rot1.xyz, frame.rot2.xyz));  // world -> camera
    mat3 W = V * turn;  // rest frame -> camera, for the covariance
    vec3 t = V * pos + vec3(frame.rot0.w, frame.rot1.w, frame.rot2.w);
    float fx = frame.intrinsics.x, fy = frame.intrinsics.y;
    float cx = frame.intrinsics.z, cy = frame.intrinsics.w;
    float inv_z = 1.0 / t.z;
    float lim_x = 1.3 * max(cx, width - cx) / fx;
    float lim_y = 1.3 * max(cy, height - cy) / fy;
    float rx = t.x * inv_z, ry = t.y * inv_z;
    bool clamp_x = rx < -lim_x || rx > lim_x;
    bool clamp_y = ry < -lim_y || ry > lim_y;
    float tx = clamp(rx, -lim_x, lim_x), ty = clamp(ry, -lim_y, lim_y);
    mat3 J = mat3(fx * inv_z, 0.0, 0.0, 0.0, fy * inv_z, 0.0, -fx * tx * inv_z, -fy * ty * inv_z, 0.0);
    float qlen = max(length(q), 1e-12);
    vec4 qn = q / qlen;
    mat3 R = quat_to_matrix(qn);
    vec3 s = exp(p1.xyz);
    mat3 M = mat3(R[0] * s.x, R[1] * s.y, R[2] * s.z);
    mat3 Sigma = M * transpose(M);
    mat3 T = J * W;
    mat3 cov = T * Sigma * transpose(T);
    float A = cov[0][0] + kDilation, B = cov[0][1], C = cov[1][1] + kDilation;
    float D = A * C - B * B;

    // Colour -> SH coefficients and view direction (in the rest frame).
    vec3 dir_raw = transpose(turn) * (pos - frame.cam_pos.xyz);
    float dir_len = max(length(dir_raw), 1e-12);
    vec3 dir = dir_raw / dir_len;
    uint clamped = floatBitsToUint(projected[i * 4u + 2u].w);
    if ((clamped & 1u) != 0u) dL_dcolor.r = 0.0;
    if ((clamped & 2u) != 0u) dL_dcolor.g = 0.0;
    if ((clamped & 4u) != 0u) dL_dcolor.b = 0.0;
    float basis[16];
    sh_basis(dir, degree, basis);
    dp3.xyz = basis[0] * dL_dcolor;
    sh_colour = vec4(dL_dcolor, 0.0);
    sh_dir = vec4(dir, 0.0);
    vec3 dL_ddir = vec3(0.0);
    if (coeffs > 0) {
      vec3 dbasis[16];
      sh_basis_grad(dir, degree, dbasis);
      uint base = i * frame.model.z;
      for (int k = 0; k < coeffs; ++k) {
        uint o = base + uint(k) * 3u;
        float w = dot(vec3(sh[o], sh[o + 1u], sh[o + 2u]), dL_dcolor);
        dL_ddir += w * dbasis[k + 1];
      }
    }
    vec3 dL_ddir_raw = (dL_ddir - dir * dot(dir, dL_ddir)) / dir_len;

    // Opacity (sigmoid of the logit).
    float opacity = sigmoid(p0.w);
    dp0.w = dL_dopacity * opacity * (1.0 - opacity);

    // Conic -> dilated 2D covariance [[A, B], [B, C]].
    float ga = dL_dconic.x, gb = dL_dconic.y, gc = dL_dconic.z;
    float inv_d2 = 1.0 / (D * D);
    float gA = (-C * C * ga + B * C * gb - B * B * gc) * inv_d2;
    float gB = (2.0 * B * C * ga - (D + 2.0 * B * B) * gb + 2.0 * A * B * gc) * inv_d2;
    float gC = (-B * B * ga + A * B * gb - A * A * gc) * inv_d2;
    mat3 G2 = mat3(gA, 0.5 * gB, 0.0, 0.5 * gB, gC, 0.0, 0.0, 0.0, 0.0);

    // 2D covariance -> 3D covariance and the projection Jacobian.
    mat3 dSigma = transpose(T) * G2 * T;
    mat3 dT = 2.0 * G2 * T * Sigma;
    mat3 dJ = dT * transpose(W);
    vec3 dL_dt = vec3(0.0);
    float inv_z2 = inv_z * inv_z;
    dL_dt.z += -fx * inv_z2 * dJ[0][0] - fy * inv_z2 * dJ[1][1];
    if (clamp_x) {
      dL_dt.z += fx * tx * inv_z2 * dJ[2][0];
    } else {
      dL_dt.x += -fx * inv_z2 * dJ[2][0];
      dL_dt.z += 2.0 * fx * t.x * inv_z2 * inv_z * dJ[2][0];
    }
    if (clamp_y) {
      dL_dt.z += fy * ty * inv_z2 * dJ[2][1];
    } else {
      dL_dt.y += -fy * inv_z2 * dJ[2][1];
      dL_dt.z += 2.0 * fy * t.y * inv_z2 * inv_z * dJ[2][1];
    }
    // Screen centre.
    dL_dt.x += dL_dmean.x * fx * inv_z;
    dL_dt.y += dL_dmean.y * fy * inv_z;
    dL_dt.z += -(dL_dmean.x * fx * t.x + dL_dmean.y * fy * t.y) * inv_z2;
    // dL/d(pos), then the rest position through pos = sum w (R_k p + b_k).
    vec3 dL_dx = transpose(V) * dL_dt + turn * dL_ddir_raw;
    dp0.xyz = transpose(dxdp) * dL_dx;

    // Camera pose: through the camera-space position (t moves by w x t + d)
    // and through W in the covariance (T = J W, W moves by [w]x W).
    grad_d += dL_dt;
    mat3 GW = transpose(J) * dT * transpose(W);  // dL/dW W^T; GW[c][r] is row r, column c
    vec3 dL_dturn = vec3(GW[1][2] - GW[2][1], GW[2][0] - GW[0][2], GW[0][1] - GW[1][0]);  // camera axes
    grad_w += cross(t, dL_dt) + dL_dturn;
    share.dpos = dL_dx;
    share.dturn = transpose(V) * dL_dturn;
    share.any = true;

    // 3D covariance -> scale and rotation.
    mat3 dM = 2.0 * dSigma * M;
    dp1.xyz = vec3(dot(dM[0], R[0]), dot(dM[1], R[1]), dot(dM[2], R[2])) * s;  // d/d(log scale)
    mat3 dR = mat3(dM[0] * s.x, dM[1] * s.y, dM[2] * s.z);
    // g_rc = dL/dR[row r][col c]; GLSL indexes [col][row].
    float g00 = dR[0][0], g01 = dR[1][0], g02 = dR[2][0];
    float g10 = dR[0][1], g11 = dR[1][1], g12 = dR[2][1];
    float g20 = dR[0][2], g21 = dR[1][2], g22 = dR[2][2];
    float w = qn.x, x = qn.y, y = qn.z, z = qn.w;
    vec4 dqn = 2.0 * vec4(-z * g01 + y * g02 + z * g10 - x * g12 - y * g20 + x * g21,
                          y * g01 + z * g02 + y * g10 - 2.0 * x * g11 - w * g12 + z * g20 + w * g21 - 2.0 * x * g22,
                          -2.0 * y * g00 + x * g01 + w * g02 + x * g10 + z * g12 - w * g20 + z * g21 - 2.0 * y * g22,
                          -2.0 * z * g00 - w * g01 + x * g02 + w * g10 - 2.0 * z * g11 + y * g12 + x * g20 + y * g21);
    dq = (dqn - qn * dot(qn, dqn)) / qlen;
  }

  if (pose_only) return;
  vec4 lr0 = vec4(vec3(frame.lr0.x), frame.lr0.w);
  vec4 lr1 = vec4(vec3(frame.lr0.y), 0.0);
  params[i * 4u] = adam(p0, dp0, i * 4u, lr0);
  params[i * 4u + 1u] = adam(p1, dp1, i * 4u + 1u, lr1);
  params[i * 4u + 2u] = adam(q, dq, i * 4u + 2u, vec4(frame.lr0.z));
  params[i * 4u + 3u] = adam(p3, dp3, i * 4u + 3u, vec4(vec3(frame.lr1.x), 0.0));

  sh_input[i * 2u] = sh_colour;
  sh_input[i * 2u + 1u] = sh_dir;
}

void main() {
  uint i = gl_GlobalInvocationID.x;
  uint flags = frame.model.w;
  vec3 grad_d = vec3(0.0), grad_w = vec3(0.0);
  MotionShare share = MotionShare(vec3(0.0), vec3(0.0), false);
  // More pairs than the buffers hold: some were dropped, so the gradients are
  // incomplete. Leave everything as it was; the trainer grows the buffers.
  if (i < frame.model.x && totals[0] <= frame.limits.x) {
    step_gaussian(i, (flags & kFlagPoseOnly) != 0u, grad_d, grad_w, share);
  }
  uint l = gl_LocalInvocationID.x;
  // The flags are the same for the whole dispatch, so are the barriers below.
  if ((flags & kFlagPoseGrad) != 0u) {
    s_grad_d[l] = grad_d;
    s_grad_w[l] = grad_w;
    barrier();
    for (uint stride = 64u; stride > 0u; stride >>= 1u) {
      if (l < stride) {
        s_grad_d[l] += s_grad_d[l + stride];
        s_grad_w[l] += s_grad_w[l + stride];
      }
      barrier();
    }
    if (l == 0u) {
      pose_grad[gl_WorkGroupID.x * 2u] = vec4(s_grad_d[0], 0.0);
      pose_grad[gl_WorkGroupID.x * 2u + 1u] = vec4(s_grad_w[0], 0.0);
    }
  }
#ifdef GS_MOTION_GRAD
  if ((flags & kFlagMotionGrad) != 0u) {
    // Each node's sum over this workgroup's gaussians, one thread per node
    // (at most 128): no float atomics.
    s_rest[l] = share.any ? params[i * 4u].xyz : vec3(0.0);
    s_dpos[l] = share.dpos;
    s_dturn[l] = share.dturn;
    s_nodes[l] = share.any ? bind[i * 2u] : uvec4(0u);
    s_weights[l] = share.any ? uintBitsToFloat(bind[i * 2u + 1u]) : vec4(0.0);
    barrier();
    uint nodes = frame.limits.y;
    if (l < nodes) {
      vec4 c = motion[l * kMotionNodeVec4 + 4u];
      vec3 shift = vec3(0.0), turn = vec3(0.0);
      for (uint j = 0u; j < 128u; ++j) {
        uvec4 n = s_nodes[j];
        vec4 w = s_weights[j];
        float wl = n.x == l ? w.x : n.y == l ? w.y : n.z == l ? w.z : n.w == l ? w.w : 0.0;
        if (wl == 0.0) continue;
        vec3 arm = node_apply(l, s_rest[j]) - c.xyz;  // R_l (p - n_l)
        shift += wl * s_dpos[j];
        turn += wl * (s_dturn[j] + cross(arm, s_dpos[j]));
      }
      uint o = (gl_WorkGroupID.x * nodes + l) * 2u;
      motion_grad[o] = vec4(shift, 0.0);
      motion_grad[o + 1u] = vec4(turn, 0.0);
    }
  }
#endif
}
