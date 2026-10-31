// Subject motion (SC-GS style): the model is the subject at rest, and in each
// view a few control nodes over it move rigidly (a turn about the node and a
// shift). A gaussian follows its 4 nearest nodes, blended by weight:
//   position     x = sum_k w_k (R_k p + b_k),  b_k = c_k - R_k n_k
//   orientation  turned by exp(sum_k w_k turn_k)
// (n_k the node at rest, c_k = n_k + shift_k where it moved to). Include after
// declaring `motion` (this view's nodes, 5 vec4 each: rows of [R_k | b_k],
// turn_k, c_k) and `bind` (2 uvec4 per slot: node indices, weights as float bits).

const uint kMotionNodeVec4 = 5u;

// exp([w]x), column-major.
mat3 rotation_from(vec3 w) {
  float th2 = dot(w, w);
  float th = sqrt(th2);
  float a = th > 1e-4 ? sin(th) / th : 1.0 - th2 / 6.0;
  float b = th > 1e-4 ? (1.0 - cos(th)) / th2 : 0.5 - th2 / 24.0;
  mat3 K = mat3(0.0, w.z, -w.y, -w.z, 0.0, w.x, w.y, -w.x, 0.0);
  return mat3(1.0) + a * K + b * (K * K);
}

// Node k's [R | b] applied to p.
vec3 node_apply(uint k, vec3 p) {
  vec4 r0 = motion[k * kMotionNodeVec4], r1 = motion[k * kMotionNodeVec4 + 1u], r2 = motion[k * kMotionNodeVec4 + 2u];
  return vec3(dot(r0.xyz, p) + r0.w, dot(r1.xyz, p) + r1.w, dot(r2.xyz, p) + r2.w);
}

mat3 node_rotation(uint k) {
  return transpose(mat3(motion[k * kMotionNodeVec4].xyz, motion[k * kMotionNodeVec4 + 1u].xyz,
                        motion[k * kMotionNodeVec4 + 2u].xyz));
}

// Slot i at rest position p in this view: its position x, the turn of its
// orientation R, and A = dx/dp.
void deform(uint i, vec3 p, out vec3 x, out mat3 R, out mat3 A) {
  uvec4 nodes = bind[i * 2u];
  vec4 w = uintBitsToFloat(bind[i * 2u + 1u]);
  x = vec3(0.0);
  A = mat3(0.0);
  vec3 turn = vec3(0.0);
  for (int j = 0; j < 4; ++j) {
    if (w[j] == 0.0) continue;
    uint k = nodes[j];
    x += w[j] * node_apply(k, p);
    A += w[j] * node_rotation(k);
    turn += w[j] * motion[k * kMotionNodeVec4 + 3u].xyz;
  }
  R = rotation_from(turn);
}
