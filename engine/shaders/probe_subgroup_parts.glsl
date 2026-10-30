// One subgroup feature per probe (PROBE 1..5), to find which one a driver
// rejects. Each writes something to result[] so it is not optimised away.
#extension GL_KHR_shader_subgroup_basic : require
#if PROBE >= 4 && PROBE != 5
#extension GL_KHR_shader_subgroup_arithmetic : require
#endif
#if PROBE == 5
#extension GL_KHR_shader_subgroup_ballot : require
#endif
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer Out { uint result[]; };

shared float s_x[256];

void main() {
  uint lid = gl_LocalInvocationIndex;
#if PROBE == 1
  result[lid] = gl_SubgroupSize * 1000u + gl_SubgroupInvocationID;  // basic builtins
#elif PROBE == 2
  result[lid] = subgroupElect() ? 1u : 0u;
#elif PROBE == 3
  result[lid] = gl_SubgroupID * 1000u + gl_NumSubgroups;
#elif PROBE == 4
  result[lid] = uint(subgroupAdd(float(lid)));
#elif PROBE == 5
  result[lid] = subgroupBallot(true).x;
#elif PROBE == 6  // vec4 add
  result[lid] = uint(subgroupAdd(vec4(1.0, float(lid), 2.0, 0.5)).y);
#elif PROBE == 7  // add + subgroup id
  float s = subgroupAdd(float(lid));
  result[gl_SubgroupID] = uint(s);
#elif PROBE == 8  // add + elect
  float s = subgroupAdd(float(lid));
  if (subgroupElect()) result[lid] = uint(s);
#elif PROBE == 9  // add + number of subgroups
  result[lid] = uint(subgroupAdd(float(lid))) + gl_NumSubgroups;
#elif PROBE == 10  // add + shared memory + barrier
  s_x[lid] = subgroupAdd(float(lid));
  barrier();
  result[lid] = uint(s_x[255u - lid]);
#elif PROBE == 11  // add in a loop with barriers
  float acc = 0.0;
  for (uint i = 0u; i < result[0] + 4u; ++i) {
    acc += subgroupAdd(float(lid + i));
    barrier();
  }
  result[lid] = uint(acc);
#elif PROBE == 12  // the full probe with scalar adds only
  float s = subgroupAdd(float(lid));
  if (subgroupElect() && gl_SubgroupID < 63u) {
    result[gl_SubgroupID * 4u] = uint(s);
    result[gl_SubgroupID * 4u + 3u] = gl_SubgroupSize;
  }
  if (lid == 0u) result[255] = gl_NumSubgroups;
#endif
}
