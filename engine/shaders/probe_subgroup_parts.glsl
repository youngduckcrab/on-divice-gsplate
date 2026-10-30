// One subgroup feature per probe (PROBE 1..5), to find which one a driver
// rejects. Each writes something to result[] so it is not optimised away.
#extension GL_KHR_shader_subgroup_basic : require
#if PROBE == 4
#extension GL_KHR_shader_subgroup_arithmetic : require
#endif
#if PROBE == 5
#extension GL_KHR_shader_subgroup_ballot : require
#endif
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer Out { uint result[]; };

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
#endif
}
