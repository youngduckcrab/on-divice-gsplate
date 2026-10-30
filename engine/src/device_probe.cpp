#include "gs/device_probe.h"

#include <cstdio>

#include "gs/json.h"

namespace gs {
namespace {

std::string hex(uint32_t v) {
  char buf[16];
  std::snprintf(buf, sizeof(buf), "0x%08x", v);
  return buf;
}

const char* device_type_name(VkPhysicalDeviceType type) {
  switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated";
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual";
    case VK_PHYSICAL_DEVICE_TYPE_CPU: return "cpu";
    default: return "other";
  }
}

void write_subgroup_ops(JsonWriter& w, VkSubgroupFeatureFlags ops) {
  static const std::pair<VkSubgroupFeatureFlagBits, const char*> kOps[] = {
      {VK_SUBGROUP_FEATURE_BASIC_BIT, "basic"},
      {VK_SUBGROUP_FEATURE_VOTE_BIT, "vote"},
      {VK_SUBGROUP_FEATURE_ARITHMETIC_BIT, "arithmetic"},
      {VK_SUBGROUP_FEATURE_BALLOT_BIT, "ballot"},
      {VK_SUBGROUP_FEATURE_SHUFFLE_BIT, "shuffle"},
      {VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT, "shuffle_relative"},
      {VK_SUBGROUP_FEATURE_CLUSTERED_BIT, "clustered"},
      {VK_SUBGROUP_FEATURE_QUAD_BIT, "quad"},
  };
  w.begin_array();
  for (const auto& [bit, name] : kOps) {
    if (ops & bit) w.value(name);
  }
  w.end_array();
}

void write_stages(JsonWriter& w, VkShaderStageFlags stages) {
  w.begin_array();
  if (stages & VK_SHADER_STAGE_COMPUTE_BIT) w.value("compute");
  if (stages & VK_SHADER_STAGE_FRAGMENT_BIT) w.value("fragment");
  if (stages & VK_SHADER_STAGE_VERTEX_BIT) w.value("vertex");
  w.end_array();
}

void write_memory_flags(JsonWriter& w, VkMemoryPropertyFlags flags) {
  w.begin_array();
  if (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) w.value("device_local");
  if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) w.value("host_visible");
  if (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) w.value("host_coherent");
  if (flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) w.value("host_cached");
  if (flags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) w.value("lazily_allocated");
  w.end_array();
}

}  // namespace

std::string probe_json(const VkContext& ctx) {
  const DeviceInfo& info = ctx.info();
  const VkPhysicalDeviceProperties& p = info.properties;
  const VkPhysicalDeviceLimits& lim = p.limits;
  JsonWriter w;
  w.begin_object();
  w.field("instance_api_version", api_version_string(info.instance_api_version));

  w.key("device").begin_object();
  w.field("name", p.deviceName);
  w.field("type", device_type_name(p.deviceType));
  w.field("vendor_id", hex(p.vendorID));
  w.field("device_id", hex(p.deviceID));
  w.field("api_version", api_version_string(p.apiVersion));
  w.field("effective_api_version", api_version_string(info.api_version));
  w.field("driver_version_raw", hex(p.driverVersion));
  w.field("driver_version", api_version_string(p.driverVersion));
  if (info.has_driver_properties) {
    w.field("driver_name", info.driver.driverName);
    w.field("driver_info", info.driver.driverInfo);
  }
  w.end_object();

  w.key("subgroup").begin_object();
  w.field("size", info.subgroup.subgroupSize);
  w.key("stages");
  write_stages(w, info.subgroup.supportedStages);
  w.key("ops");
  write_subgroup_ops(w, info.subgroup.supportedOperations);
  w.field("size_control", info.has_subgroup_size_control &&
                              info.subgroup_size_control_features.subgroupSizeControl);
  w.field("compute_full_subgroups", info.has_subgroup_size_control &&
                                        info.subgroup_size_control_features.computeFullSubgroups);
  if (info.has_subgroup_size_control) {
    w.field("min_size", info.subgroup_size_control.minSubgroupSize);
    w.field("max_size", info.subgroup_size_control.maxSubgroupSize);
  }
  w.end_object();

  w.key("features").begin_object();
  w.field("shader_int64", info.features.shaderInt64 == VK_TRUE);
  w.field("shader_int16", info.features.shaderInt16 == VK_TRUE);
  w.field("shader_float64", info.features.shaderFloat64 == VK_TRUE);
  w.field("shader_float16", info.has_float16_int8 && info.float16_int8.shaderFloat16);
  w.field("shader_int8", info.has_float16_int8 && info.float16_int8.shaderInt8);
  w.field("storage_buffer_16bit", info.storage16.storageBuffer16BitAccess == VK_TRUE);
  w.key("atomic_float").begin_object();
  w.field("extension", info.has_atomic_float);
  w.field("buffer_add", info.has_atomic_float && info.atomic_float.shaderBufferFloat32AtomicAdd);
  w.field("buffer_atomics", info.has_atomic_float && info.atomic_float.shaderBufferFloat32Atomics);
  w.field("shared_add", info.has_atomic_float && info.atomic_float.shaderSharedFloat32AtomicAdd);
  w.end_object();
  w.end_object();

  const EnabledFeatures& en = ctx.enabled();
  w.key("enabled").begin_object();
  w.field("atomic_float_add", en.atomic_float_add);
  w.field("float16", en.float16);
  w.field("timestamps", en.timestamps);
  w.end_object();

  w.key("limits").begin_object();
  w.field("max_compute_shared_memory_size", lim.maxComputeSharedMemorySize);
  w.field("max_compute_work_group_invocations", lim.maxComputeWorkGroupInvocations);
  w.key("max_compute_work_group_size").begin_array();
  for (uint32_t v : lim.maxComputeWorkGroupSize) w.value(v);
  w.end_array();
  w.key("max_compute_work_group_count").begin_array();
  for (uint32_t v : lim.maxComputeWorkGroupCount) w.value(v);
  w.end_array();
  w.field("max_storage_buffer_range", lim.maxStorageBufferRange);
  w.field("max_memory_allocation_size", static_cast<uint64_t>(info.maintenance3.maxMemoryAllocationSize));
  w.field("max_push_constants_size", lim.maxPushConstantsSize);
  w.field("min_storage_buffer_offset_alignment", static_cast<uint64_t>(lim.minStorageBufferOffsetAlignment));
  w.field("timestamp_period_ns", static_cast<double>(lim.timestampPeriod));
  w.field("timestamp_compute_and_graphics", lim.timestampComputeAndGraphics == VK_TRUE);
  w.end_object();

  const VkQueueFamilyProperties& q = info.queue_families[info.queue_family];
  w.key("queue").begin_object();
  w.field("family_index", info.queue_family);
  w.field("graphics", (q.queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0);
  w.field("compute", (q.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0);
  w.field("timestamp_valid_bits", q.timestampValidBits);
  w.end_object();

  w.key("memory").begin_object();
  w.key("heaps").begin_array();
  for (uint32_t i = 0; i < info.memory.memoryHeapCount; ++i) {
    const VkMemoryHeap& h = info.memory.memoryHeaps[i];
    w.begin_object();
    w.field("size_mb", static_cast<uint64_t>(h.size / (1024 * 1024)));
    w.field("device_local", (h.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0);
    w.end_object();
  }
  w.end_array();
  w.key("types").begin_array();
  for (uint32_t i = 0; i < info.memory.memoryTypeCount; ++i) {
    const VkMemoryType& t = info.memory.memoryTypes[i];
    w.begin_object();
    w.field("heap", t.heapIndex);
    w.key("flags");
    write_memory_flags(w, t.propertyFlags);
    w.end_object();
  }
  w.end_array();
  w.end_object();

  w.key("extensions").begin_array();
  for (const std::string& e : info.extensions) w.value(e);
  w.end_array();

  w.end_object();
  return w.str();
}

}  // namespace gs
