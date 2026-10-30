#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace gs {

const char* vk_result_name(VkResult result);

class VulkanError : public std::runtime_error {
 public:
  VulkanError(const char* expr, VkResult result);
  VkResult result;
};

#define GS_VK_CHECK(expr)                                  \
  do {                                                     \
    VkResult gs_vk_result_ = (expr);                       \
    if (gs_vk_result_ != VK_SUCCESS)                       \
      throw ::gs::VulkanError(#expr, gs_vk_result_);       \
  } while (0)

// Everything queried from the physical device at startup. Reported verbatim by
// the device probe, and used to decide which optional features get enabled.
struct DeviceInfo {
  uint32_t instance_api_version = 0;
  uint32_t api_version = 0;  // effective: min(instance, device)
  VkPhysicalDeviceProperties properties{};
  VkPhysicalDeviceSubgroupProperties subgroup{};
  VkPhysicalDeviceMaintenance3Properties maintenance3{};
  bool has_driver_properties = false;
  VkPhysicalDeviceDriverProperties driver{};
  bool has_subgroup_size_control = false;
  VkPhysicalDeviceSubgroupSizeControlProperties subgroup_size_control{};
  VkPhysicalDeviceSubgroupSizeControlFeatures subgroup_size_control_features{};
  VkPhysicalDeviceFeatures features{};
  VkPhysicalDevice16BitStorageFeatures storage16{};
  bool has_float16_int8 = false;
  VkPhysicalDeviceShaderFloat16Int8Features float16_int8{};
  bool has_atomic_float = false;
  VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float{};
  VkPhysicalDeviceMemoryProperties memory{};
  std::vector<VkQueueFamilyProperties> queue_families;
  uint32_t queue_family = 0;
  std::vector<std::string> extensions;
};

// Optional features actually enabled on the logical device.
struct EnabledFeatures {
  bool atomic_float_add = false;  // shaderBufferFloat32AtomicAdd
  bool float16 = false;           // shaderFloat16 arithmetic
  bool timestamps = false;        // timestamp queries on the compute queue
};

enum class MemoryKind {
  Device,    // GPU-local, accessed through staging copies
  Upload,    // host-visible, persistently mapped
  Readback,  // host-visible (cached if possible), persistently mapped
};

class Buffer {
 public:
  Buffer() = default;
  Buffer(VkDevice device, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize size, void* mapped)
      : device_(device), buffer_(buffer), memory_(memory), size_(size), mapped_(mapped) {}
  ~Buffer() { reset(); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&& other) noexcept { *this = std::move(other); }
  Buffer& operator=(Buffer&& other) noexcept;

  void reset();
  VkBuffer handle() const { return buffer_; }
  VkDeviceSize size() const { return size_; }
  void* mapped() const { return mapped_; }

 private:
  VkDevice device_ = VK_NULL_HANDLE;
  VkBuffer buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
  VkDeviceSize size_ = 0;
  void* mapped_ = nullptr;
};

// A compute pipeline whose descriptor set 0 holds `binding_count` storage
// buffers at bindings 0..binding_count-1, plus an optional push-constant block.
class ComputePipeline {
 public:
  ComputePipeline() = default;
  ~ComputePipeline() { reset(); }
  ComputePipeline(const ComputePipeline&) = delete;
  ComputePipeline& operator=(const ComputePipeline&) = delete;
  ComputePipeline(ComputePipeline&& other) noexcept { *this = std::move(other); }
  ComputePipeline& operator=(ComputePipeline&& other) noexcept;

  void reset();

 private:
  friend class VkContext;
  VkDevice device_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout layout_ = VK_NULL_HANDLE;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
  uint32_t binding_count_ = 0;
  uint32_t push_constant_size_ = 0;
};

struct RunTiming {
  double gpu_ms = -1.0;  // from timestamp queries; negative if unavailable
  double cpu_ms = 0.0;   // wall clock around submit + wait
  double best_ms() const { return gpu_ms >= 0.0 ? gpu_ms : cpu_ms; }
};

// Headless Vulkan context with a single compute-capable queue. Work is recorded
// into a one-shot command buffer by run(), which submits and waits.
class VkContext {
 public:
  VkContext();
  ~VkContext();
  VkContext(const VkContext&) = delete;
  VkContext& operator=(const VkContext&) = delete;

  const DeviceInfo& info() const { return info_; }
  const EnabledFeatures& enabled() const { return enabled_; }
  VkDevice device() const { return device_; }
  bool has_extension(const char* name) const;

  Buffer create_buffer(VkDeviceSize size, MemoryKind kind);
  void upload(const Buffer& dst, const void* data, VkDeviceSize size);
  void download(const Buffer& src, void* data, VkDeviceSize size);

  ComputePipeline create_compute_pipeline(std::span<const uint32_t> spirv, uint32_t binding_count,
                                          uint32_t push_constant_size);

  RunTiming run(const std::function<void(VkCommandBuffer)>& record);

  // Binds `pipeline` with `buffers` at bindings 0..n-1 and dispatches.
  // Only valid inside a run() callback (descriptor sets live until the next run).
  void dispatch(VkCommandBuffer cmd, const ComputePipeline& pipeline,
                std::initializer_list<const Buffer*> buffers, const void* push, uint32_t push_size,
                uint32_t groups_x, uint32_t groups_y = 1, uint32_t groups_z = 1);

  template <typename Push>
  void dispatch(VkCommandBuffer cmd, const ComputePipeline& pipeline,
                std::initializer_list<const Buffer*> buffers, const Push& push, uint32_t groups_x,
                uint32_t groups_y = 1, uint32_t groups_z = 1) {
    dispatch(cmd, pipeline, buffers, &push, sizeof(Push), groups_x, groups_y, groups_z);
  }

  // Full compute/transfer memory barrier between consecutive commands.
  static void barrier(VkCommandBuffer cmd);

 private:
  void init();
  void destroy();
  void select_physical_device();
  void query_device_info();
  void create_device();
  uint32_t find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags required,
                            VkMemoryPropertyFlags preferred) const;

  DeviceInfo info_;
  EnabledFeatures enabled_;
  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  VkCommandPool command_pool_ = VK_NULL_HANDLE;
  VkCommandBuffer cmd_ = VK_NULL_HANDLE;
  VkFence fence_ = VK_NULL_HANDLE;
  VkQueryPool query_pool_ = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
  bool in_run_ = false;
};

inline uint32_t div_up(uint64_t value, uint32_t divisor) {
  return static_cast<uint32_t>((value + divisor - 1) / divisor);
}

std::string api_version_string(uint32_t version);

}  // namespace gs
