#include "gs/vk_context.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <utility>

#include "gs/log.h"

namespace gs {

const char* vk_result_name(VkResult result) {
  switch (result) {
#define GS_CASE(x) \
  case x:          \
    return #x;
    GS_CASE(VK_SUCCESS)
    GS_CASE(VK_NOT_READY)
    GS_CASE(VK_TIMEOUT)
    GS_CASE(VK_INCOMPLETE)
    GS_CASE(VK_ERROR_OUT_OF_HOST_MEMORY)
    GS_CASE(VK_ERROR_OUT_OF_DEVICE_MEMORY)
    GS_CASE(VK_ERROR_INITIALIZATION_FAILED)
    GS_CASE(VK_ERROR_DEVICE_LOST)
    GS_CASE(VK_ERROR_MEMORY_MAP_FAILED)
    GS_CASE(VK_ERROR_LAYER_NOT_PRESENT)
    GS_CASE(VK_ERROR_EXTENSION_NOT_PRESENT)
    GS_CASE(VK_ERROR_FEATURE_NOT_PRESENT)
    GS_CASE(VK_ERROR_INCOMPATIBLE_DRIVER)
    GS_CASE(VK_ERROR_TOO_MANY_OBJECTS)
    GS_CASE(VK_ERROR_FORMAT_NOT_SUPPORTED)
    GS_CASE(VK_ERROR_FRAGMENTED_POOL)
    GS_CASE(VK_ERROR_OUT_OF_POOL_MEMORY)
#undef GS_CASE
    default:
      return "VK_ERROR_UNKNOWN";
  }
}

VulkanError::VulkanError(const char* expr, VkResult r)
    : std::runtime_error(std::string(expr) + " failed: " + vk_result_name(r)), result(r) {}

std::string api_version_string(uint32_t version) {
  return std::to_string(VK_API_VERSION_MAJOR(version)) + "." +
         std::to_string(VK_API_VERSION_MINOR(version)) + "." +
         std::to_string(VK_API_VERSION_PATCH(version));
}

// --- Buffer / ComputePipeline -------------------------------------------------

Buffer& Buffer::operator=(Buffer&& other) noexcept {
  if (this != &other) {
    reset();
    device_ = std::exchange(other.device_, VK_NULL_HANDLE);
    buffer_ = std::exchange(other.buffer_, VK_NULL_HANDLE);
    memory_ = std::exchange(other.memory_, VK_NULL_HANDLE);
    size_ = std::exchange(other.size_, 0);
    mapped_ = std::exchange(other.mapped_, nullptr);
  }
  return *this;
}

void Buffer::reset() {
  if (device_ == VK_NULL_HANDLE) return;
  if (buffer_ != VK_NULL_HANDLE) vkDestroyBuffer(device_, buffer_, nullptr);
  if (memory_ != VK_NULL_HANDLE) vkFreeMemory(device_, memory_, nullptr);  // implicitly unmaps
  device_ = VK_NULL_HANDLE;
  buffer_ = VK_NULL_HANDLE;
  memory_ = VK_NULL_HANDLE;
  size_ = 0;
  mapped_ = nullptr;
}

ComputePipeline& ComputePipeline::operator=(ComputePipeline&& other) noexcept {
  if (this != &other) {
    reset();
    device_ = std::exchange(other.device_, VK_NULL_HANDLE);
    set_layout_ = std::exchange(other.set_layout_, VK_NULL_HANDLE);
    layout_ = std::exchange(other.layout_, VK_NULL_HANDLE);
    pipeline_ = std::exchange(other.pipeline_, VK_NULL_HANDLE);
    binding_count_ = std::exchange(other.binding_count_, 0);
    push_constant_size_ = std::exchange(other.push_constant_size_, 0);
  }
  return *this;
}

void ComputePipeline::reset() {
  if (device_ == VK_NULL_HANDLE) return;
  if (pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, pipeline_, nullptr);
  if (layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, layout_, nullptr);
  if (set_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, set_layout_, nullptr);
  device_ = VK_NULL_HANDLE;
  pipeline_ = VK_NULL_HANDLE;
  layout_ = VK_NULL_HANDLE;
  set_layout_ = VK_NULL_HANDLE;
}

// --- VkContext ------------------------------------------------------------------

VkContext::VkContext() {
  try {
    init();
  } catch (...) {
    destroy();
    throw;
  }
}

VkContext::~VkContext() { destroy(); }

bool VkContext::has_extension(const char* name) const {
  return std::find(info_.extensions.begin(), info_.extensions.end(), name) != info_.extensions.end();
}

void VkContext::init() {
  uint32_t loader_version = VK_API_VERSION_1_0;
  auto enumerate_version = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
      vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
  if (enumerate_version != nullptr) enumerate_version(&loader_version);
  if (loader_version < VK_API_VERSION_1_1) {
    throw std::runtime_error("Vulkan 1.1 is required (loader reports " +
                             api_version_string(loader_version) + ")");
  }
  info_.instance_api_version = std::min<uint32_t>(loader_version, VK_API_VERSION_1_3);

  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "gsplat";
  app.applicationVersion = 1;
  app.pEngineName = "gsengine";
  app.engineVersion = 1;
  app.apiVersion = info_.instance_api_version;

  // Surface extensions are enabled whenever the platform offers them, so the same
  // context can drive both compute work and an on-screen viewer.
  uint32_t ext_count = 0;
  GS_VK_CHECK(vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, nullptr));
  std::vector<VkExtensionProperties> available(ext_count);
  GS_VK_CHECK(vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, available.data()));
  auto instance_has = [&](const char* name) {
    return std::any_of(available.begin(), available.end(),
                       [&](const VkExtensionProperties& e) { return std::strcmp(e.extensionName, name) == 0; });
  };
  std::vector<const char*> instance_extensions;
  const char* kAndroidSurface = "VK_KHR_android_surface";
  if (instance_has(VK_KHR_SURFACE_EXTENSION_NAME) && instance_has(kAndroidSurface)) {
    instance_extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
    instance_extensions.push_back(kAndroidSurface);
    enabled_.presentation = true;
  }

  VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  instance_info.pApplicationInfo = &app;
  instance_info.enabledExtensionCount = static_cast<uint32_t>(instance_extensions.size());
  instance_info.ppEnabledExtensionNames = instance_extensions.data();
  GS_VK_CHECK(vkCreateInstance(&instance_info, nullptr, &instance_));

  select_physical_device();
  query_device_info();
  create_device();

  VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pool_info.queueFamilyIndex = info_.queue_family;
  GS_VK_CHECK(vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_));

  VkCommandBufferAllocateInfo cmd_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cmd_info.commandPool = command_pool_;
  cmd_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmd_info.commandBufferCount = 1;
  GS_VK_CHECK(vkAllocateCommandBuffers(device_, &cmd_info, &cmd_));

  VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  GS_VK_CHECK(vkCreateFence(device_, &fence_info, nullptr, &fence_));

  if (enabled_.timestamps) {
    VkQueryPoolCreateInfo query_info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    query_info.queryCount = 2;
    GS_VK_CHECK(vkCreateQueryPool(device_, &query_info, nullptr, &query_pool_));
  }

  VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096};
  VkDescriptorPoolCreateInfo desc_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  desc_info.maxSets = 1024;
  desc_info.poolSizeCount = 1;
  desc_info.pPoolSizes = &pool_size;
  GS_VK_CHECK(vkCreateDescriptorPool(device_, &desc_info, nullptr, &descriptor_pool_));

  log_info("Vulkan device: %s (api %s, subgroup %u)", info_.properties.deviceName,
           api_version_string(info_.api_version).c_str(), info_.subgroup.subgroupSize);
}

void VkContext::destroy() {
  if (device_ != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(device_);
    if (descriptor_pool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
    if (query_pool_ != VK_NULL_HANDLE) vkDestroyQueryPool(device_, query_pool_, nullptr);
    if (fence_ != VK_NULL_HANDLE) vkDestroyFence(device_, fence_, nullptr);
    if (command_pool_ != VK_NULL_HANDLE) vkDestroyCommandPool(device_, command_pool_, nullptr);
    vkDestroyDevice(device_, nullptr);
  }
  if (instance_ != VK_NULL_HANDLE) vkDestroyInstance(instance_, nullptr);
  descriptor_pool_ = VK_NULL_HANDLE;
  query_pool_ = VK_NULL_HANDLE;
  fence_ = VK_NULL_HANDLE;
  command_pool_ = VK_NULL_HANDLE;
  device_ = VK_NULL_HANDLE;
  instance_ = VK_NULL_HANDLE;
}

void VkContext::select_physical_device() {
  uint32_t count = 0;
  GS_VK_CHECK(vkEnumeratePhysicalDevices(instance_, &count, nullptr));
  if (count == 0) throw std::runtime_error("no Vulkan physical device found");
  std::vector<VkPhysicalDevice> devices(count);
  GS_VK_CHECK(vkEnumeratePhysicalDevices(instance_, &count, devices.data()));

  auto type_score = [](VkPhysicalDeviceType type) {
    switch (type) {
      case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 4;
      case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 3;
      case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 2;
      case VK_PHYSICAL_DEVICE_TYPE_CPU: return 1;
      default: return 0;
    }
  };
  int best_score = -1;
  for (VkPhysicalDevice device : devices) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(device, &props);
    if (props.apiVersion < VK_API_VERSION_1_1) continue;
    int score = type_score(props.deviceType);
    if (score > best_score) {
      best_score = score;
      physical_ = device;
    }
  }
  if (physical_ == VK_NULL_HANDLE) throw std::runtime_error("no Vulkan 1.1 capable device found");
}

void VkContext::query_device_info() {
  uint32_t ext_count = 0;
  GS_VK_CHECK(vkEnumerateDeviceExtensionProperties(physical_, nullptr, &ext_count, nullptr));
  std::vector<VkExtensionProperties> exts(ext_count);
  GS_VK_CHECK(vkEnumerateDeviceExtensionProperties(physical_, nullptr, &ext_count, exts.data()));
  for (const auto& e : exts) info_.extensions.emplace_back(e.extensionName);
  std::sort(info_.extensions.begin(), info_.extensions.end());

  VkPhysicalDeviceProperties base_props;
  vkGetPhysicalDeviceProperties(physical_, &base_props);
  info_.api_version = std::min(info_.instance_api_version, base_props.apiVersion);
  const bool v12 = info_.api_version >= VK_API_VERSION_1_2;
  const bool v13 = info_.api_version >= VK_API_VERSION_1_3;

  info_.has_driver_properties = v12 || has_extension(VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME);
  info_.has_subgroup_size_control = v13 || has_extension(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
  info_.has_float16_int8 = v12 || has_extension(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
  info_.has_atomic_float = has_extension(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
  info_.has_atomic_int64 = v12 || has_extension(VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME);

  // Properties.
  VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  info_.subgroup = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
  info_.maintenance3 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES};
  info_.driver = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
  info_.subgroup_size_control = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
  void** next = &props2.pNext;
  auto chain = [&next](auto& s) {
    *next = &s;
    next = &s.pNext;
  };
  chain(info_.subgroup);
  chain(info_.maintenance3);
  if (info_.has_driver_properties) chain(info_.driver);
  if (info_.has_subgroup_size_control) chain(info_.subgroup_size_control);
  vkGetPhysicalDeviceProperties2(physical_, &props2);
  info_.properties = props2.properties;

  // Features.
  VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  info_.storage16 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
  info_.float16_int8 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
  info_.atomic_float = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
  info_.atomic_int64 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES};
  info_.subgroup_size_control_features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
  next = &features2.pNext;
  chain(info_.storage16);
  if (info_.has_float16_int8) chain(info_.float16_int8);
  if (info_.has_atomic_float) chain(info_.atomic_float);
  if (info_.has_atomic_int64) chain(info_.atomic_int64);
  if (info_.has_subgroup_size_control) chain(info_.subgroup_size_control_features);
  vkGetPhysicalDeviceFeatures2(physical_, &features2);
  info_.features = features2.features;

  // The chained structs are copied by value later; drop the dangling links.
  info_.subgroup.pNext = nullptr;
  info_.maintenance3.pNext = nullptr;
  info_.driver.pNext = nullptr;
  info_.subgroup_size_control.pNext = nullptr;
  info_.storage16.pNext = nullptr;
  info_.float16_int8.pNext = nullptr;
  info_.atomic_float.pNext = nullptr;
  info_.atomic_int64.pNext = nullptr;
  info_.subgroup_size_control_features.pNext = nullptr;

  vkGetPhysicalDeviceMemoryProperties(physical_, &info_.memory);

  uint32_t family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_, &family_count, nullptr);
  info_.queue_families.resize(family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_, &family_count, info_.queue_families.data());

  // Prefer a universal (graphics + compute) family so later rendering can share it.
  int chosen = -1;
  for (uint32_t i = 0; i < family_count; ++i) {
    VkQueueFlags flags = info_.queue_families[i].queueFlags;
    if (!(flags & VK_QUEUE_COMPUTE_BIT)) continue;
    bool chosen_is_universal =
        chosen >= 0 && (info_.queue_families[chosen].queueFlags & VK_QUEUE_GRAPHICS_BIT);
    if (chosen < 0 || ((flags & VK_QUEUE_GRAPHICS_BIT) && !chosen_is_universal)) {
      chosen = static_cast<int>(i);
    }
  }
  if (chosen < 0) throw std::runtime_error("no compute-capable queue family");
  info_.queue_family = static_cast<uint32_t>(chosen);
}

void VkContext::create_device() {
  const bool v12 = info_.api_version >= VK_API_VERSION_1_2;
  std::vector<const char*> extensions;

  VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  features2.features.shaderInt64 = info_.features.shaderInt64;
  features2.features.shaderInt16 = info_.features.shaderInt16;
  void** next = &features2.pNext;

  VkPhysicalDeviceShaderFloat16Int8Features float16{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
  if (info_.has_float16_int8 && info_.float16_int8.shaderFloat16) {
    float16.shaderFloat16 = VK_TRUE;
    *next = &float16;
    next = &float16.pNext;
    if (!v12) extensions.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
    enabled_.float16 = true;
  }

  VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
  if (info_.has_atomic_float && info_.atomic_float.shaderBufferFloat32AtomicAdd) {
    atomic_float.shaderBufferFloat32AtomicAdd = VK_TRUE;
    atomic_float.shaderBufferFloat32Atomics = info_.atomic_float.shaderBufferFloat32Atomics;
    *next = &atomic_float;
    next = &atomic_float.pNext;
    extensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    enabled_.atomic_float_add = true;
  }

  if (enabled_.presentation && has_extension(VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
    extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  } else {
    enabled_.presentation = false;
  }

  const float priority = 1.0f;
  VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  queue_info.queueFamilyIndex = info_.queue_family;
  queue_info.queueCount = 1;
  queue_info.pQueuePriorities = &priority;

  VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  device_info.pNext = &features2;
  device_info.queueCreateInfoCount = 1;
  device_info.pQueueCreateInfos = &queue_info;
  device_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  device_info.ppEnabledExtensionNames = extensions.data();
  GS_VK_CHECK(vkCreateDevice(physical_, &device_info, nullptr, &device_));
  vkGetDeviceQueue(device_, info_.queue_family, 0, &queue_);

  enabled_.timestamps = info_.queue_families[info_.queue_family].timestampValidBits > 0 &&
                        info_.properties.limits.timestampPeriod > 0.0f;
}

uint32_t VkContext::find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags required,
                                     VkMemoryPropertyFlags preferred) const {
  for (VkMemoryPropertyFlags wanted : {required | preferred, required}) {
    for (uint32_t i = 0; i < info_.memory.memoryTypeCount; ++i) {
      if ((type_bits & (1u << i)) &&
          (info_.memory.memoryTypes[i].propertyFlags & wanted) == wanted) {
        return i;
      }
    }
  }
  throw std::runtime_error("no suitable Vulkan memory type");
}

Buffer VkContext::create_buffer(VkDeviceSize size, MemoryKind kind, VkBufferUsageFlags extra_usage) {
  size = std::max<VkDeviceSize>(size, 4);
  VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  buffer_info.size = size;
  buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT | extra_usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VkBuffer buffer = VK_NULL_HANDLE;
  GS_VK_CHECK(vkCreateBuffer(device_, &buffer_info, nullptr, &buffer));

  VkMemoryPropertyFlags required = 0;
  VkMemoryPropertyFlags preferred = 0;
  switch (kind) {
    case MemoryKind::Device:
      preferred = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
      break;
    case MemoryKind::Upload:
      required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
      break;
    case MemoryKind::Readback:
      required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
      preferred = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
      break;
  }

  VkDeviceMemory memory = VK_NULL_HANDLE;
  void* mapped = nullptr;
  try {
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device_, buffer, &req);
    VkMemoryAllocateInfo alloc_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc_info.allocationSize = req.size;
    alloc_info.memoryTypeIndex = find_memory_type(req.memoryTypeBits, required, preferred);
    GS_VK_CHECK(vkAllocateMemory(device_, &alloc_info, nullptr, &memory));
    GS_VK_CHECK(vkBindBufferMemory(device_, buffer, memory, 0));
    if (kind != MemoryKind::Device) {
      GS_VK_CHECK(vkMapMemory(device_, memory, 0, VK_WHOLE_SIZE, 0, &mapped));
    }
  } catch (...) {
    if (memory != VK_NULL_HANDLE) vkFreeMemory(device_, memory, nullptr);
    vkDestroyBuffer(device_, buffer, nullptr);
    throw;
  }
  return Buffer(device_, buffer, memory, size, mapped);
}

void VkContext::upload(const Buffer& dst, const void* data, VkDeviceSize size) {
  if (size == 0) return;
  Buffer staging = create_buffer(size, MemoryKind::Upload);
  std::memcpy(staging.mapped(), data, size);
  run([&](VkCommandBuffer cmd) {
    VkBufferCopy region{0, 0, size};
    vkCmdCopyBuffer(cmd, staging.handle(), dst.handle(), 1, &region);
  });
}

void VkContext::download(const Buffer& src, void* data, VkDeviceSize size) {
  if (size == 0) return;
  Buffer staging = create_buffer(size, MemoryKind::Readback);
  run([&](VkCommandBuffer cmd) {
    VkBufferCopy region{0, 0, size};
    vkCmdCopyBuffer(cmd, src.handle(), staging.handle(), 1, &region);
    barrier_to_host(cmd);
  });
  std::memcpy(data, staging.mapped(), size);
}

VkShaderModule VkContext::create_shader_module(std::span<const uint32_t> spirv) const {
  VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  module_info.codeSize = spirv.size_bytes();
  module_info.pCode = spirv.data();
  VkShaderModule module = VK_NULL_HANDLE;
  GS_VK_CHECK(vkCreateShaderModule(device_, &module_info, nullptr, &module));
  return module;
}

VkDescriptorPool VkContext::create_descriptor_pool(uint32_t max_sets, uint32_t max_storage_buffers) const {
  VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, max_storage_buffers};
  VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  info.maxSets = max_sets;
  info.poolSizeCount = 1;
  info.pPoolSizes = &pool_size;
  VkDescriptorPool pool = VK_NULL_HANDLE;
  GS_VK_CHECK(vkCreateDescriptorPool(device_, &info, nullptr, &pool));
  return pool;
}

VkDescriptorSet VkContext::allocate_set(VkDescriptorPool pool, VkDescriptorSetLayout layout,
                                        std::initializer_list<const Buffer*> buffers) const {
  VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  alloc.descriptorPool = pool;
  alloc.descriptorSetCount = 1;
  alloc.pSetLayouts = &layout;
  VkDescriptorSet set = VK_NULL_HANDLE;
  GS_VK_CHECK(vkAllocateDescriptorSets(device_, &alloc, &set));

  std::vector<VkDescriptorBufferInfo> infos;
  infos.reserve(buffers.size());
  std::vector<VkWriteDescriptorSet> writes;
  writes.reserve(buffers.size());
  uint32_t binding = 0;
  for (const Buffer* b : buffers) {
    infos.push_back({b->handle(), 0, VK_WHOLE_SIZE});
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = set;
    w.dstBinding = binding++;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.pBufferInfo = &infos.back();
    writes.push_back(w);
  }
  vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  return set;
}

VkDescriptorSet VkContext::allocate_transient_set(const ComputePipeline& pipeline,
                                                  std::initializer_list<const Buffer*> buffers) const {
  if (buffers.size() != pipeline.binding_count_) {
    throw std::logic_error("descriptor set: buffer count does not match pipeline bindings");
  }
  return allocate_set(descriptor_pool_, pipeline.set_layout_, buffers);
}

VkResult VkContext::submit(const VkSubmitInfo& submit_info, VkFence fence) {
  std::lock_guard<std::mutex> lock(queue_mutex_);
  return vkQueueSubmit(queue_, 1, &submit_info, fence);
}

VkResult VkContext::present(const VkPresentInfoKHR& present_info) {
  std::lock_guard<std::mutex> lock(queue_mutex_);
  return vkQueuePresentKHR(queue_, &present_info);
}

void VkContext::queue_wait_idle() {
  std::lock_guard<std::mutex> lock(queue_mutex_);
  vkQueueWaitIdle(queue_);
}

ComputePipeline VkContext::create_compute_pipeline(std::span<const uint32_t> spirv,
                                                   uint32_t binding_count,
                                                   uint32_t push_constant_size,
                                                   std::span<const uint32_t> specialization) {
  ComputePipeline p;
  p.device_ = device_;
  p.binding_count_ = binding_count;
  p.push_constant_size_ = push_constant_size;

  std::vector<VkDescriptorSetLayoutBinding> bindings(binding_count);
  for (uint32_t i = 0; i < binding_count; ++i) {
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  set_info.bindingCount = binding_count;
  set_info.pBindings = bindings.data();
  GS_VK_CHECK(vkCreateDescriptorSetLayout(device_, &set_info, nullptr, &p.set_layout_));

  VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_constant_size};
  VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layout_info.setLayoutCount = 1;
  layout_info.pSetLayouts = &p.set_layout_;
  layout_info.pushConstantRangeCount = push_constant_size > 0 ? 1 : 0;
  layout_info.pPushConstantRanges = &range;
  GS_VK_CHECK(vkCreatePipelineLayout(device_, &layout_info, nullptr, &p.layout_));

  VkShaderModule module = create_shader_module(spirv);

  VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipeline_info.stage.module = module;
  pipeline_info.stage.pName = "main";
  std::vector<VkSpecializationMapEntry> entries(specialization.size());
  for (uint32_t i = 0; i < entries.size(); ++i) entries[i] = {i, i * 4, 4};
  VkSpecializationInfo spec_info{static_cast<uint32_t>(entries.size()), entries.data(),
                                 specialization.size_bytes(), specialization.data()};
  if (!specialization.empty()) pipeline_info.stage.pSpecializationInfo = &spec_info;
  pipeline_info.layout = p.layout_;
  VkResult result =
      vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &p.pipeline_);
  vkDestroyShaderModule(device_, module, nullptr);
  if (result != VK_SUCCESS) throw VulkanError("vkCreateComputePipelines", result);
  return p;
}

RunTiming VkContext::run(const std::function<void(VkCommandBuffer)>& record) {
  std::lock_guard<std::recursive_mutex> run_lock(run_mutex_);
  if (in_run_) throw std::logic_error("VkContext::run is not reentrant");
  in_run_ = true;
  struct Reset {
    bool& flag;
    ~Reset() { flag = false; }
  } reset{in_run_};

  GS_VK_CHECK(vkResetDescriptorPool(device_, descriptor_pool_, 0));
  GS_VK_CHECK(vkResetCommandBuffer(cmd_, 0));
  VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  GS_VK_CHECK(vkBeginCommandBuffer(cmd_, &begin));
  if (enabled_.timestamps) {
    vkCmdResetQueryPool(cmd_, query_pool_, 0, 2);
    vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool_, 0);
  }
  record(cmd_);
  if (enabled_.timestamps) {
    vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_, 1);
  }
  GS_VK_CHECK(vkEndCommandBuffer(cmd_));

  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cmd_;
  auto start = std::chrono::steady_clock::now();
  GS_VK_CHECK(this->submit(submit, fence_));
  VkResult wait = vkWaitForFences(device_, 1, &fence_, VK_TRUE, 60ull * 1000 * 1000 * 1000);
  auto end = std::chrono::steady_clock::now();
  if (wait != VK_SUCCESS) throw VulkanError("vkWaitForFences", wait);
  GS_VK_CHECK(vkResetFences(device_, 1, &fence_));

  RunTiming timing;
  timing.cpu_ms = std::chrono::duration<double, std::milli>(end - start).count();
  if (enabled_.timestamps) {
    uint64_t stamps[2] = {0, 0};
    VkResult r = vkGetQueryPoolResults(device_, query_pool_, 0, 2, sizeof(stamps), stamps,
                                       sizeof(uint64_t),
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (r == VK_SUCCESS) {
      uint32_t bits = info_.queue_families[info_.queue_family].timestampValidBits;
      uint64_t mask = bits >= 64 ? ~0ull : ((1ull << bits) - 1);
      uint64_t ticks = (stamps[1] - stamps[0]) & mask;
      timing.gpu_ms = static_cast<double>(ticks) * info_.properties.limits.timestampPeriod / 1e6;
    }
  }
  return timing;
}

void VkContext::dispatch(VkCommandBuffer cmd, const ComputePipeline& pipeline, VkDescriptorSet set,
                         const void* push, uint32_t push_size, uint32_t groups_x, uint32_t groups_y,
                         uint32_t groups_z) const {
  if (push_size != pipeline.push_constant_size_) {
    throw std::logic_error("dispatch: push constant size mismatch");
  }
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline_);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout_, 0, 1, &set, 0,
                          nullptr);
  if (push_size > 0) {
    vkCmdPushConstants(cmd, pipeline.layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size, push);
  }
  vkCmdDispatch(cmd, groups_x, groups_y, groups_z);
}

void VkContext::dispatch_indirect(VkCommandBuffer cmd, const ComputePipeline& pipeline, VkDescriptorSet set,
                                  const void* push, uint32_t push_size, const Buffer& args,
                                  VkDeviceSize offset) const {
  if (push_size != pipeline.push_constant_size_) {
    throw std::logic_error("dispatch_indirect: push constant size mismatch");
  }
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline_);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout_, 0, 1, &set, 0, nullptr);
  if (push_size > 0) {
    vkCmdPushConstants(cmd, pipeline.layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size, push);
  }
  vkCmdDispatchIndirect(cmd, args.handle(), offset);
}

void VkContext::dispatch(VkCommandBuffer cmd, const ComputePipeline& pipeline,
                         std::initializer_list<const Buffer*> buffers, const void* push,
                         uint32_t push_size, uint32_t groups_x, uint32_t groups_y,
                         uint32_t groups_z) {
  VkDescriptorSet set = allocate_transient_set(pipeline, buffers);
  dispatch(cmd, pipeline, set, push, push_size, groups_x, groups_y, groups_z);
}

void VkContext::barrier_to_host(VkCommandBuffer cmd) {
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

void VkContext::barrier(VkCommandBuffer cmd) {
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                     VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
  const VkPipelineStageFlags src = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
  const VkPipelineStageFlags dst = src | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
  vkCmdPipelineBarrier(cmd, src, dst, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

}  // namespace gs
