#pragma once

#include <cstdint>

#include "gs/vk_context.h"

namespace gs {

// Stable LSD radix sort of (uint32 key, uint32 value) pairs, 4 bits per pass.
// Uses only workgroup barriers and shared memory (no subgroup operations and no
// cross-workgroup forward-progress assumptions), so it behaves the same on
// every GPU regardless of subgroup size.
class RadixSort {
 public:
  // Descriptor sets for sorting one particular (keys, values) pair.
  struct Binding {
    VkDescriptorSet count[2] = {};    // [0] reads keys, [1] reads scratch
    VkDescriptorSet scan = VK_NULL_HANDLE;
    VkDescriptorSet scatter[2] = {};  // [0] keys -> scratch, [1] scratch -> keys
  };

  explicit RadixSort(VkContext& ctx);

  // Grows the scratch buffers to fit `n` pairs. Invalidates existing bindings
  // when it reallocates.
  void reserve(uint32_t n);

  // Persistent descriptor sets from `pool` (5 sets, 19 storage buffers), for
  // sorting the same buffers every frame. Call reserve() first. With `count`,
  // only the first min(n, count[0]) items are sorted, n as passed to record():
  // the count can then be written by an earlier pass on the GPU.
  Binding bind(VkDescriptorPool pool, const Buffer& keys, const Buffer& values, const Buffer* count = nullptr);

  // Records the sort into `cmd`; the result ends up in `keys`/`values`. Only
  // the low `key_bits` bits of each key are considered (rounded up to a
  // multiple of 4).
  void record(VkCommandBuffer cmd, const Binding& binding, const Buffer& keys,
              const Buffer& values, uint32_t n, uint32_t key_bits = 32);

  // One-off variant using transient descriptor sets; only valid inside VkContext::run().
  void record(VkCommandBuffer cmd, const Buffer& keys, const Buffer& values, uint32_t n,
              uint32_t key_bits = 32);

  static constexpr uint32_t kWorkgroupSize = 128;
  static constexpr uint32_t kItemsPerThread = 8;
  static constexpr uint32_t kBlockSize = kWorkgroupSize * kItemsPerThread;

 private:
  VkContext& ctx_;
  ComputePipeline count_;
  ComputePipeline scan_;
  ComputePipeline scatter_;
  Buffer tmp_keys_;
  Buffer tmp_values_;
  Buffer histogram_;
  Buffer unlimited_;  // count[0] = UINT32_MAX: sort all n
  uint32_t capacity_ = 0;
};

}  // namespace gs
