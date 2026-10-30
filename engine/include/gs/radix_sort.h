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
  explicit RadixSort(VkContext& ctx);

  // Records the sort into `cmd`. The result ends up in `keys`/`values`.
  // Only the low `key_bits` bits of each key are considered (rounded up to a
  // multiple of 4). Scratch buffers grow to fit `n` on demand, so call this
  // outside of an active run() the first time a larger `n` is used, or call
  // reserve() beforehand.
  void record(VkCommandBuffer cmd, const Buffer& keys, const Buffer& values, uint32_t n,
              uint32_t key_bits = 32);

  void reserve(uint32_t n);

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
  uint32_t capacity_ = 0;
};

}  // namespace gs
