#include "gs/radix_sort.h"

#include "shaders/radix_count.h"
#include "shaders/radix_scan.h"
#include "shaders/radix_scatter.h"

namespace gs {
namespace {

struct PassParams {
  uint32_t n;
  uint32_t shift;
  uint32_t num_blocks;
};

struct ScanParams {
  uint32_t count;
};

constexpr uint32_t kRadixBits = 4;
constexpr uint32_t kBuckets = 1u << kRadixBits;

}  // namespace

RadixSort::RadixSort(VkContext& ctx)
    : ctx_(ctx),
      count_(ctx.create_compute_pipeline(shaders::radix_count, 2, sizeof(PassParams))),
      scan_(ctx.create_compute_pipeline(shaders::radix_scan, 1, sizeof(ScanParams))),
      scatter_(ctx.create_compute_pipeline(shaders::radix_scatter, 5, sizeof(PassParams))) {}

void RadixSort::reserve(uint32_t n) {
  if (n <= capacity_) return;
  uint32_t num_blocks = div_up(n, kBlockSize);
  tmp_keys_ = ctx_.create_buffer(uint64_t(n) * 4, MemoryKind::Device);
  tmp_values_ = ctx_.create_buffer(uint64_t(n) * 4, MemoryKind::Device);
  histogram_ = ctx_.create_buffer(uint64_t(num_blocks) * kBuckets * 4, MemoryKind::Device);
  capacity_ = n;
}

void RadixSort::record(VkCommandBuffer cmd, const Buffer& keys, const Buffer& values, uint32_t n,
                       uint32_t key_bits) {
  if (n <= 1) return;
  reserve(n);
  const uint32_t num_blocks = div_up(n, kBlockSize);
  const uint32_t passes = div_up(key_bits == 0 ? 32 : key_bits, kRadixBits);

  const Buffer* src_keys = &keys;
  const Buffer* src_values = &values;
  const Buffer* dst_keys = &tmp_keys_;
  const Buffer* dst_values = &tmp_values_;
  for (uint32_t pass = 0; pass < passes; ++pass) {
    PassParams params{n, pass * kRadixBits, num_blocks};
    ctx_.dispatch(cmd, count_, {src_keys, &histogram_}, params, num_blocks);
    VkContext::barrier(cmd);
    ScanParams scan{num_blocks * kBuckets};
    ctx_.dispatch(cmd, scan_, {&histogram_}, scan, 1);
    VkContext::barrier(cmd);
    ctx_.dispatch(cmd, scatter_, {src_keys, src_values, dst_keys, dst_values, &histogram_}, params,
                  num_blocks);
    VkContext::barrier(cmd);
    std::swap(src_keys, dst_keys);
    std::swap(src_values, dst_values);
  }

  if (src_keys != &keys) {
    // Odd number of passes: the result sits in the scratch buffers.
    VkBufferCopy region{0, 0, uint64_t(n) * 4};
    vkCmdCopyBuffer(cmd, src_keys->handle(), keys.handle(), 1, &region);
    vkCmdCopyBuffer(cmd, src_values->handle(), values.handle(), 1, &region);
    VkContext::barrier(cmd);
  }
}

}  // namespace gs
