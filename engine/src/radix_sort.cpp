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

RadixSort::Binding RadixSort::bind(VkDescriptorPool pool, const Buffer& keys, const Buffer& values) {
  Binding b;
  b.count[0] = ctx_.allocate_set(pool, count_.set_layout(), {&keys, &histogram_});
  b.count[1] = ctx_.allocate_set(pool, count_.set_layout(), {&tmp_keys_, &histogram_});
  b.scan = ctx_.allocate_set(pool, scan_.set_layout(), {&histogram_});
  b.scatter[0] = ctx_.allocate_set(pool, scatter_.set_layout(),
                                   {&keys, &values, &tmp_keys_, &tmp_values_, &histogram_});
  b.scatter[1] = ctx_.allocate_set(pool, scatter_.set_layout(),
                                   {&tmp_keys_, &tmp_values_, &keys, &values, &histogram_});
  return b;
}

void RadixSort::record(VkCommandBuffer cmd, const Binding& binding, const Buffer& keys,
                       const Buffer& values, uint32_t n, uint32_t key_bits) {
  if (n <= 1) return;
  if (n > capacity_) throw std::logic_error("RadixSort: reserve() before recording a larger sort");
  const uint32_t num_blocks = div_up(n, kBlockSize);
  const uint32_t passes = div_up(key_bits == 0 ? 32 : key_bits, kRadixBits);

  for (uint32_t pass = 0; pass < passes; ++pass) {
    const uint32_t dir = pass & 1;  // 0: keys -> scratch, 1: scratch -> keys
    PassParams params{n, pass * kRadixBits, num_blocks};
    ctx_.dispatch(cmd, count_, binding.count[dir], params, num_blocks);
    VkContext::barrier(cmd);
    ScanParams scan{num_blocks * kBuckets};
    ctx_.dispatch(cmd, scan_, binding.scan, scan, 1);
    VkContext::barrier(cmd);
    ctx_.dispatch(cmd, scatter_, binding.scatter[dir], params, num_blocks);
    VkContext::barrier(cmd);
  }

  if (passes & 1) {
    // Odd number of passes: the result sits in the scratch buffers.
    VkBufferCopy region{0, 0, uint64_t(n) * 4};
    vkCmdCopyBuffer(cmd, tmp_keys_.handle(), keys.handle(), 1, &region);
    vkCmdCopyBuffer(cmd, tmp_values_.handle(), values.handle(), 1, &region);
    VkContext::barrier(cmd);
  }
}

void RadixSort::record(VkCommandBuffer cmd, const Buffer& keys, const Buffer& values, uint32_t n,
                       uint32_t key_bits) {
  if (n <= 1) return;
  reserve(n);
  Binding b;
  b.count[0] = ctx_.allocate_transient_set(count_, {&keys, &histogram_});
  b.count[1] = ctx_.allocate_transient_set(count_, {&tmp_keys_, &histogram_});
  b.scan = ctx_.allocate_transient_set(scan_, {&histogram_});
  b.scatter[0] = ctx_.allocate_transient_set(scatter_, {&keys, &values, &tmp_keys_, &tmp_values_, &histogram_});
  b.scatter[1] = ctx_.allocate_transient_set(scatter_, {&tmp_keys_, &tmp_values_, &keys, &values, &histogram_});
  record(cmd, b, keys, values, n, key_bits);
}

}  // namespace gs
