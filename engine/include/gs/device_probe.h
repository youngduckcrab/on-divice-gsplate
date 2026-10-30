#pragma once

#include <string>

#include "gs/vk_context.h"

namespace gs {

// Serializes the Vulkan capabilities relevant to 3DGS training (subgroups,
// float atomics, fp16, limits, memory heaps, extensions) as a JSON object.
std::string probe_json(const VkContext& ctx);

}  // namespace gs
