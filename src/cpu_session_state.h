#ifndef NCNN_MOE_CPU_SESSION_STATE_H
#define NCNN_MOE_CPU_SESSION_STATE_H

#include <cstdint>
#include <vector>

namespace ncnn {
namespace moe {

struct CpuLayerCache
{
    std::vector<float> keys;
    std::vector<float> values;
    uint64_t start_position = 0;
    uint64_t token_count = 0;
};

class CpuSessionState
{
public:
    std::vector<CpuLayerCache> layers;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_SESSION_STATE_H
