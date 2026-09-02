#ifndef NCNN_MOE_CPU_TOPOLOGY_H
#define NCNN_MOE_CPU_TOPOLOGY_H

#include <cstdint>
#include <string_view>
#include <vector>

namespace ncnn {
namespace moe {

struct CpuTopology
{
    std::vector<uint32_t> allowed_cpus;
    std::vector<std::vector<uint32_t>> numa_nodes;
    uint32_t physical_cpu_count = 0;
};

[[nodiscard]] std::vector<uint32_t> parse_linux_cpu_list(std::string_view value);
[[nodiscard]] CpuTopology discover_cpu_topology();
[[nodiscard]] std::vector<std::vector<uint32_t>> partition_cpu_topology(const CpuTopology& topology, uint32_t worker_count);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_TOPOLOGY_H
