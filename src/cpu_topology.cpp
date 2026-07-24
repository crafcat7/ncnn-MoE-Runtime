#include "cpu_topology.h"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <iterator>
#include <numeric>
#include <string>

#if defined(__linux__)
#include <sched.h>
#endif

namespace ncnn {
namespace moe {

static bool parse_cpu_index(
    std::string_view value,
    uint32_t& cpu) noexcept
{
    if (value.empty())
        return false;
    const char* begin = value.data();
    const char* end = begin + value.size();
    const auto parsed = std::from_chars(begin, end, cpu);
    return parsed.ec == std::errc() && parsed.ptr == end;
}

std::vector<uint32_t> parse_linux_cpu_list(std::string_view value)
{
    std::vector<uint32_t> cpus;
    if (value.empty() || value.back() == ',')
        return cpus;
    size_t cursor = 0;
    while (cursor < value.size()) {
        const size_t comma = value.find(',', cursor);
        const size_t end = comma == std::string_view::npos ? value.size() : comma;
        const std::string_view item = value.substr(cursor, end - cursor);
        const size_t dash = item.find('-');
        uint32_t first = 0;
        uint32_t last = 0;
        const bool valid = dash == std::string_view::npos
                               ? parse_cpu_index(item, first)
                               : parse_cpu_index(item.substr(0, dash), first)
                                     && parse_cpu_index(item.substr(dash + 1), last);
        if (!valid)
            return {};
        if (dash == std::string_view::npos)
            last = first;
        if (last < first || last - first > 65535)
            return {};
        for (uint64_t cpu = first; cpu <= last; ++cpu)
            cpus.push_back(static_cast<uint32_t>(cpu));
        cursor = end + (comma == std::string_view::npos ? 0 : 1);
    }
    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    return cpus;
}

#if defined(__linux__)
static std::string read_first_line(const std::string& path)
{
    std::ifstream stream(path);
    std::string line;
    if (stream)
        std::getline(stream, line);
    return line;
}
#endif

CpuTopology discover_cpu_topology()
{
    CpuTopology topology;
#if defined(__linux__)
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0) {
        for (uint32_t cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (CPU_ISSET(cpu, &allowed))
                topology.allowed_cpus.push_back(cpu);
        }
    }
    if (topology.allowed_cpus.empty())
        return topology;

    const std::vector<uint32_t> node_ids = parse_linux_cpu_list(
        read_first_line("/sys/devices/system/node/online"));
    for (uint32_t node_id : node_ids) {
        const std::vector<uint32_t> node_cpus = parse_linux_cpu_list(
            read_first_line(
                "/sys/devices/system/node/node"
                + std::to_string(node_id)
                + "/cpulist"));
        std::vector<uint32_t> allowed_node_cpus;
        std::set_intersection(
            node_cpus.begin(),
            node_cpus.end(),
            topology.allowed_cpus.begin(),
            topology.allowed_cpus.end(),
            std::back_inserter(allowed_node_cpus));
        if (!allowed_node_cpus.empty())
            topology.numa_nodes.push_back(std::move(allowed_node_cpus));
    }
#endif
    return topology;
}

static std::vector<std::vector<uint32_t> > partition_flat_cpus(
    const std::vector<uint32_t>& cpus,
    uint32_t worker_count)
{
    std::vector<std::vector<uint32_t> > partitions(worker_count);
    if (cpus.empty() || worker_count == 0)
        return {};
    if (worker_count > cpus.size()) {
        for (uint32_t worker = 0; worker < worker_count; ++worker)
            partitions[worker].push_back(cpus[worker % cpus.size()]);
        return partitions;
    }
    const uint32_t base = static_cast<uint32_t>(cpus.size()) / worker_count;
    const uint32_t remainder = static_cast<uint32_t>(cpus.size()) % worker_count;
    size_t offset = 0;
    for (uint32_t worker = 0; worker < worker_count; ++worker) {
        const size_t count = base + (worker < remainder ? 1u : 0u);
        partitions[worker].insert(
            partitions[worker].end(),
            cpus.begin() + offset,
            cpus.begin() + offset + count);
        offset += count;
    }
    return partitions;
}

std::vector<std::vector<uint32_t> > partition_cpu_topology(
    const CpuTopology& topology,
    uint32_t worker_count)
{
    if (worker_count == 0 || topology.allowed_cpus.empty())
        return {};
    if (topology.numa_nodes.empty())
        return partition_flat_cpus(topology.allowed_cpus, worker_count);

    std::vector<size_t> node_order(topology.numa_nodes.size());
    std::iota(node_order.begin(), node_order.end(), 0);
    std::stable_sort(
        node_order.begin(),
        node_order.end(),
        [&topology](size_t left, size_t right) {
            return topology.numa_nodes[left].size()
                   > topology.numa_nodes[right].size();
        });
    if (worker_count < node_order.size())
        node_order.resize(worker_count);

    std::vector<uint32_t> workers_per_node(node_order.size(), 1);
    uint32_t assigned_workers = static_cast<uint32_t>(node_order.size());
    while (assigned_workers < worker_count) {
        size_t selected = 0;
        for (size_t node = 1; node < node_order.size(); ++node) {
            const size_t selected_cpu_count
                = topology.numa_nodes[node_order[selected]].size();
            const size_t candidate_cpu_count
                = topology.numa_nodes[node_order[node]].size();
            if (candidate_cpu_count * workers_per_node[selected]
                > selected_cpu_count * workers_per_node[node])
                selected = node;
        }
        ++workers_per_node[selected];
        ++assigned_workers;
    }

    std::vector<std::vector<uint32_t> > partitions;
    partitions.reserve(worker_count);
    for (size_t node = 0; node < node_order.size(); ++node) {
        std::vector<std::vector<uint32_t> > node_partitions
            = partition_flat_cpus(
                topology.numa_nodes[node_order[node]],
                workers_per_node[node]);
        for (std::vector<uint32_t>& partition : node_partitions)
            partitions.push_back(std::move(partition));
    }
    return partitions;
}

} // namespace moe
} // namespace ncnn
