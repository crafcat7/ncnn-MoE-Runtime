#include "kernels/mxfp4.h"
#include "kernels/ops.h"
#include "storage/expertcache.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

static void gguf_cache_check(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

static float gguf_value(const uint8_t* block, uint32_t column)
{
    static constexpr float values[] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                       -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    const uint8_t code = static_cast<uint8_t>((block[1 + column % 16] >> (column < 16 ? 0 : 4)) & 15u);
    return values[code] * std::ldexp(1.f, static_cast<int>(block[0]) - 127);
}

static void gguf_unpack_nibble_order()
{
    std::array<uint8_t, 34> source{};
    source[0] = 125;
    source[17] = 128;
    for (uint32_t block = 0; block < 2; ++block)
        for (uint32_t column = 0; column < 16; ++column)
            source[block * 17 + 1 + column] = static_cast<uint8_t>((column + block) % 16 | ((15 - column + block) % 16) << 4);
    std::array<uint8_t, 32> blocks{};
    std::array<uint8_t, 2> scales{};
    gguf_cache_check(unpack_gguf_mxfp4(source, blocks, scales), "GGUF MXFP4 unpack");
    for (uint32_t column = 0; column < 64; ++column)
    {
        std::array<float, 64> input{};
        input[column] = 1.f;
        const float actual = mxfp4_dot(blocks.data(), scales.data(), 2, input.data());
        const float expected = gguf_value(source.data() + column / 32 * 17, column % 32);
        gguf_cache_check(actual == expected, "GGUF low/high half-block nibbles retain logical input columns");
    }
    gguf_cache_check(!unpack_gguf_mxfp4(std::span<const uint8_t>(source).first(33), blocks, scales),
                     "GGUF unpack rejects truncated blocks");
    gguf_cache_check(!unpack_gguf_mxfp4(source, std::span<uint8_t>(blocks).first(31), scales),
                     "GGUF unpack rejects incorrect output sizes");
}

struct GgufExpertFixture
{
    static constexpr uint32_t rows = 32;
    static constexpr uint32_t experts = 3;
    static constexpr uint64_t matrix_size = rows * 17;
    static constexpr uint64_t pair_size = matrix_size * 3;
    std::filesystem::path directory;
    std::filesystem::path path;
    std::vector<uint8_t> file;
    std::array<std::array<uint64_t, experts>, 3> offsets{};

    GgufExpertFixture()
    {
        directory = std::filesystem::temp_directory_path()
                    / ("ncnn_moe_gguf_cache_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(directory);
        path = directory / "banks.gguf";
        file.resize(127, 0x91);
        for (uint32_t projection = 0; projection < 3; ++projection)
        {
            file.resize(file.size() + 19, 0x37);
            for (uint32_t expert = 0; expert < experts; ++expert)
            {
                offsets[projection][expert] = file.size();
                for (uint32_t row = 0; row < rows; ++row)
                {
                    file.push_back(static_cast<uint8_t>(122 + (row + projection + expert) % 4));
                    for (uint32_t column = 0; column < 16; ++column)
                    {
                        const uint32_t seed = expert * 3 + projection * 7 + row * 11;
                        file.push_back(static_cast<uint8_t>((seed + column) % 16 | ((seed * 3 + column * 7) % 16) << 4));
                    }
                }
            }
        }
        std::ofstream stream(path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(file.data()), static_cast<std::streamsize>(file.size()));
        gguf_cache_check(static_cast<bool>(stream), "write GGUF raw expert fixture");
    }

    ~GgufExpertFixture()
    {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }

    TensorData tensor(uint32_t expert, bool gate_up) const
    {
        TensorData result;
        result.dtype = DType::MxFp4;
        result.shape = {gate_up ? rows * 2 : rows, rows};
        auto storage = std::make_shared<MxFp4FileStorage>();
        storage->layout = MxFp4FileLayout::GgufBlockInterleaved;
        storage->blocks_path = path.string();
        storage->blocks_offset = offsets[gate_up ? 0 : 2][expert];
        storage->blocks_size = matrix_size;
        if (gate_up)
        {
            storage->secondary_blocks_path = path.string();
            storage->secondary_blocks_offset = offsets[1][expert];
            storage->secondary_blocks_size = matrix_size;
            storage->interleave_rows = true;
        }
        result.mxfp4_file_storage = std::move(storage);
        return result;
    }

    void check_output(const ExpertCacheLease& lease, uint32_t expert) const
    {
        gguf_cache_check(lease.gate_up->mxfp4_blocks.size() == rows * 32
                             && lease.gate_up->mxfp4_scales.size() == rows * 2
                             && lease.down->mxfp4_blocks.size() == rows * 16
                             && lease.down->mxfp4_scales.size() == rows,
                         "cache holds exactly one normalized expert payload");
        ActivationBuffer input(1, rows);
        for (uint32_t column = 0; column < rows; ++column)
            input.row(0)[column] = static_cast<float>(static_cast<int>(column % 13) - 6) * .03125f;
        ActivationBuffer activated;
        forward_gate_up_mxfp4(*lease.gate_up, nullptr, input, ExpertActivation::Silu, 0.f, activated, 0);
        const ActivationBuffer actual = forward_linear(*lease.down, activated, 0);
        std::array<float, rows> expected_activated{};
        for (uint32_t row = 0; row < rows; ++row)
        {
            float gate = 0.f;
            float up = 0.f;
            for (uint32_t column = 0; column < rows; ++column)
            {
                gate += gguf_value(file.data() + offsets[0][expert] + row * 17, column) * input.row(0)[column];
                up += gguf_value(file.data() + offsets[1][expert] + row * 17, column) * input.row(0)[column];
            }
            expected_activated[row] = gate / (1.f + std::exp(-gate)) * up;
        }
        float maximum = 0.f;
        for (uint32_t row = 0; row < rows; ++row)
        {
            float expected = 0.f;
            for (uint32_t column = 0; column < rows; ++column)
                expected += gguf_value(file.data() + offsets[2][expert] + row * 17, column) * expected_activated[column];
            maximum = std::max(maximum, std::abs(expected));
            gguf_cache_check(std::abs(actual.row(0)[row] - expected) < 0.000001f,
                             "GGUF expert cache matches independent dequantized SiLU oracle");
        }
        gguf_cache_check(maximum > 0.00001f, "GGUF expert CPU oracle is nonzero");
    }
};

static void gguf_expert_cache_ranges()
{
    GgufExpertFixture fixture;
    const TensorData gate_zero = fixture.tensor(0, true);
    const TensorData down_zero = fixture.tensor(0, false);
    const TensorData gate_one = fixture.tensor(1, true);
    const TensorData down_one = fixture.tensor(1, false);
    for (ExpertIoMode mode : {ExpertIoMode::Buffered, ExpertIoMode::Mmap, ExpertIoMode::Direct, ExpertIoMode::Auto})
    {
        ExpertCache cache(fixture.pair_size, 1, {}, mode, ExpertCacheCrossExpertReadCoalescing);
        gguf_cache_check(gate_zero.mxfp4_blocks.empty() && gate_zero.mxfp4_scales.empty(),
                         "deferred GGUF source does not retain the expert bank");
        {
            auto first = cache.acquire_pair(gate_zero, down_zero);
            gguf_cache_check(static_cast<bool>(first), "demand reads GGUF ranges for the first expert");
            fixture.check_output(first.value(), 0);
            auto hit = cache.acquire_pair(gate_zero, down_zero);
            gguf_cache_check(static_cast<bool>(hit) && cache.statistics().bytes_read == fixture.pair_size,
                             "GGUF resident hit performs no second file read");
        }
        {
            auto second = cache.acquire_pair(gate_one, down_one);
            gguf_cache_check(static_cast<bool>(second), "GGUF ARC admits the next expert");
            fixture.check_output(second.value(), 1);
        }
        {
            auto reloaded = cache.acquire_pair(gate_zero, down_zero);
            gguf_cache_check(static_cast<bool>(reloaded), "GGUF cache reloads an evicted expert");
            fixture.check_output(reloaded.value(), 0);
        }
        const ExpertCacheStatistics statistics = cache.statistics();
        gguf_cache_check(statistics.bytes_read == fixture.pair_size * 3
                             && statistics.resident_size == fixture.pair_size
                             && statistics.evictions == 2,
                         "GGUF I/O and residency charge only selected experts");
        if (mode == ExpertIoMode::Mmap)
            gguf_cache_check(statistics.mapped_ranges == 9 && statistics.mapped_bytes == fixture.pair_size * 3,
                             "GGUF mapping touches three selected ranges per expert");
    }
    {
        ExpertCache merged(fixture.pair_size * 2, 1, {}, ExpertIoMode::Buffered, ExpertCacheCrossExpertReadCoalescing);
        const std::array<std::string, 2> keys = {ExpertCache::make_pair_key(gate_zero, down_zero),
                                                 ExpertCache::make_pair_key(gate_one, down_one)};
        const std::array<ExpertCachePairRequest, 2> requests = {{{&gate_zero, &down_zero, 0, keys[0]},
                                                                 {&gate_one, &down_one, 0, keys[1]}}};
        auto queued = merged.request_pairs(requests);
        gguf_cache_check(queued && queued.value() == requests.size(), "GGUF demand batch admits exact expert ranges");
        auto first = merged.acquire_pair(gate_zero, down_zero);
        auto second = merged.acquire_pair(gate_one, down_one);
        gguf_cache_check(first && second && merged.statistics().bytes_read == fixture.pair_size * 2,
                         "GGUF demand batches bypass split-layout coalescing safely");
        fixture.check_output(first.value(), 0);
        fixture.check_output(second.value(), 1);
    }
    auto malformed = std::make_shared<MxFp4FileStorage>(*gate_zero.mxfp4_file_storage);
    --malformed->blocks_size;
    TensorData invalid = gate_zero;
    invalid.mxfp4_file_storage = malformed;
    ExpertCache cache(fixture.pair_size, 1, {}, ExpertIoMode::Buffered);
    gguf_cache_check(!cache.request_pair(invalid, down_zero), "GGUF malformed range is rejected before admission");
    malformed->blocks_size = fixture.matrix_size;
    malformed->blocks_offset = std::numeric_limits<uint64_t>::max() - fixture.matrix_size + 1;
    gguf_cache_check(!cache.request_pair(invalid, down_zero), "GGUF overflowing file offset is rejected");
    malformed->blocks_offset = fixture.file.size();
    gguf_cache_check(!cache.acquire_pair(invalid, down_zero), "GGUF truncated file range fails exact read");
}

} // namespace moe
} // namespace ncnn

int main()
{
    try
    {
        ncnn::moe::gguf_unpack_nibble_order();
        ncnn::moe::gguf_expert_cache_ranges();
        std::cout << "GGUF MXFP4 cache tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "GGUF MXFP4 cache test failed: " << error.what() << '\n';
        return 1;
    }
}
