#include "kernels/mxfp4.h"
#include "kernels/ops.h"
#include "storage/expertcache.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

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
    const std::array<uint8_t, 34> unchanged = source;
    gguf_cache_check(!unpack_gguf_mxfp4(source, std::span<uint8_t>(source).first(32), scales),
                     "GGUF unpack rejects source/block aliasing");
    gguf_cache_check(!unpack_gguf_mxfp4(source, blocks, std::span<uint8_t>(source).last(2)),
                     "GGUF unpack rejects source/scale aliasing");
    gguf_cache_check(!unpack_gguf_mxfp4(source, blocks, std::span<uint8_t>(blocks).subspan(15, 2)),
                     "GGUF unpack rejects block/scale aliasing");
    gguf_cache_check(source == unchanged, "GGUF invalid aliasing is rejected before modifying the source");
}

static void gguf_unpack_all_codes()
{
    constexpr size_t block_count = 65536;
    std::vector<uint8_t> source(block_count * 17 + 2, 0xcd);
    std::vector<uint8_t> blocks(block_count * 16 + 6, 0xcd);
    std::vector<uint8_t> scales(block_count + 2, 0xcd);
    for (size_t block = 0; block < block_count; ++block)
    {
        source[1 + block * 17] = static_cast<uint8_t>(block);
        for (size_t pair = 0; pair < 8; ++pair)
        {
            source[2 + block * 17 + pair * 2] = static_cast<uint8_t>((block >> 8) + pair * 13);
            source[3 + block * 17 + pair * 2] = static_cast<uint8_t>(block + pair * 7);
        }
    }
    gguf_cache_check(unpack_gguf_mxfp4({source.data() + 1, block_count * 17},
                                       {blocks.data() + 3, block_count * 16},
                                       {scales.data() + 1, block_count}),
                     "GGUF unpack accepts unaligned buffers and many blocks");
    for (size_t block = 0; block < block_count; ++block)
    {
        gguf_cache_check(scales[block + 1] == source[block * 17 + 1], "GGUF exponent bytes are bit exact");
        for (size_t column = 0; column < 32; ++column)
        {
            const uint8_t expected = static_cast<uint8_t>((source[block * 17 + 2 + column % 16] >> (column < 16 ? 0 : 4)) & 15u);
            const uint8_t actual = static_cast<uint8_t>((blocks[block * 16 + 3 + column / 2] >> ((column % 2) * 4)) & 15u);
            gguf_cache_check(actual == expected, "GGUF unpack retains every pair of byte codes");
        }
    }
    gguf_cache_check(blocks.front() == 0xcd && blocks[2] == 0xcd && blocks.back() == 0xcd
                         && scales.front() == 0xcd && scales.back() == 0xcd,
                     "GGUF unpack writes only its destination spans");
}

struct GgufExpertFixture
{
    static constexpr uint32_t rows = 32;
    static constexpr uint32_t experts = 3;
    static constexpr uint64_t matrix_size = rows * 17;
    static constexpr uint64_t pair_size = matrix_size * 3;
    std::filesystem::path directory;
    std::filesystem::path path;
    std::filesystem::path separate_path;
    std::vector<uint8_t> file;
    std::array<std::array<uint64_t, experts>, 3> offsets{};
    std::array<std::array<uint64_t, experts>, 2> separate_offsets{};

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

        // A separate-layout copy uses an independent scalar nibble oracle,
        // so both encodings exercise the same host ARC policy and CPU result.
        separate_path = directory / "banks.mxfp4";
        std::vector<uint8_t> separate;
        for (uint32_t expert = 0; expert < experts; ++expert)
        {
            for (uint32_t matrix = 0; matrix < 2; ++matrix)
            {
                const uint32_t output_rows = matrix == 0 ? rows * 2 : rows;
                const size_t base = separate.size();
                separate_offsets[matrix][expert] = base;
                separate.resize(base + output_rows * 17);
                for (uint32_t row = 0; row < output_rows; ++row)
                {
                    const uint32_t projection = matrix == 0 ? row % 2 : 2;
                    const uint32_t source_row = matrix == 0 ? row / 2 : row;
                    const uint8_t* raw = file.data() + offsets[projection][expert] + source_row * 17;
                    separate[base + output_rows * 16 + row] = raw[0];
                    for (uint32_t pair = 0; pair < 16; ++pair)
                    {
                        const uint32_t column = pair * 2;
                        const uint8_t low = static_cast<uint8_t>((raw[1 + column % 16] >> (column < 16 ? 0 : 4)) & 15u);
                        const uint8_t high = static_cast<uint8_t>((raw[1 + (column + 1) % 16] >> (column < 16 ? 0 : 4)) & 15u);
                        separate[base + row * 16 + pair] = static_cast<uint8_t>(low | high << 4);
                    }
                }
            }
        }
        std::ofstream separate_stream(separate_path, std::ios::binary);
        separate_stream.write(reinterpret_cast<const char*>(separate.data()), static_cast<std::streamsize>(separate.size()));
        gguf_cache_check(static_cast<bool>(separate_stream), "write separate MXFP4 expert fixture");
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

    TensorData separate_tensor(uint32_t expert, bool gate_up) const
    {
        TensorData result;
        result.dtype = DType::MxFp4;
        result.shape = {gate_up ? rows * 2 : rows, rows};
        auto storage = std::make_shared<MxFp4FileStorage>();
        storage->blocks_path = storage->scales_path = separate_path.string();
        storage->blocks_offset = separate_offsets[gate_up ? 0 : 1][expert];
        storage->blocks_size = static_cast<uint64_t>(result.shape[0]) * 16;
        storage->scales_offset = storage->blocks_offset + storage->blocks_size;
        storage->scales_size = result.shape[0];
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
        gguf_cache_check(lease.gate_up->mxfp4_blocks.data() + rows * 32 == lease.gate_up->mxfp4_scales.data()
                             && lease.down->mxfp4_blocks.data() + rows * 16 == lease.down->mxfp4_scales.data(),
                         "GGUF admission shares final blocks/scales storage without a resident copy");
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

struct GgufDirectFixture
{
    struct Bank
    {
        uint32_t hidden = 0;
        uint32_t intermediate = 0;
        uint64_t matrix_size = 0;
        std::filesystem::path path;
        std::vector<uint8_t> bytes;
        std::array<uint64_t, 3> offsets{};

        TensorData tensor(bool gate_up, const std::filesystem::path& override_path = {}) const
        {
            TensorData result;
            result.dtype = DType::MxFp4;
            result.shape = gate_up ? std::vector<uint32_t>{intermediate * 2, hidden}
                                   : std::vector<uint32_t>{hidden, intermediate};
            auto storage = std::make_shared<MxFp4FileStorage>();
            storage->layout = MxFp4FileLayout::GgufBlockInterleaved;
            storage->blocks_path = (override_path.empty() ? path : override_path).string();
            storage->blocks_offset = offsets[gate_up ? 0 : 2];
            storage->blocks_size = matrix_size;
            if (gate_up)
            {
                storage->secondary_blocks_path = storage->blocks_path;
                storage->secondary_blocks_offset = offsets[1];
                storage->secondary_blocks_size = matrix_size;
                storage->interleave_rows = true;
            }
            result.mxfp4_file_storage = std::move(storage);
            return result;
        }

        uint64_t pair_size() const
        {
            return matrix_size * 3;
        }

        void check_output(const ExpertCacheLease& lease) const
        {
            gguf_cache_check(lease.gate_up->mxfp4_blocks.size() == matrix_size / 17 * 32
                                 && lease.gate_up->mxfp4_scales.size() == matrix_size / 17 * 2
                                 && lease.down->mxfp4_blocks.size() == matrix_size / 17 * 16
                                 && lease.down->mxfp4_scales.size() == matrix_size / 17,
                             "Direct GGUF final payload has exactly the selected logical dimensions");
            gguf_cache_check(lease.gate_up->mxfp4_blocks.data() + lease.gate_up->mxfp4_blocks.size() == lease.gate_up->mxfp4_scales.data()
                                 && lease.gate_up->mxfp4_scales.data() + lease.gate_up->mxfp4_scales.size() == lease.down->mxfp4_blocks.data()
                                 && lease.down->mxfp4_blocks.data() + lease.down->mxfp4_blocks.size() == lease.down->mxfp4_scales.data(),
                             "all GGUF I/O modes publish one final Gate/Up/Down allocation");
            for (uint32_t projection = 0; projection < 3; ++projection)
            {
                const TensorData& tensor = projection == 2 ? *lease.down : *lease.gate_up;
                const uint32_t rows = projection == 2 ? hidden : intermediate;
                const uint32_t columns = projection == 2 ? intermediate : hidden;
                const size_t blocks_per_row = columns / 32;
                for (uint32_t row = 0; row < rows; ++row)
                {
                    const size_t output_row = projection == 2 ? row : row * 2 + projection;
                    for (size_t block = 0; block < blocks_per_row; ++block)
                    {
                        const uint8_t* source = bytes.data() + offsets[projection] + (row * blocks_per_row + block) * 17;
                        const size_t output_block = output_row * blocks_per_row + block;
                        gguf_cache_check(tensor.mxfp4_scales[output_block] == source[0], "Direct GGUF exponent byte parity");
                        for (uint32_t column = 0; column < 32; ++column)
                        {
                            const uint8_t expected = static_cast<uint8_t>((source[1 + column % 16] >> (column < 16 ? 0 : 4)) & 15u);
                            const uint8_t actual = static_cast<uint8_t>((tensor.mxfp4_blocks[output_block * 16 + column / 2] >> (column % 2 * 4)) & 15u);
                            gguf_cache_check(actual == expected, "Direct GGUF byte oracle preserves prefix, row order, and logical EOF tail");
                        }
                    }
                }
            }
        }
    };

    std::filesystem::path directory;
    std::array<Bank, 6> banks;

    GgufDirectFixture()
    {
        directory = std::filesystem::temp_directory_path()
                    / ("ncnn_moe_gguf_direct_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(directory);
        const std::array<uint32_t, 6> hidden = {64, 64, 64, 96, 64, 64};
        const std::array<uint32_t, 6> intermediate = {32, 2048, 4096, 64, 4096, 4096};
        for (size_t index = 0; index < banks.size(); ++index)
        {
            Bank& bank = banks[index];
            bank.hidden = hidden[index];
            bank.intermediate = intermediate[index];
            bank.matrix_size = static_cast<uint64_t>(bank.hidden) * bank.intermediate / 32 * 17;
            bank.path = directory / ("bank_" + std::to_string(index) + ".gguf");
            // The small bank needs a 64 KiB aligned read; the other banks
            // cross 64 KiB with nonzero prefixes and grow the aligned request.
            bank.bytes.resize(index == 0 ? 17 : 64 * 1024 - 7 - index, 0x91);
            for (uint32_t projection = 0; projection < 3; ++projection)
            {
                if (projection != 0) bank.bytes.resize(bank.bytes.size() + projection * 19 + index, 0x37);
                bank.offsets[projection] = bank.bytes.size();
                for (uint64_t block = 0; block < bank.matrix_size / 17; ++block)
                {
                    bank.bytes.push_back(static_cast<uint8_t>(122 + (block + projection + index) % 5));
                    for (uint32_t code = 0; code < 16; ++code)
                        bank.bytes.push_back(static_cast<uint8_t>((block * 73 + projection * 29 + index * 17 + code * 41) & 255));
                }
            }
            gguf_cache_check(bank.bytes.size() % (64 * 1024) != 0
                                 && bank.offsets[2] + bank.matrix_size == bank.bytes.size(),
                             "Direct fixture ends at a complete logical range inside an incomplete aligned EOF page");
            std::ofstream stream(bank.path, std::ios::binary);
            stream.write(reinterpret_cast<const char*>(bank.bytes.data()), static_cast<std::streamsize>(bank.bytes.size()));
            gguf_cache_check(static_cast<bool>(stream), "write variable-size Direct GGUF fixture");
        }
    }

    ~GgufDirectFixture()
    {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }
};

static bool gguf_direct_supported(const std::filesystem::path& path)
{
#if defined(_WIN32)
    // Probe the filesystem independently: a regression that silently routes
    // every Expert range through Buffered must fail on a Direct-capable disk.
    const HANDLE handle = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, nullptr);
    const DWORD open_error = handle == INVALID_HANDLE_VALUE ? GetLastError() : ERROR_SUCCESS;
    uint8_t* allocation = static_cast<uint8_t*>(VirtualAlloc(nullptr, 64 * 1024, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!allocation)
    {
        if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle);
        throw std::runtime_error("cannot allocate independent Direct filesystem probe buffer");
    }
    DWORD completed = 0;
    const BOOL read = handle != INVALID_HANDLE_VALUE && allocation
                          ? ReadFile(handle, allocation, 64 * 1024, &completed, nullptr)
                          : FALSE;
    const DWORD read_error = read ? ERROR_SUCCESS : GetLastError();
    if (allocation) VirtualFree(allocation, 0, MEM_RELEASE);
    if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle);
    if (read && completed == 64 * 1024)
        return true;
    std::cout << "SKIP true Direct GGUF evidence: independent NO_BUFFERING filesystem probe failed (open="
              << open_error << ", read=" << read_error << ", bytes=" << completed << "); fallback parity remains required\n";
#else
    (void)path;
    std::cout << "SKIP true Direct GGUF evidence: this platform uses portable Buffered fallback\n";
#endif
    return false;
}

class GgufReadBarrier final : public ExpertVictimCache
{
public:
    void admit(std::string, std::shared_ptr<const TensorData>, std::shared_ptr<const TensorData>, ExpertVictimExecutionMetadata) override
    {
    }

    std::optional<ExpertVictimPair> restore(const std::string& key, const TensorData&, const TensorData&) override
    {
        if (key != "barrier")
            return std::nullopt;
        std::unique_lock<std::mutex> lock(mutex);
        entered = true;
        changed.notify_all();
        changed.wait(lock, [this] { return released; });
        return std::nullopt;
    }

    bool wait_until_entered()
    {
        std::unique_lock<std::mutex> lock(mutex);
        return changed.wait_for(lock, std::chrono::seconds(5), [this] { return entered; });
    }

    void release()
    {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        changed.notify_all();
    }

    void wait_for_background_work() override
    {
    }

    ExpertVictimCacheStatistics statistics() const override
    {
        return {};
    }

    uint64_t capacity() const noexcept override
    {
        return 0;
    }

private:
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false;
    bool released = false;
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
        auto barrier = std::make_shared<GgufReadBarrier>();
        ExpertCache merged(fixture.pair_size * 3, 1, barrier, ExpertIoMode::Buffered, ExpertCacheCrossExpertReadCoalescing);
        auto blocked = merged.request_pair(fixture.tensor(2, true), fixture.tensor(2, false), 0, "barrier");
        const bool entered = blocked && barrier->wait_until_entered();
        const std::array<std::string, 2> keys = {ExpertCache::make_pair_key(gate_zero, down_zero),
                                                 ExpertCache::make_pair_key(gate_one, down_one)};
        const std::array<ExpertCachePairRequest, 2> requests = {{{&gate_zero, &down_zero, 0, keys[0]},
                                                                 {&gate_one, &down_one, 0, keys[1]}}};
        auto queued = merged.request_pairs(requests);
        barrier->release();
        gguf_cache_check(entered, "GGUF coalescing test blocks the worker before queuing neighbors");
        gguf_cache_check(queued && queued.value() == requests.size(), "GGUF demand batch admits exact expert ranges");
        auto first = merged.acquire_pair(gate_zero, down_zero);
        auto second = merged.acquire_pair(gate_one, down_one);
        gguf_cache_check(first && second && merged.statistics().bytes_read == fixture.pair_size * 3,
                         "GGUF demand batches charge only requested expert payloads");
        fixture.check_output(first.value(), 0);
        fixture.check_output(second.value(), 1);
        const ExpertCacheStatistics statistics = merged.statistics();
        gguf_cache_check(statistics.coalesced_read_batches == 1 && statistics.coalesced_experts == 2
                             && statistics.coalesced_read_ranges_saved == 3
                             && statistics.buffered_read_ranges == 6,
                         "GGUF neighboring Gate/Up/Down bank ranges merge three exact reads");
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

static void gguf_buffered_scratch_and_failures()
{
    GgufExpertFixture fixture;
    const TensorData gate_zero = fixture.tensor(0, true);
    const TensorData down_zero = fixture.tensor(0, false);
    {
        ExpertCache cache(fixture.pair_size * 2, 1, {}, ExpertIoMode::Buffered);
        auto pinned = cache.acquire_pair(gate_zero, down_zero);
        gguf_cache_check(static_cast<bool>(pinned), "buffered GGUF pair is admitted");
        const auto& gate = *pinned.value().gate_up;
        const auto& down = *pinned.value().down;
        gguf_cache_check(gate.mxfp4_scales.data() + gate.mxfp4_scales.size() == down.mxfp4_blocks.data(),
                         "buffered GGUF Gate/Up and Down share one final resident allocation");
        for (uint32_t iteration = 0; iteration < 6; ++iteration)
        {
            const uint32_t expert = 1 + iteration % 2;
            auto current = cache.acquire_pair(fixture.tensor(expert, true), fixture.tensor(expert, false));
            gguf_cache_check(static_cast<bool>(current), "buffered scratch reuse admits the next unpinned expert");
            fixture.check_output(current.value(), expert);
            fixture.check_output(pinned.value(), 0);
        }
        const ExpertCacheStatistics statistics = cache.statistics();
        gguf_cache_check(statistics.resident_size == fixture.pair_size * 2
                             && statistics.resident_size == cache.capacity()
                             && statistics.evictions == 5
                             && statistics.bytes_read == fixture.pair_size * 7
                             && statistics.buffered_read_bytes == fixture.pair_size * 7
                             && statistics.buffered_read_ranges == 21,
                         "scratch reuse leaves held leases intact and preserves bounded residency and exact I/O accounting");
    }
    // Fail after the Gate read, and after both Gate/Up reads. Neither a
    // partially written resident allocation nor its reservation may escape.
    for (bool fail_down : {false, true})
    {
        TensorData gate = gate_zero;
        TensorData down = down_zero;
        if (fail_down)
        {
            auto storage = std::make_shared<MxFp4FileStorage>(*down.mxfp4_file_storage);
            storage->blocks_offset = fixture.file.size();
            down.mxfp4_file_storage = std::move(storage);
        }
        else
        {
            auto storage = std::make_shared<MxFp4FileStorage>(*gate.mxfp4_file_storage);
            storage->secondary_blocks_offset = fixture.file.size();
            gate.mxfp4_file_storage = std::move(storage);
        }
        ExpertCache cache(fixture.pair_size, 1, {}, ExpertIoMode::Buffered);
        gguf_cache_check(!cache.acquire_pair(gate, down), "a partial GGUF pair read fails admission");
        const ExpertCacheStatistics failed = cache.statistics();
        gguf_cache_check(failed.resident_size == 0 && failed.bytes_read == 0
                             && failed.buffered_read_ranges == (fail_down ? 2u : 1u),
                         "partial GGUF reads release resident capacity and count only completed raw ranges");
        auto recovered = cache.acquire_pair(fixture.tensor(1, true), fixture.tensor(1, false));
        gguf_cache_check(static_cast<bool>(recovered), "the worker scratch remains usable after a failed pair");
        fixture.check_output(recovered.value(), 1);
        gguf_cache_check(cache.statistics().resident_size == fixture.pair_size
                             && cache.statistics().bytes_read == fixture.pair_size,
                         "recovery publishes only a complete expert pair");
    }
}

static void gguf_auto_direct_scratch_and_failures()
{
    GgufDirectFixture fixture;
    [[maybe_unused]] const bool direct_supported = gguf_direct_supported(fixture.banks[2].path);
    const auto& pinned_bank = fixture.banks[0];
    const uint64_t capacity = fixture.banks[2].pair_size() * 2;
    // The largest three aligned requests occupy 640 KiB. This budget allows
    // all three slots, but its held small pin prevents caching two large pairs.
    gguf_cache_check(capacity >= 10 * 64 * 1024, "Direct pipeline budget covers three aligned requests");
    for (ExpertIoMode mode : {ExpertIoMode::Auto, ExpertIoMode::Direct, ExpertIoMode::Mmap})
    {
        // The default constructor argument, rather than only an explicit Auto,
        // must exercise the runtime's normal adaptive policy.
        auto cache = mode == ExpertIoMode::Auto ? std::make_unique<ExpertCache>(capacity, 1)
                                                : std::make_unique<ExpertCache>(capacity, 1, nullptr, mode);
        auto pinned = cache->acquire_pair(pinned_bank.tensor(true), pinned_bank.tensor(false));
        gguf_cache_check(static_cast<bool>(pinned), "default Auto, Direct and Mmap admit the held GGUF Expert");
        pinned_bank.check_output(pinned.value());
        uint64_t expected_bytes = pinned_bank.pair_size();
        size_t requests = 1;
        for (size_t bank_index : {size_t{3}, size_t{1}, size_t{2}, size_t{4}, size_t{5}, size_t{2}, size_t{4}, size_t{5}})
        {
            const auto& bank = fixture.banks[bank_index];
            const auto before = cache->statistics();
            auto loaded = cache->acquire_pair(bank.tensor(true), bank.tensor(false));
            gguf_cache_check(static_cast<bool>(loaded), "variable Direct bank sizes admit after scratch reuse and growth");
            bank.check_output(loaded.value());
            pinned_bank.check_output(pinned.value());
            // The held pin makes two large pairs exceed capacity; cycling
            // independent banks therefore exercises new reads rather than hits.
            gguf_cache_check(cache->statistics().misses == before.misses + 1, "Direct fixture sequence requires an actual read each time");
            expected_bytes += bank.pair_size();
            ++requests;
        }
        const auto statistics = cache->statistics();
        gguf_cache_check(statistics.bytes_read == expected_bytes && statistics.resident_size <= cache->capacity(),
                         "Direct raw prefixes and aligned EOF suffixes do not inflate logical residency or byte counts");
        if (mode == ExpertIoMode::Mmap)
        {
            gguf_cache_check(statistics.mapped_ranges == requests * 3 && statistics.mapped_bytes == expected_bytes
                                 && statistics.direct_read_ranges == 0 && statistics.buffered_read_ranges == 0
                                 && statistics.direct_pipeline_batches == 0 && statistics.direct_arena_allocations == 0,
                             "Mmap keeps its mapping and prefault contract after unified final-pair allocation");
        }
#if defined(_WIN32)
        else if (direct_supported)
        {
            gguf_cache_check(statistics.direct_read_ranges == requests * 3 && statistics.direct_read_bytes == expected_bytes
                                 && statistics.direct_read_fallbacks == 0 && statistics.buffered_read_ranges == 0,
                             "Direct-capable Windows must execute true Direct I/O, including complete logical ranges at a short aligned EOF tail");
            gguf_cache_check(statistics.direct_pipeline_batches == requests
                                 && statistics.direct_pipeline_completed_ranges == requests * 3
                                 && statistics.direct_arena_allocations > 1
                                 && statistics.direct_arena_allocated_bytes >= 10 * 64 * 1024
                                 && statistics.direct_pipeline_peak_pending <= 3,
                             "three accepted Direct submissions precede each wave wait while the bounded arena grows across banks");
        }
#endif
        else
        {
            gguf_cache_check(statistics.direct_read_bytes + statistics.buffered_read_bytes == expected_bytes
                                 && statistics.direct_read_ranges + statistics.buffered_read_ranges == requests * 3,
                             "unsupported Direct keeps explicit fallback parity and counts only completed logical ranges");
        }
    }

    // Smaller caps retain two or one slots, or cannot retain even one aligned
    // page. Every case must preserve Direct rather than silently use Buffered.
    for (ExpertIoMode mode : {ExpertIoMode::Auto, ExpertIoMode::Direct})
    {
        for (uint64_t raw_cap : {uint64_t{3 * 64 * 1024}, uint64_t{2 * 64 * 1024}, uint64_t{64 * 1024}, pinned_bank.pair_size()})
        {
            ExpertCache cache(raw_cap, 1, {}, mode);
            auto loaded = cache.acquire_pair(pinned_bank.tensor(true), pinned_bank.tensor(false));
            gguf_cache_check(static_cast<bool>(loaded), "bounded Direct waves and over-cap transient reads admit a complete Expert");
            pinned_bank.check_output(loaded.value());
            const auto statistics = cache.statistics();
            gguf_cache_check(statistics.bytes_read == pinned_bank.pair_size() && statistics.resident_size == pinned_bank.pair_size(),
                             "transient aligned raw memory does not enlarge the ARC resident payload");
            gguf_cache_check(statistics.direct_read_ranges + statistics.buffered_read_ranges == 3
                                 && statistics.direct_read_bytes + statistics.buffered_read_bytes == pinned_bank.pair_size(),
                             "transient Direct or its portable fallback reads exactly three logical ranges");
#if defined(_WIN32)
            if (direct_supported)
            {
                gguf_cache_check(statistics.direct_read_ranges == 3 && statistics.direct_read_bytes == pinned_bank.pair_size()
                                     && statistics.buffered_read_ranges == 0 && statistics.direct_read_fallbacks == 0,
                                 "over-cap requests preserve true Direct rather than silently using Buffered");
                const uint64_t batched_ranges = raw_cap >= 3 * 64 * 1024 ? 3 : raw_cap >= 2 * 64 * 1024 ? 2
                                                                                                        : 0;
                gguf_cache_check(statistics.direct_pipeline_batches == (batched_ranges != 0 ? 1u : 0u)
                                     && statistics.direct_pipeline_completed_ranges == batched_ranges
                                     && statistics.direct_arena_allocated_bytes <= raw_cap,
                                 "raw budget selects three, two or one slots without hiding extra retained allocations");
                if (raw_cap < 64 * 1024)
                    gguf_cache_check(statistics.direct_arena_allocations == 0 && statistics.direct_arena_allocated_bytes == 0,
                                     "a raw range above the arena cap keeps only a transient Direct allocation");
                else
                    gguf_cache_check(statistics.direct_arena_allocations > 0,
                                     "eligible Direct raw waves retain their bounded aligned arena");
                // Small uncached reads can complete synchronously. The batch
                // counter proves submission ordering without requiring a
                // positive sample of physical ERROR_IO_PENDING operations.
                gguf_cache_check(statistics.direct_pipeline_peak_pending <= 3,
                                 "sampled real pending I/O never exceeds the fixed three-slot bound");
            }
#endif
        }
    }

    const auto& bank = fixture.banks[2];
    for (ExpertIoMode mode : {ExpertIoMode::Auto, ExpertIoMode::Direct})
    {
        for (uint32_t failed_projection : {1u, 2u})
        {
            const std::filesystem::path truncated = fixture.directory / ("truncated_" + std::to_string(failed_projection) + ".gguf");
            {
                std::ofstream stream(truncated, std::ios::binary | std::ios::trunc);
                stream.write(reinterpret_cast<const char*>(bank.bytes.data()), static_cast<std::streamsize>(bank.offsets[failed_projection] + bank.matrix_size - 1));
                gguf_cache_check(static_cast<bool>(stream), "write one-byte-short Up or Down Direct range");
            }
            ExpertCache cache(capacity, 1, {}, mode);
            auto pinned = cache.acquire_pair(pinned_bank.tensor(true), pinned_bank.tensor(false));
            gguf_cache_check(static_cast<bool>(pinned), "hold a healthy lease across a failed Direct pipeline");
            const auto before = cache.statistics();
            TensorData failing_gate = bank.tensor(true);
            TensorData failing_down = bank.tensor(false);
            if (failed_projection == 1)
            {
                auto storage = std::make_shared<MxFp4FileStorage>(*failing_gate.mxfp4_file_storage);
                storage->secondary_blocks_path = truncated.string();
                failing_gate.mxfp4_file_storage = std::move(storage);
            }
            else
            {
                auto storage = std::make_shared<MxFp4FileStorage>(*failing_down.mxfp4_file_storage);
                storage->blocks_path = truncated.string();
                failing_down.mxfp4_file_storage = std::move(storage);
            }
            gguf_cache_check(!cache.acquire_pair(failing_gate, failing_down),
                             "Direct and Buffered fallback both reject a one-byte-short logical range");
            const auto failed = cache.statistics();
            const uint64_t completed_ranges = failed.direct_read_ranges + failed.buffered_read_ranges
                                              - before.direct_read_ranges - before.buffered_read_ranges;
            const uint64_t completed_bytes = failed.direct_read_bytes + failed.buffered_read_bytes
                                             - before.direct_read_bytes - before.buffered_read_bytes;
            gguf_cache_check(failed.resident_size == before.resident_size && failed.bytes_read == before.bytes_read
                                 && completed_ranges <= 2 && completed_bytes == completed_ranges * bank.matrix_size,
                             "failed Direct pipeline counts every completed healthy range independently of order and releases only its own reservation");
            pinned_bank.check_output(pinned.value());
#if defined(_WIN32)
            if (direct_supported)
            {
                gguf_cache_check(failed.direct_read_fallbacks > before.direct_read_fallbacks,
                                 "failed Direct attempt is counted before the exact Buffered retry");
                gguf_cache_check(failed.direct_pipeline_completed_ranges - before.direct_pipeline_completed_ranges <= completed_ranges
                                     && failed.direct_pipeline_peak_pending <= 3,
                                 "error cleanup accounts successful Direct completions without inventing a completion order");
            }
#endif
            auto recovered = cache.acquire_pair(bank.tensor(true), bank.tensor(false));
            gguf_cache_check(static_cast<bool>(recovered), "the same worker recovers after partial Direct read failure");
            bank.check_output(recovered.value());
            pinned_bank.check_output(pinned.value());
            const auto after = cache.statistics();
            gguf_cache_check(after.resident_size == before.resident_size + bank.pair_size()
                                 && after.bytes_read == before.bytes_read + bank.pair_size(),
                             "failed admission cannot leak capacity into the next complete pair");
            if (mode == ExpertIoMode::Auto)
            {
                gguf_cache_check(after.adaptive_read_policy == 1
                                     && after.direct_read_ranges == failed.direct_read_ranges
                                     && after.direct_read_fallbacks == failed.direct_read_fallbacks
                                     && after.direct_pipeline_batches == failed.direct_pipeline_batches
                                     && after.direct_pipeline_completed_ranges == failed.direct_pipeline_completed_ranges
                                     && after.buffered_read_ranges == failed.buffered_read_ranges + 3
                                     && after.buffered_read_bytes == failed.buffered_read_bytes + bank.pair_size(),
                                 "Auto fallback persists for later GGUF ranges and preserves Buffered statistics");
            }
#if defined(_WIN32)
            else if (direct_supported)
            {
                gguf_cache_check(after.direct_read_ranges == failed.direct_read_ranges + 3
                                     && after.direct_read_bytes == failed.direct_read_bytes + bank.pair_size()
                                     && after.direct_read_fallbacks == failed.direct_read_fallbacks
                                     && after.direct_pipeline_batches == failed.direct_pipeline_batches + 1
                                     && after.direct_pipeline_completed_ranges == failed.direct_pipeline_completed_ranges + 3,
                                 "explicit Direct retries true Direct on the next healthy bank after failure");
            }
#endif

            // Hold the worker in a public victim-cache callback until shutdown
            // is about to begin. The read already owns an ARC reservation;
            // after release the destructor must join its successful/error
            // cleanup before freeing events or the raw arena. Completion of
            // the actual filesystem I/O may still be synchronous.
            for (bool fail_during_shutdown : {false, true})
            {
                auto barrier = std::make_shared<GgufReadBarrier>();
                auto stopping = std::make_unique<ExpertCache>(capacity, 1, barrier, mode);
                auto survivor = stopping->acquire_pair(pinned_bank.tensor(true), pinned_bank.tensor(false));
                gguf_cache_check(static_cast<bool>(survivor), "hold a final resident lease beyond Direct worker shutdown");
                auto queued = stopping->request_pair(fail_during_shutdown ? failing_gate : bank.tensor(true),
                                                     fail_during_shutdown ? failing_down : bank.tensor(false), 0, "barrier");
                const bool entered = queued && barrier->wait_until_entered();
                barrier->release();
                gguf_cache_check(entered, "Direct shutdown test has a reserved read already executing on the worker");
                stopping.reset();
                pinned_bank.check_output(survivor.value());
            }
        }
    }
}

static void host_arc_class_precedes_layer_fair_share()
{
    GgufExpertFixture fixture;
    for (bool separate : {false, true})
    {
        const auto tensor = [&](uint32_t expert, bool gate_up) {
            return separate ? fixture.separate_tensor(expert, gate_up) : fixture.tensor(expert, gate_up);
        };
        const TensorData hot_gate = tensor(0, true);
        const TensorData hot_down = tensor(0, false);
        const TensorData recent_gate = tensor(1, true);
        const TensorData recent_down = tensor(1, false);
        ExpertCache cache(fixture.pair_size * 2, 1, {}, ExpertIoMode::Buffered, 0, 2);
        for (uint32_t access = 0; access < 2; ++access)
        {
            auto hot = cache.acquire_pair(hot_gate, hot_down, 0);
            gguf_cache_check(static_cast<bool>(hot), "host ARC promotes the repeatedly demanded expert");
            fixture.check_output(hot.value(), 0);
        }
        {
            auto recent = cache.acquire_pair(recent_gate, recent_down, 1);
            gguf_cache_check(static_cast<bool>(recent), "host ARC fills the other layer with a recent entry");
        }
        {
            auto incoming = cache.acquire_pair(tensor(2, true), tensor(2, false), 0);
            gguf_cache_check(static_cast<bool>(incoming), "host ARC admits another expert from the hot entry's layer");
            fixture.check_output(incoming.value(), 2);
        }
        gguf_cache_check(cache.is_ready(hot_gate, hot_down) && !cache.is_ready(recent_gate, recent_down),
                         "ARC evicts another layer's recent entry before the preferred layer's hot entry");
        const uint64_t bytes_before_hit = cache.statistics().bytes_read;
        auto hot = cache.acquire_pair(hot_gate, hot_down, 0);
        gguf_cache_check(hot && cache.statistics().bytes_read == bytes_before_hit
                             && bytes_before_hit == fixture.pair_size * 3
                             && cache.statistics().evictions == 1
                             && cache.statistics().resident_size == cache.capacity(),
                         "class-first ARC avoids a hot expert reload without increasing the cache budget");
        fixture.check_output(hot.value(), 0);
    }
}

} // namespace moe
} // namespace ncnn

int main()
{
    try
    {
        ncnn::moe::gguf_unpack_nibble_order();
        ncnn::moe::gguf_unpack_all_codes();
        ncnn::moe::gguf_expert_cache_ranges();
        ncnn::moe::gguf_buffered_scratch_and_failures();
        ncnn::moe::gguf_auto_direct_scratch_and_failures();
        ncnn::moe::host_arc_class_precedes_layer_fair_share();
        std::cout << "GGUF MXFP4 cache tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "GGUF MXFP4 cache test failed: " << error.what() << '\n';
        return 1;
    }
}
