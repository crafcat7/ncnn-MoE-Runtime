#include "qnk.h"

#include "ncnn/moe/option.h"

#include "mxfp4.h"
#include "qnk_msvc.h"
#include "activationbuffer.h"
#include "ops.h"
#include "vector.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace ncnn {
namespace moe {
bool qnk_q8_act(const DType* dtypes, size_t count) noexcept
{
    if (dtypes == nullptr || count == 0)
        return false;
    for (size_t index = 0; index < count; ++index)
    {
        const DType dtype = dtypes[index];
        if (dtype != DType::Q8_0 && dtype != DType::Q4_0 && dtype != DType::Q4_1
            && dtype != DType::Q5_0 && dtype != DType::Q5_1 && dtype != DType::IQ4NL)
            return false;
    }
    return true;
}

static inline const uint8_t* qnk_row0(const uint8_t* raw, const QnKPack* pack, size_t row,
                                      uint32_t block_count, size_t block_bytes, size_t& stride) noexcept
{
    stride = pack ? block_bytes * 8 : block_bytes;
    return pack ? qnk_packed_block(*pack, row, 0)
                : raw + row * static_cast<size_t>(block_count) * block_bytes;
}

// Non-owning view of a quantized activation row owned by QnkQ8Input (either the
// caller's shared input or the thread-local scratch).
struct QnkQ8Row
{
    const int8_t* codes = nullptr;
    const float* scales = nullptr;
    const int32_t* sums = nullptr;
};

template<bool paired>
static void qnk_gemm(const uint8_t* raw,
                     const QnKPack* packed,
                     DType dtype,
                     size_t rows,
                     uint32_t columns,
                     const float* input,
                     size_t input_stride,
                     size_t token_count,
                     float* output,
                     size_t output_stride,
                     const TensorData* matrix2 = nullptr,
                     const QnKPack* packed2 = nullptr,
                     ActivationBuffer* output2 = nullptr,
                     const QnkQ8Row* q8 = nullptr,
                     // Paired K-quant decode converts each matrix to its own
                     // Q8_0 buffer; pass matrix2's converted bytes here instead
                     // of its native block payload.
                     const uint8_t* raw2_override = nullptr) noexcept
{
    static_cast<void>(q8);
    assert(!q8 || token_count == 1);
    const uint32_t block_size = qnk_block_size(dtype);
    const uint32_t block_count = columns / block_size;
    const size_t block_bytes = qnk_block_bytes(dtype);
    size_t rows2 = 0;
    const uint8_t* raw2 = nullptr;
    float* output2_data = nullptr;
    size_t output_stride2 = 0;
    if constexpr (paired)
    {
        rows2 = matrix2->shape[0];
        raw2 = raw2_override != nullptr ? raw2_override : matrix2->qnk_values().data();
        output2_data = output2->row(0);
        output_stride2 = output2->columns();
    }

#if defined(NCNN_MOE_MSVC_X86_SIMD)
    // 32-value GGML blocks decode eight lanes per step inside the fused row
    // kernel; the ISA check stays out of the per-block loop.
    const MxFp4KernelKind simd_kind = block_size == 32 ? mxfp4_kernel_kind() : MxFp4KernelKind::Scalar;
    const bool use_simd_block32 = simd_kind == MxFp4KernelKind::X86Avx2 || simd_kind == MxFp4KernelKind::X86Avx512;
#endif
    const auto dot = [&](const uint8_t* data, const QnKPack* pack, float* result, size_t stride, size_t row)
#if defined(__GNUC__) || defined(__clang__)
                         __attribute__((always_inline))
#endif
    {
#if defined(__aarch64__) || defined(_M_ARM64)
        static_cast<void>(stride);
        float sum = 0.0f;
        for (uint32_t block = 0; block < block_count; ++block)
        {
            const uint8_t* encoded = pack
                                         ? qnk_packed_block(*pack, row, block)
                                         : data + ((row * static_cast<size_t>(block_count) + block) * block_bytes);
            sum += qnk_dot_block(dtype,
                                 encoded,
                                 input + static_cast<size_t>(block) * block_size);
        }
        result[row] = sum;
#else
#if defined(NCNN_MOE_MSVC_X86_SIMD)
        if (use_simd_block32)
        {
            size_t encoded_stride = 0;
            const uint8_t* first = qnk_row0(data, pack, row, block_count, block_bytes, encoded_stride);
            for (size_t token = 0; token < token_count; ++token)
            {
                result[token * stride + row] = msvc_avx2_qnk_dot_row(dtype,
                                                                     first,
                                                                     encoded_stride,
                                                                     block_count,
                                                                     input + token * input_stride);
            }
            return;
        }
#endif
        for (size_t token = 0; token < token_count; ++token)
        {
            const float* token_input = input + token * input_stride;
            float sum = 0.0f;
            for (uint32_t block = 0; block < block_count; ++block)
            {
                const uint8_t* encoded = pack
                                             ? qnk_packed_block(*pack, row, block)
                                             : data + ((row * static_cast<size_t>(block_count) + block) * block_bytes);
                sum += qnk_dot_block(dtype,
                                     encoded,
                                     token_input + static_cast<size_t>(block) * block_size);
            }
            result[token * stride + row] = sum;
        }
#endif
    };
#if defined(__aarch64__) || defined(_M_ARM64) || defined(NCNN_MOE_MSVC_X86_SIMD)
    // Single-token int8 rows are selected once per call instead of branching
    // inside the output-row loop.
    const auto row_q8 = [&](const uint8_t* data, const QnKPack* pack, float* result, size_t row) {
        size_t encoded_stride = 0;
        const uint8_t* first = qnk_row0(data, pack, row, block_count, block_bytes, encoded_stride);
#if defined(__aarch64__) || defined(_M_ARM64)
        result[row] = neon_qnk_q8_row(dtype, first, encoded_stride, block_count,
                                      q8->codes, q8->scales, q8->sums);
#else
        result[row] = msvc_avx2_qnk_q8_row(dtype, first, encoded_stride, block_count,
                                           q8->codes, q8->scales, q8->sums);
#endif
    };
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
    if (token_count == 1)
    {
#if defined(_OPENMP)
        const size_t task_count = rows + rows2;
        const uint64_t operation_count = static_cast<uint64_t>(task_count) * columns;
        const int team_size = static_cast<int>(std::min<size_t>(task_count,
                                                                static_cast<size_t>(cpu_linear_team_size(operation_count, dtype))));
        if (team_size > 1)
        {
            if (q8)
            {
                if constexpr (paired)
                {
#pragma omp parallel num_threads(team_size)
                    {
#pragma omp for schedule(static) nowait
                        for (int64_t row = 0; row < static_cast<int64_t>(rows); ++row)
                            row_q8(raw, packed, output, static_cast<size_t>(row));
#pragma omp for schedule(static) nowait
                        for (int64_t row = 0; row < static_cast<int64_t>(rows2); ++row)
                            row_q8(raw2, packed2, output2_data, static_cast<size_t>(row));
                    }
                }
                else
                {
#pragma omp parallel for schedule(static) num_threads(team_size)
                    for (int64_t row = 0; row < static_cast<int64_t>(rows); ++row)
                        row_q8(raw, packed, output, static_cast<size_t>(row));
                }
            }
            else if constexpr (paired)
            {
#pragma omp parallel num_threads(team_size)
                {
#pragma omp for schedule(static) nowait
                    for (int64_t row = 0; row < static_cast<int64_t>(rows); ++row)
                        dot(raw, packed, output, output_stride, static_cast<size_t>(row));
#pragma omp for schedule(static) nowait
                    for (int64_t row = 0; row < static_cast<int64_t>(rows2); ++row)
                        dot(raw2, packed2, output2_data, output_stride2, static_cast<size_t>(row));
                }
            }
            else
            {
#pragma omp parallel for schedule(static) num_threads(team_size)
                for (int64_t row = 0; row < static_cast<int64_t>(rows); ++row)
                    dot(raw, packed, output, output_stride, static_cast<size_t>(row));
            }
            return;
        }
#endif
        if (q8)
        {
            for (size_t row = 0; row < rows; ++row)
                row_q8(raw, packed, output, row);
            if constexpr (paired)
                for (size_t row = 0; row < rows2; ++row)
                    row_q8(raw2, packed2, output2_data, row);
            return;
        }
        for (size_t row = 0; row < rows; ++row)
            dot(raw, packed, output, output_stride, row);
        if constexpr (paired)
            for (size_t row = 0; row < rows2; ++row)
                dot(raw2, packed2, output2_data, output_stride2, row);
        return;
    }
#else
    if (token_count == 1)
    {
#if defined(NCNN_MOE_MSVC_X86_SIMD)
        if (q8)
        {
#if defined(_OPENMP)
            const uint64_t q8_operation_count = static_cast<uint64_t>(rows + rows2) * columns;
            const int q8_team_size = static_cast<int>(std::min<size_t>(rows + rows2, static_cast<size_t>(cpu_linear_team_size(q8_operation_count, dtype))));
#pragma omp parallel for schedule(static) num_threads(q8_team_size) if (q8_team_size > 1)
#endif
            for (int64_t index = 0; index < static_cast<int64_t>(rows + rows2); ++index)
            {
                if (static_cast<size_t>(index) < rows)
                    row_q8(raw, packed, output, static_cast<size_t>(index));
                else if constexpr (paired)
                    row_q8(raw2, packed2, output2_data, static_cast<size_t>(index) - rows);
            }
            return;
        }
#endif
#if defined(_OPENMP)
        const uint64_t operation_count = static_cast<uint64_t>(rows + rows2) * columns;
        // Every block layout benefits from row parallelism; 32-value formats are
        // no longer the only ones that would otherwise run this GEMV on one thread.
        const int team_size = static_cast<int>(std::min<size_t>(rows + rows2, static_cast<size_t>(cpu_linear_team_size(operation_count, dtype))));
#endif
        // Gate/up share one parallel region so a fused projection pays one fork.
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(team_size) if (team_size > 1)
#endif
        for (int64_t index = 0; index < static_cast<int64_t>(rows + rows2); ++index)
        {
            if (static_cast<size_t>(index) < rows)
                dot(raw, packed, output, output_stride, static_cast<size_t>(index));
            else if constexpr (paired)
                dot(raw2, packed2, output2_data, output_stride2, static_cast<size_t>(index) - rows);
        }
        return;
    }
#endif
    // Prefill decodes one 8-row weight tile per block and reuses it across four
    // tokens, matching the GPU kernel's token tile instead of re-decoding the
    // same block for every token.
    const size_t first_tiles = (rows + 7) / 8;
    const size_t tile_count = first_tiles + (rows2 + 7) / 8;
#if defined(_OPENMP)
    const uint64_t operation_count = static_cast<uint64_t>(rows + rows2) * columns * token_count;
    const int team_size = static_cast<int>(std::min<size_t>(tile_count, static_cast<size_t>(cpu_linear_team_size(operation_count, dtype))));
#pragma omp parallel for schedule(static) num_threads(team_size) if (team_size > 1)
#endif
    for (int64_t tile = 0; tile < static_cast<int64_t>(tile_count); ++tile)
    {
        size_t first_row = static_cast<size_t>(tile) * 8;
        size_t tile_rows = rows;
        const uint8_t* tile_raw = raw;
        const QnKPack* tile_packed = packed;
        float* tile_output = output;
        size_t tile_stride = output_stride;
        if constexpr (paired)
        {
            if (static_cast<size_t>(tile) >= first_tiles)
            {
                first_row = (static_cast<size_t>(tile) - first_tiles) * 8;
                tile_rows = rows2;
                tile_raw = raw2;
                tile_packed = packed2;
                tile_output = output2_data;
                tile_stride = output_stride2;
            }
        }
        const uint32_t output_count = static_cast<uint32_t>(std::min<size_t>(8, tile_rows - first_row));
        alignas(16) float decoded[8 * qnk_block_elements];
        for (uint32_t block = 0; block < block_count; ++block)
        {
            for (uint32_t row = 0; row < output_count; ++row)
            {
                const size_t weight_row = first_row + row;
                const uint8_t* encoded = tile_packed
                                             ? qnk_packed_block(*tile_packed, weight_row, block)
                                             : tile_raw + ((weight_row * block_count + block) * block_bytes);
                qnk_dequantize_block(dtype, encoded, decoded + static_cast<size_t>(row) * block_size);
            }
            for (size_t first_token = 0; first_token < token_count; first_token += 4)
            {
                const uint32_t input_count = static_cast<uint32_t>(std::min<size_t>(4, token_count - first_token));
                const float* input_tile = input + first_token * input_stride
                                          + static_cast<size_t>(block) * block_size;
                float* destination = tile_output + first_token * tile_stride + first_row;
                float partial[4 * 8];
                float_gemm_4x8(decoded,
                               block_size,
                               input_tile,
                               input_stride,
                               block_size,
                               output_count,
                               input_count,
                               block == 0 ? destination : partial,
                               block == 0 ? tile_stride : 8);
                if (block != 0)
                {
                    for (uint32_t token = 0; token < input_count; ++token)
                    {
                        float* token_output = tile_output + (first_token + token) * tile_stride + first_row;
                        const float* values = partial + token * 8;
                        for (uint32_t row = 0; row < output_count; ++row)
                            token_output[row] += values[row];
                    }
                }
            }
        }
    }
}

enum QnKNeed : uint32_t
{
    QnKNeedNative = 1u,
    QnKNeedQ8 = 2u
};

// K-quant -> Q8_0 blocks for the int8-activation dots.
static bool build_q8_compat(const uint8_t* raw, DType dtype, size_t rows, uint32_t columns,
                            std::vector<uint8_t>& output)
{
    const uint32_t block_size = qnk_block_size(dtype);
    const uint32_t source_blocks = block_size == 0 ? 0 : columns / block_size;
    const uint32_t target_blocks = columns / 32;
    const size_t source_bytes = qnk_block_bytes(dtype);
    if (block_size < 32 || source_bytes == 0 || target_blocks == 0 || columns % block_size != 0)
        return false;
    output.resize(rows * static_cast<size_t>(target_blocks) * 34);
    alignas(64) float decoded[256];
    for (size_t row = 0; row < rows; ++row)
        for (uint32_t block = 0; block < source_blocks; ++block)
        {
            const uint8_t* source = raw
                                    + (row * static_cast<size_t>(source_blocks) + block) * source_bytes;
            qnk_dequantize_block(dtype, source, decoded);
            for (uint32_t sub = 0; sub < block_size / 32; ++sub)
            {
                const float* values = decoded + sub * 32;
                float maximum = 0.0f;
                for (uint32_t index = 0; index < 32; ++index)
                    maximum = std::max(maximum, std::fabs(values[index]));
                const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
                uint8_t* destination = output.data()
                                       + (row * static_cast<size_t>(target_blocks)
                                          + block * (block_size / 32) + sub)
                                             * 34;
                const uint16_t half = float_to_f16(scale);
                std::memcpy(destination, &half, sizeof(half));
                const float inverse = 1.0f / scale;
                for (uint32_t index = 0; index < 32; ++index)
                {
                    const float scaled = std::max(-127.0f, std::min(127.0f, values[index] * inverse));
                    destination[2 + index] = static_cast<uint8_t>(static_cast<int8_t>(std::lround(scaled)));
                }
            }
        }
    return true;
}

// Q8_0 4-row x 4-token repack for the 4x8 GEMM.
static bool build_q8_pack(const uint8_t* raw, size_t rows, uint32_t columns,
                          std::vector<uint8_t>& output)
{
    if (raw == nullptr || rows == 0 || rows % 4 != 0 || columns % 32 != 0)
        return false;
    const uint32_t block_count = columns / 32;
    const size_t tile_count = rows / 4;
    output.resize(tile_count * block_count * 144);
    for (size_t tile = 0; tile < tile_count; ++tile)
        for (uint32_t block = 0; block < block_count; ++block)
        {
            uint8_t* destination = output.data() + (tile * block_count + block) * 144;
            for (uint32_t row = 0; row < 4; ++row)
            {
                const uint8_t* source = raw
                                        + ((tile * 4 + row) * static_cast<size_t>(block_count) + block) * 34;
                std::memcpy(destination + row * 2, source, 2);
            }
            for (uint32_t index = 0; index < 16; ++index)
            {
                const uint32_t row = index % 4;
                const uint32_t chunk = index / 4;
                const uint8_t* source = raw
                                        + ((tile * 4 + row) * static_cast<size_t>(block_count) + block) * 34 + 2
                                        + chunk * 8;
                std::memcpy(destination + 16 + index * 8, source, 8);
            }
        }
    return true;
}

// Lazily builds the requested derived layouts once and publishes them on the
// operator sidecar, so the lifetime follows the model instead of a
// process-global pointer-keyed cache.
static std::shared_ptr<const QnKPack> get_qnk_packed_weights(const TensorData& matrix,
                                                             std::shared_ptr<const QnKPack>* sidecar,
                                                             size_t rows,
                                                             uint32_t columns,
                                                             uint32_t need) noexcept
{
    static std::mutex build_locks[64];
    const std::span<const uint8_t> raw = matrix.qnk_values();
    if (raw.empty())
        return {};
    const auto satisfied = [&](const std::shared_ptr<const QnKPack>& pack) noexcept {
        if (!pack || !pack->valid_shape(matrix.dtype, rows, columns))
            return false;
        if ((need & QnKNeedNative) != 0 && !pack->valid())
            return false;
        if ((need & QnKNeedQ8) != 0 && pack->q8_compat.empty() && pack->q8_pack.empty())
            return false;
        return true;
    };
    // Published packs are immutable, so the steady-state read is lock-free.
    if (sidecar != nullptr)
    {
        const std::shared_ptr<const QnKPack> fast = std::atomic_load_explicit(sidecar, std::memory_order_acquire);
        if (satisfied(fast))
            return fast;
    }
    const uintptr_t storage_key = reinterpret_cast<uintptr_t>(raw.data());
    std::lock_guard<std::mutex> lock(build_locks[(storage_key >> 6) & 63u]);
    const std::shared_ptr<const QnKPack> cached = sidecar ? std::atomic_load_explicit(sidecar, std::memory_order_acquire) : nullptr;
    const bool shape_ok = cached && cached->valid_shape(matrix.dtype, rows, columns);
    const bool want_native = (need & QnKNeedNative) != 0 || (shape_ok && cached->valid());
    const bool want_q8 = (need & QnKNeedQ8) != 0
                         || (shape_ok && (!cached->q8_compat.empty() || !cached->q8_pack.empty()));
    if (shape_ok && (!want_native || cached->valid())
        && (!want_q8 || !cached->q8_compat.empty() || !cached->q8_pack.empty()))
        return cached;

    try
    {
        auto packed = std::make_shared<QnKPack>();
        if (want_native)
        {
            if (!qnk_pack_weights(raw.data(), raw.size(), matrix.dtype, rows, columns, *packed))
                return {};
        }
        else
        {
            packed->dtype = matrix.dtype;
            packed->rows = rows;
            packed->columns = columns;
            packed->block_count = columns / qnk_block_size(matrix.dtype);
        }
        if (want_q8)
        {
            if (qnk_block_size(matrix.dtype) == 32)
            {
                if (matrix.dtype != DType::Q8_0
                    || !build_q8_pack(raw.data(), rows, columns, packed->q8_pack))
                    return {};
            }
            else if (!build_q8_compat(raw.data(), matrix.dtype, rows, columns, packed->q8_compat))
            {
                return {};
            }
        }
        if (sidecar)
            std::atomic_store_explicit(sidecar, std::shared_ptr<const QnKPack>(packed), std::memory_order_release);
        return packed;
    }
    catch (...)
    {
        return {};
    }
}

// Q8 payload for the int8-activation kernels; the owner keeps it alive.
struct Q8Payload
{
    std::shared_ptr<const QnKPack> pack;
    const std::vector<uint8_t>* data = nullptr;

    [[nodiscard]] bool valid() const noexcept
    {
        return data != nullptr && !data->empty();
    }
};

static Q8Payload q8_payload(const TensorData& matrix,
                            std::shared_ptr<const QnKPack>* sidecar,
                            size_t rows,
                            uint32_t columns)
{
    Q8Payload payload;
    payload.pack = get_qnk_packed_weights(matrix, sidecar, rows, columns, QnKNeedQ8);
    if (!payload.pack)
        return payload;
    payload.data = !payload.pack->q8_compat.empty() ? &payload.pack->q8_compat : &payload.pack->q8_pack;
    return payload;
}

bool qnk_pack_weights(const uint8_t* raw,
                      size_t raw_bytes,
                      DType dtype,
                      size_t rows,
                      uint32_t columns,
                      QnKPack& output) noexcept
{
    const uint64_t expected = qnk_storage_bytes(dtype, rows, columns);
    const uint64_t packed_bytes = qnk_packed_storage_bytes(dtype, rows, columns);
    if (!raw || expected == 0 || expected != raw_bytes || packed_bytes == 0
        || packed_bytes > std::numeric_limits<size_t>::max())
        return false;
    const uint32_t block_count = columns / qnk_block_size(dtype);
    const size_t tile_count = (rows + 7) / 8;
    const size_t block_bytes = qnk_block_bytes(dtype);
    QnKPack packed;
    packed.dtype = dtype;
    packed.rows = rows;
    packed.columns = columns;
    packed.block_count = block_count;
    packed.storage.resize(static_cast<size_t>(packed_bytes), 0);
    for (size_t tile = 0; tile < tile_count; ++tile)
    {
        for (uint32_t block = 0; block < block_count; ++block)
        {
            for (uint32_t lane = 0; lane < 8; ++lane)
            {
                const size_t row = tile * 8 + lane;
                uint8_t* destination = packed.storage.data()
                                       + (((tile * block_count + block) * 8 + lane) * block_bytes);
                if (row < rows)
                {
                    const uint8_t* source = raw
                                            + ((row * static_cast<size_t>(block_count) + block) * block_bytes);
                    std::memcpy(destination, source, block_bytes);
                }
            }
        }
    }
    output = std::move(packed);
    return true;
}

static bool qnk_linear_impl(std::initializer_list<const TensorData*> matrices,
                            const ActivationBuffer& input,
                            std::initializer_list<ActivationBuffer*> outputs,
                            bool use_packed_weights,
                            std::span<std::shared_ptr<const QnKPack>* const> sidecars,
                            bool use_q8_activations,
                            QnkQ8Input* q8_input)
{
    const size_t matrix_count = matrices.size();
    const TensorData* const* weights = matrices.begin();
    ActivationBuffer* const* results = outputs.begin();
    bool convert_activations = input.dtype() == DType::BFloat16;
    for (size_t index = 0; index < matrix_count; ++index)
        convert_activations = convert_activations || results[index]->dtype() == DType::BFloat16;
    if (convert_activations)
    {
        // Only activations are widened. Immutable GGML weights and
        // optional packed sidecars retain their original native bytes.
        ActivationBuffer prepared;
        const ActivationBuffer* float_input = &input;
        if (input.dtype() == DType::BFloat16)
        {
            prepared.reset(input.rows(), input.columns(), false);
            for (size_t row = 0; row < input.rows(); ++row)
            {
                const auto source = input.row_bytes(row);
                for (uint32_t column = 0; column < input.columns(); ++column)
                {
                    uint16_t value;
                    std::memcpy(&value, source.data() + column * sizeof(value), sizeof(value));
                    prepared.row(row)[column] = bfloat16_to_float(value);
                }
            }
            float_input = &prepared;
        }
        std::array<ActivationBuffer, 2> projected;
        const bool success = matrix_count == 1
                                 ? qnk_linear_impl({weights[0]}, *float_input, {&projected[0]}, use_packed_weights, sidecars, use_q8_activations, q8_input)
                                 : qnk_linear_impl({weights[0], weights[1]}, *float_input, {&projected[0], &projected[1]}, use_packed_weights, sidecars, use_q8_activations, q8_input);
        if (!success)
            return false;
        for (size_t index = 0; index < matrix_count; ++index)
        {
            ActivationBuffer& destination = *results[index];
            if (destination.dtype() == DType::Float32)
                destination.swap(projected[index]);
            else
            {
                destination.reset(projected[index].rows(), projected[index].columns(), false);
                for (size_t row = 0; row < destination.rows(); ++row)
                {
                    const float* source = projected[index].row(row);
                    auto bytes = destination.mutable_row_bytes(row);
                    for (uint32_t column = 0; column < destination.columns(); ++column)
                    {
                        const uint16_t value = float_to_bfloat16(source[column]);
                        std::memcpy(bytes.data() + column * sizeof(value), &value, sizeof(value));
                    }
                }
            }
        }
        return true;
    }
    // Thread-local activation quantization reused across projections on this
    // thread; qnk_quantize_input keeps the buffer capacity between calls.
    static thread_local QnkQ8Input q8_scratch;
    // Shared quantization for this input, reusing the caller's buffer when it
    // matches and otherwise the thread-local scratch.
    const auto q8_get = [&](uint32_t columns) -> const QnkQ8Input& {
        if (q8_input != nullptr && q8_input->columns == columns
            && qnk_q8_input_matches(*q8_input, input))
            return *q8_input;
        qnk_quantize_input(input, q8_scratch);
        return q8_scratch;
    };
#if defined(__aarch64__) || defined(_M_ARM64)
    struct Q8PackScratch
    {
        std::vector<int8_t> pack;
        std::vector<float> scales;
    };
    static thread_local Q8PackScratch q8_pack_scratch;
#endif

    // View of the quantized activation row used by both prefill and decode.
    const auto q8_row = [&](uint32_t columns) {
        const QnkQ8Input& act = q8_get(columns);
        if (act.rows != input.rows() || act.columns != columns)
            return QnkQ8Row{};
        return QnkQ8Row{act.codes.data(), act.scales.data(), act.sums.data()};
    };

    if (matrix_count == 1)
    {
        const TensorData& matrix = *weights[0];
        const size_t rows = matrix.shape[0];
        const uint32_t columns = matrix.shape[1];
        const std::span<const uint8_t> raw = matrix.qnk_values();
        results[0]->reset(input.rows(), static_cast<uint32_t>(rows), false);
        if (input.rows() == 0)
            return true;
        const bool q8_dtype = matrix.dtype == DType::Q4_0 || matrix.dtype == DType::Q4_1
                              || matrix.dtype == DType::Q5_0 || matrix.dtype == DType::Q5_1
                              || matrix.dtype == DType::Q8_0 || matrix.dtype == DType::IQ4NL;
#if defined(__aarch64__) || defined(_M_ARM64)
        if (use_q8_activations && input.rows() > 1 && is_qnk_dtype(matrix.dtype)
            && qnk_block_size(matrix.dtype) != 32 && neon_qnk_q8_supported())
        {
            const Q8Payload fallback = q8_payload(matrix, sidecars.empty() ? nullptr : sidecars[0], rows, columns);
            if (fallback.valid())
            {
                const QnkQ8Row view = q8_row(columns);
                if (view.codes != nullptr)
                {
                    if (rows % 8 == 0)
                        neon_qnk_prefill_q8_0_rows8(fallback.data->data(), rows, columns,
                                                    view.codes, view.scales, input.rows(),
                                                    results[0]->row(0), results[0]->columns());
                    else
                        neon_qnk_prefill_q8_0(fallback.data->data(), rows, columns,
                                              view.codes, view.scales, input.rows(),
                                              results[0]->row(0), results[0]->columns());
                    return true;
                }
            }
        }
        if (use_q8_activations && input.rows() > 1 && q8_dtype
            && neon_qnk_q8_supported())
        {
            if (matrix.dtype == DType::Q8_0 && rows % 4 == 0 && input.rows() % 4 == 0
                && neon_qnk_i8mm_supported())
            {
                const Q8Payload weight_pack = q8_payload(matrix, sidecars.empty() ? nullptr : sidecars[0], rows, columns);
                if (weight_pack.valid())
                {
                    Q8PackScratch& scratch = q8_pack_scratch;
                    scratch.pack.resize(input.rows() * static_cast<std::size_t>(columns));
                    scratch.scales.resize(input.rows() * static_cast<std::size_t>(columns / 32));
                    neon_qnk_q8_quant_pack4(input.row(0), input.rows(), columns,
                                            scratch.pack.data(), scratch.scales.data());
                    const std::vector<int8_t>& activation_pack = scratch.pack;
                    const std::vector<float>& activation_scales = scratch.scales;
                    if (input.rows() % 8 == 0)
                        neon_qnk_prefill_q8_0_gemm8(weight_pack.data->data(), rows, columns,
                                                    reinterpret_cast<const uint8_t*>(activation_pack.data()),
                                                    activation_scales.data(), input.rows(),
                                                    results[0]->row(0), results[0]->columns());
                    else
                        neon_qnk_prefill_q8_0_gemm4(weight_pack.data->data(), rows, columns,
                                                    reinterpret_cast<const uint8_t*>(activation_pack.data()),
                                                    activation_scales.data(), input.rows(),
                                                    results[0]->row(0), results[0]->columns());
                    return true;
                }
            }
            const QnkQ8Row view = q8_row(columns);
            if (view.codes != nullptr)
            {
                if (matrix.dtype == DType::Q8_0)
                {
                    if (rows % 8 == 0)
                        neon_qnk_prefill_q8_0_rows8(raw.data(), rows, columns,
                                                    view.codes, view.scales, input.rows(),
                                                    results[0]->row(0), results[0]->columns());
                    else
                        neon_qnk_prefill_q8_0(raw.data(), rows, columns,
                                              view.codes, view.scales, input.rows(),
                                              results[0]->row(0), results[0]->columns());
                }
                else if (rows % 8 == 0)
                    neon_qnk_prefill_q8_rows8(matrix.dtype, raw.data(), rows, columns,
                                              view.codes, view.scales, view.sums, input.rows(),
                                              results[0]->row(0), results[0]->columns());
                else
                    neon_qnk_prefill_q8(matrix.dtype, raw.data(), rows, columns,
                                        view.codes, view.scales, view.sums, input.rows(),
                                        results[0]->row(0), results[0]->columns());
                return true;
            }
        }
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
        if (use_q8_activations && input.rows() == 1 && is_qnk_dtype(matrix.dtype)
            && qnk_block_size(matrix.dtype) != 32 && neon_qnk_q8_supported())
        {
            const Q8Payload fallback = q8_payload(matrix, sidecars.empty() ? nullptr : sidecars[0], rows, columns);
            if (fallback.valid())
            {
                const QnkQ8Row q8 = q8_row(columns);
                if (q8.codes != nullptr)
                {
                    qnk_gemm<false>(fallback.data->data(), nullptr, DType::Q8_0, rows, columns,
                                    input.row(0), columns, 1,
                                    results[0]->row(0), results[0]->columns(),
                                    nullptr, nullptr, nullptr, &q8);
                    return true;
                }
            }
        }
#endif
        QnkQ8Row q8;
        bool q8_enabled = false;
#if !defined(NCNN_MOE_MSVC_X86_SIMD) && !defined(__aarch64__) && !defined(_M_ARM64)
        static_cast<void>(q8_dtype);
#endif
#if defined(NCNN_MOE_MSVC_X86_SIMD)
        // Single-token decode only. One quantized activation row serves every
        // output row, and qnk_gemm keeps the threading and packed-weight policy,
        // so this function no longer needs its own row loop.
        q8_enabled = use_q8_activations && input.rows() == 1
                     && qnk_block_size(matrix.dtype) == 32 && q8_dtype
                     && (mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx2
                         || mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx512);
        if (q8_enabled)
        {
            q8 = q8_row(columns);
            q8_enabled = q8.codes != nullptr;
        }
#elif defined(__aarch64__) || defined(_M_ARM64)
        q8_enabled = use_q8_activations && input.rows() == 1
                     && qnk_block_size(matrix.dtype) == 32 && q8_dtype
                     && neon_qnk_q8_supported();
        if (q8_enabled)
        {
            q8 = q8_row(columns);
            q8_enabled = q8.codes != nullptr;
        }
#endif
        if (!use_packed_weights)
        {
            qnk_gemm<false>(raw.data(), nullptr, matrix.dtype, rows, columns,
                            input.row(0), columns, input.rows(), results[0]->row(0), results[0]->columns(),
                            nullptr, nullptr, nullptr, q8_enabled ? &q8 : nullptr);
            return true;
        }
        const std::shared_ptr<const QnKPack> packed = get_qnk_packed_weights(matrix,
                                                                             sidecars.empty() ? nullptr : sidecars[0], rows, columns,
                                                                             QnKNeedNative);
        if (!packed)
            return false;

#if defined(NCNN_MOE_MSVC_X86_SIMD)
        if (qnk_block_size(matrix.dtype) == qnk_block_elements && matrix.dtype != DType::IQ4XS
            && mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx512)
        {
            msvc_avx512_qnk_gemm(*packed, input.row(0), columns, input.rows(),
                                 results[0]->row(0), results[0]->columns());
            return true;
        }
        if (qnk_block_size(matrix.dtype) == qnk_block_elements && matrix.dtype != DType::IQ4XS
            && mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx2)
        {
            msvc_avx2_qnk_gemm(*packed, input.row(0), columns, input.rows(),
                               results[0]->row(0), results[0]->columns());
            return true;
        }
#endif
        qnk_gemm<false>(nullptr, packed.get(), matrix.dtype, rows, columns,
                        input.row(0), columns, input.rows(), results[0]->row(0), results[0]->columns(),
                        nullptr, nullptr, nullptr, q8_enabled ? &q8 : nullptr);
        return true;
    }

    const TensorData& matrix0 = *weights[0];
    const TensorData& matrix1 = *weights[1];
    const size_t rows0 = matrix0.shape[0];
    const size_t rows1 = matrix1.shape[0];
    const uint32_t columns = matrix0.shape[1];
    const std::span<const uint8_t> raw0 = matrix0.qnk_values();
    const std::span<const uint8_t> raw1 = matrix1.qnk_values();

    if (input.rows() == 0)
    {
        results[0]->reset(0, static_cast<uint32_t>(rows0), false);
        results[1]->reset(0, static_cast<uint32_t>(rows1), false);
        return true;
    }

    std::shared_ptr<const QnKPack> packed0;
    std::shared_ptr<const QnKPack> packed1;
    if (use_packed_weights)
    {
        packed0 = get_qnk_packed_weights(matrix0, sidecars.empty() ? nullptr : sidecars[0], rows0, columns, QnKNeedNative);
        if (!packed0)
            return false;
        packed1 = get_qnk_packed_weights(matrix1, sidecars.empty() ? nullptr : sidecars[1], rows1, columns, QnKNeedNative);
        if (!packed1)
            return false;
    }
    results[0]->reset(input.rows(), static_cast<uint32_t>(rows0), false);
    results[1]->reset(input.rows(), static_cast<uint32_t>(rows1), false);
    const bool q8_dtype = (matrix0.dtype == DType::Q4_0 || matrix0.dtype == DType::Q4_1
                           || matrix0.dtype == DType::Q5_0 || matrix0.dtype == DType::Q5_1
                           || matrix0.dtype == DType::Q8_0 || matrix0.dtype == DType::IQ4NL)
                          && qnk_block_size(matrix0.dtype) == 32;
#if defined(__aarch64__) || defined(_M_ARM64)
    if (use_q8_activations && input.rows() > 1 && is_qnk_dtype(matrix0.dtype)
        && qnk_block_size(matrix0.dtype) != 32 && neon_qnk_q8_supported())
    {
        const Q8Payload fallback0 = q8_payload(matrix0, sidecars.empty() ? nullptr : sidecars[0], rows0, columns);
        const Q8Payload fallback1 = q8_payload(matrix1, sidecars.empty() ? nullptr : sidecars[1], rows1, columns);
        if (fallback0.valid() && fallback1.valid())
        {
            const QnkQ8Row view = q8_row(columns);
            if (view.codes != nullptr)
            {
                if (rows0 % 8 == 0 && rows1 % 8 == 0)
                {
                    neon_qnk_prefill_q8_0_rows8(fallback0.data->data(), rows0, columns,
                                                view.codes, view.scales, input.rows(),
                                                results[0]->row(0), results[0]->columns());
                    neon_qnk_prefill_q8_0_rows8(fallback1.data->data(), rows1, columns,
                                                view.codes, view.scales, input.rows(),
                                                results[1]->row(0), results[1]->columns());
                }
                else
                {
                    neon_qnk_prefill_q8_0(fallback0.data->data(), rows0, columns,
                                          view.codes, view.scales, input.rows(),
                                          results[0]->row(0), results[0]->columns());
                    neon_qnk_prefill_q8_0(fallback1.data->data(), rows1, columns,
                                          view.codes, view.scales, input.rows(),
                                          results[1]->row(0), results[1]->columns());
                }
                return true;
            }
        }
    }
    if (use_q8_activations && input.rows() > 1 && q8_dtype
        && neon_qnk_q8_supported())
    {
        if (matrix0.dtype == DType::Q8_0 && rows0 % 4 == 0 && rows1 % 4 == 0
            && input.rows() % 4 == 0 && neon_qnk_i8mm_supported())
        {
            const Q8Payload weight_pack0 = q8_payload(matrix0, sidecars.empty() ? nullptr : sidecars[0], rows0, columns);
            const Q8Payload weight_pack1 = q8_payload(matrix1, sidecars.empty() ? nullptr : sidecars[1], rows1, columns);
            if (weight_pack0.valid() && weight_pack1.valid())
            {
                Q8PackScratch& scratch = q8_pack_scratch;
                scratch.pack.resize(input.rows() * static_cast<std::size_t>(columns));
                scratch.scales.resize(input.rows() * static_cast<std::size_t>(columns / 32));
                neon_qnk_q8_quant_pack4(input.row(0), input.rows(), columns,
                                        scratch.pack.data(), scratch.scales.data());
                const std::vector<int8_t>& activation_pack = scratch.pack;
                const std::vector<float>& activation_scales = scratch.scales;
                if (input.rows() % 16 == 0)
                {
                    neon_qnk_prefill_q8_0_gemm16(weight_pack0.data->data(), rows0, columns,
                                                 reinterpret_cast<const uint8_t*>(activation_pack.data()),
                                                 activation_scales.data(), input.rows(),
                                                 results[0]->row(0), results[0]->columns());
                    neon_qnk_prefill_q8_0_gemm16(weight_pack1.data->data(), rows1, columns,
                                                 reinterpret_cast<const uint8_t*>(activation_pack.data()),
                                                 activation_scales.data(), input.rows(),
                                                 results[1]->row(0), results[1]->columns());
                }
                else if (input.rows() % 8 == 0)
                {
                    neon_qnk_prefill_q8_0_gemm8(weight_pack0.data->data(), rows0, columns,
                                                reinterpret_cast<const uint8_t*>(activation_pack.data()),
                                                activation_scales.data(), input.rows(),
                                                results[0]->row(0), results[0]->columns());
                    neon_qnk_prefill_q8_0_gemm8(weight_pack1.data->data(), rows1, columns,
                                                reinterpret_cast<const uint8_t*>(activation_pack.data()),
                                                activation_scales.data(), input.rows(),
                                                results[1]->row(0), results[1]->columns());
                }
                else
                {
                    neon_qnk_prefill_q8_0_gemm4(weight_pack0.data->data(), rows0, columns,
                                                reinterpret_cast<const uint8_t*>(activation_pack.data()),
                                                activation_scales.data(), input.rows(),
                                                results[0]->row(0), results[0]->columns());
                    neon_qnk_prefill_q8_0_gemm4(weight_pack1.data->data(), rows1, columns,
                                                reinterpret_cast<const uint8_t*>(activation_pack.data()),
                                                activation_scales.data(), input.rows(),
                                                results[1]->row(0), results[1]->columns());
                }
                return true;
            }
        }
        const QnkQ8Row view = q8_row(columns);
        if (view.codes != nullptr)
        {
            if (matrix0.dtype == DType::Q8_0)
            {
                if (rows0 % 8 == 0 && rows1 % 8 == 0)
                {
                    neon_qnk_prefill_q8_0_rows8(raw0.data(), rows0, columns,
                                                view.codes, view.scales, input.rows(),
                                                results[0]->row(0), results[0]->columns());
                    neon_qnk_prefill_q8_0_rows8(raw1.data(), rows1, columns,
                                                view.codes, view.scales, input.rows(),
                                                results[1]->row(0), results[1]->columns());
                }
                else
                {
                    neon_qnk_prefill_q8_0(raw0.data(), rows0, columns,
                                          view.codes, view.scales, input.rows(),
                                          results[0]->row(0), results[0]->columns());
                    neon_qnk_prefill_q8_0(raw1.data(), rows1, columns,
                                          view.codes, view.scales, input.rows(),
                                          results[1]->row(0), results[1]->columns());
                }
            }
            else
            {
                if (rows0 % 8 == 0 && rows1 % 8 == 0)
                {
                    neon_qnk_prefill_q8_rows8(matrix0.dtype, raw0.data(), rows0, columns,
                                              view.codes, view.scales, view.sums, input.rows(),
                                              results[0]->row(0), results[0]->columns());
                    neon_qnk_prefill_q8_rows8(matrix1.dtype, raw1.data(), rows1, columns,
                                              view.codes, view.scales, view.sums, input.rows(),
                                              results[1]->row(0), results[1]->columns());
                }
                else
                {
                    neon_qnk_prefill_q8(matrix0.dtype, raw0.data(), rows0, columns,
                                        view.codes, view.scales, view.sums, input.rows(),
                                        results[0]->row(0), results[0]->columns());
                    neon_qnk_prefill_q8(matrix1.dtype, raw1.data(), rows1, columns,
                                        view.codes, view.scales, view.sums, input.rows(),
                                        results[1]->row(0), results[1]->columns());
                }
            }
            return true;
        }
    }
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
    if (use_q8_activations && input.rows() == 1 && is_qnk_dtype(matrix0.dtype)
        && qnk_block_size(matrix0.dtype) != 32 && neon_qnk_q8_supported())
    {
        const Q8Payload fallback0 = q8_payload(matrix0, sidecars.empty() ? nullptr : sidecars[0], rows0, columns);
        const Q8Payload fallback1 = q8_payload(matrix1, sidecars.empty() ? nullptr : sidecars[1], rows1, columns);
        if (fallback0.valid() && fallback1.valid())
        {
            const QnkQ8Row q8 = q8_row(columns);
            if (q8.codes != nullptr)
            {
                qnk_gemm<true>(fallback0.data->data(), nullptr, DType::Q8_0, rows0, columns,
                               input.row(0), columns, 1,
                               results[0]->row(0), results[0]->columns(),
                               &matrix1, nullptr, results[1], &q8, fallback1.data->data());
                return true;
            }
        }
    }
#endif
    QnkQ8Row q8;
    bool q8_enabled = false;
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    q8_enabled = use_q8_activations && input.rows() == 1 && q8_dtype
                 && (mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx2
                     || mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx512);
    if (q8_enabled)
    {
        q8 = q8_row(columns);
        q8_enabled = q8.codes != nullptr;
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    q8_enabled = use_q8_activations && input.rows() == 1 && q8_dtype
                 && neon_qnk_q8_supported();
    if (q8_enabled)
    {
        q8 = q8_row(columns);
        q8_enabled = q8.codes != nullptr;
    }
#else
    static_cast<void>(q8_dtype);
#endif
    qnk_gemm<true>(raw0.data(), packed0.get(), matrix0.dtype, rows0, columns,
                   input.row(0), columns, input.rows(), results[0]->row(0), results[0]->columns(),
                   &matrix1, packed1.get(), results[1], q8_enabled ? &q8 : nullptr);
    return true;
}

bool forward_linear_qnk(std::initializer_list<const TensorData*> matrices,
                        const ActivationBuffer& input,
                        std::initializer_list<ActivationBuffer*> outputs,
                        bool use_packed_weights,
                        std::span<std::shared_ptr<const QnKPack>* const> sidecars,
                        bool use_q8_activations,
                        QnkQ8Input* q8_input)
{
    return qnk_linear_impl(matrices, input, outputs, use_packed_weights, sidecars, use_q8_activations, q8_input);
}

void qnk_quantize_input(const ActivationBuffer& input, QnkQ8Input& quantized) noexcept
{
    quantized.source = nullptr;
    quantized.rows = 0;
    quantized.columns = 0;
#if defined(__aarch64__) || defined(_M_ARM64)
    const bool supported = neon_qnk_q8_supported();
#elif defined(NCNN_MOE_MSVC_X86_SIMD)
    const bool supported = mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx2
                           || mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx512;
#else
    const bool supported = false;
#endif
    if (!supported || input.dtype() != DType::Float32
        || input.rows() == 0 || input.columns() == 0 || input.columns() % 32 != 0)
        return;
    try
    {
        const uint32_t block_count = input.columns() / 32;
        quantized.codes.resize(input.rows() * input.columns());
        quantized.scales.resize(input.rows() * block_count);
        quantized.sums.resize(input.rows() * block_count);
        int8_t* codes = quantized.codes.data();
        float* scales = quantized.scales.data();
        int32_t* sums = quantized.sums.data();
#if defined(_OPENMP)
        const int team_size = omp_in_parallel() != 0
                                  ? 1
                                  : static_cast<int>(std::min<size_t>(input.rows(), static_cast<size_t>(cpu_linear_num_threads())));
#else
        const int team_size = 1;
#endif
#pragma omp parallel for schedule(static) num_threads(team_size) if (team_size > 1)
        for (int64_t token_index = 0; token_index < static_cast<int64_t>(input.rows()); ++token_index)
        {
            const auto token = static_cast<std::size_t>(token_index);
            int8_t* token_codes = codes + token * static_cast<std::size_t>(input.columns());
            float* token_scales = scales + token * static_cast<std::size_t>(block_count);
            int32_t* token_sums = sums + token * static_cast<std::size_t>(block_count);
#if defined(__aarch64__) || defined(_M_ARM64)
            neon_qnk_q8_quant(input.row(token), token_codes, token_scales, token_sums, input.columns());
#else
            msvc_avx2_qnk_q8_quant(input.row(token), token_codes, token_scales, token_sums, input.columns());
#endif
        }
        quantized.source = input.row(0);
        quantized.rows = input.rows();
        quantized.columns = input.columns();
    }
    catch (...)
    {
        quantized.source = nullptr;
        quantized.rows = 0;
        quantized.columns = 0;
    }
}

} // namespace moe
} // namespace ncnn
